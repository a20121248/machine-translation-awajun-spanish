"""
Evaluación de modelos de traducción con métricas CHRF++, BLEU
"""

import torch
import logging
from sacrebleu.metrics import CHRF, BLEU
from tqdm import tqdm

logger = logging.getLogger(__name__)

class TranslationEvaluator:
    """Evaluador de traducciones con múltiples métricas"""
    
    def __init__(self, model_wrapper, config):
        self.model = model_wrapper
        self.config = config
        self.metrics = {
            'chrf': CHRF(word_order=2),
            'bleu': BLEU()
        }
        
    def evaluate_model(self, df_eval, src_lang, tgt_lang, src_token, tgt_token):
        """Evaluar modelo completo en dataset"""
        logger.info(f"🔍 Evaluando modelo: {src_lang} → {tgt_lang}")
        
        # Usar todo el dataset por defecto, o muestra si se especifica
        sample_size = self.config['evaluation']['eval_sample_size']
        
        if sample_size is None:
            # Usar todo el dataset
            eval_df = df_eval
            logger.info(f"📊 Evaluando en dataset completo: {len(eval_df)} ejemplos")
        else:
            # Usar muestra específica
            sample_size = min(sample_size, len(df_eval))
            eval_df = df_eval.sample(sample_size, random_state=42)
            logger.info(f"📊 Evaluando en muestra de {sample_size} ejemplos")
        
        # Generar traducciones
        predictions = []
        references = eval_df[tgt_lang].tolist()
        
        self.model.model.eval()
        
        with torch.no_grad():
            for src_text in tqdm(eval_df[src_lang], desc="Generando traducciones"):
                try:
                    prediction = self.model.generate_translation(
                        src_text, src_token
                    )
                    predictions.append(prediction)
                    
                    # Limpiar cache de GPU periodicamente
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                except Exception as e:
                    logger.warning(f"Error en traducción: {e}")
                    predictions.append("")  # Traducción vacía como fallback
        
        # Calcular métricas
        results = {}
        
        for metric_name in self.config['evaluation']['metrics']:
            if metric_name in self.metrics:
                try:
                    if metric_name == 'chrf':
                        score = self.metrics[metric_name].corpus_score(
                            predictions, [references]
                        ).score
                    else:  # bleu
                        score = self.metrics[metric_name].corpus_score(
                            predictions, [references]
                        ).score
                    
                    results[f"eval_{metric_name}"] = score
                    
                except Exception as e:
                    logger.warning(f"Error calculando {metric_name}: {e}")
                    results[f"eval_{metric_name}"] = 0.0
        
        # Información adicional
        results['eval_samples'] = len(eval_df)
        results['avg_input_length'] = eval_df[src_lang].str.len().mean()
        results['avg_output_length'] = sum(len(p) for p in predictions) / len(predictions)
        
        # Mostrar resultados
        metric_str = " | ".join([
            f"{k.replace('eval_', '').upper()}: {v:.2f}"
            for k, v in results.items()
            if k.startswith('eval_') and isinstance(v, (int, float))
        ])
        logger.info(f"📈 Resultados: {metric_str}")
        
        return results
    
    def get_sample_translations(self, df_eval, src_lang, tgt_lang, src_token, num_samples=5):
        """Obtener traducciones de muestra para inspección"""
        sample_df = df_eval.sample(min(num_samples, len(df_eval)), random_state=42)
        
        samples = []
        self.model.model.eval()
        
        with torch.no_grad():
            for _, row in sample_df.iterrows():
                src_text = row[src_lang]
                tgt_text = row[tgt_lang]
                
                try:
                    prediction = self.model.generate_translation(src_text, src_token)
                    
                    samples.append({
                        'source': src_text,
                        'reference': tgt_text,
                        'prediction': prediction,
                        'domain': row.get('source', 'unknown')
                    })
                    
                except Exception as e:
                    logger.warning(f"Error generando muestra: {e}")
        
        return samples
    
    def log_sample_translations(self, samples, epoch):
        """Loggear traducciones de muestra"""
        logger.info(f"📝 Traducciones de muestra - Época {epoch}:")
        
        for i, sample in enumerate(samples[:3], 1):
            logger.info(f"  Ejemplo {i} ({sample['domain']}):")
            logger.info(f"    Origen: {sample['source'][:100]}...")
            logger.info(f"    Referencia: {sample['reference'][:100]}...")
            logger.info(f"    Predicción: {sample['prediction'][:100]}...")
            logger.info("")
    
    def calculate_convergence_metrics(self, history):
        """Calcular métricas de convergencia"""
        if len(history) < 2:
            return {}
        
        # Encontrar mejor época
        best_epoch = 0
        best_score = 0
        
        for epoch, metrics in enumerate(history):
            if 'eval_chrf' in metrics and metrics['eval_chrf'] > best_score:
                best_score = metrics['eval_chrf']
                best_epoch = epoch
        
        # Calcular estabilidad (varianza en últimas 3 épocas)
        recent_scores = [
            h.get('eval_chrf', 0) for h in history[-3:]
        ]
        stability = 1.0 / (1.0 + sum((s - sum(recent_scores)/len(recent_scores))**2 for s in recent_scores))
        
        return {
            'best_epoch': best_epoch,
            'best_chrf': best_score,
            'stability': stability,
            'improvement_rate': (best_score - history[0].get('eval_chrf', 0)) / max(1, best_epoch)
        }