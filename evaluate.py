#!/usr/bin/env python3
"""
Script para evaluar modelos NLLB fine-tuneados en datos de prueba
Uso: python evaluate.py --model_path runs/best_model --dataset_version v1 --direction es2agr
"""

import argparse
import yaml
import os
import json
import pandas as pd
from src.inference import NLLBPredictor
from src.dataset import AwajunDataLoader
from src.evaluation import TranslationEvaluator
from src.utils import setup_logging, format_time
import time

def load_config(config_path="config.yaml"):
    """Cargar configuración desde YAML"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def parse_args():
    """Argumentos de línea de comandos"""
    parser = argparse.ArgumentParser(description="Evaluación de modelos NLLB fine-tuneados")
    
    # Parámetros principales
    parser.add_argument('--model_path', type=str, required=True,
                       help='Ruta al modelo fine-tuneado')
    parser.add_argument('--direction', type=str, required=True, 
                       choices=['es2agr', 'agr2es'], 
                       help='Dirección de traducción')
    parser.add_argument('--dataset_version', type=str, default='v1',
                       choices=['v1', 'v2'],
                       help='Versión del dataset')
    
    # Datos a evaluar
    parser.add_argument('--eval_split', type=str, default='dev',
                       choices=['dev', 'test'],
                       help='Split a evaluar')
    parser.add_argument('--sample_size', type=int, default=None,
                       help='Número de muestras (None = todo el dataset)')
    
    # Configuración
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Archivo de configuración')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Directorio de resultados')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Tamaño de batch')
    
    # Opciones de evaluación
    parser.add_argument('--save_predictions', action='store_true',
                       help='Guardar todas las predicciones')
    parser.add_argument('--detailed_analysis', action='store_true',
                       help='Análisis detallado por dominio')
    parser.add_argument('--sample_translations', type=int, default=10,
                       help='Número de traducciones de muestra')
    
    return parser.parse_args()

class ModelEvaluator:
    """Evaluador completo de modelos"""
    
    def __init__(self, model_path, direction, dataset_version, config, output_dir):
        self.model_path = model_path
        self.direction = direction
        self.dataset_version = dataset_version
        self.config = config
        self.output_dir = output_dir
        
        # Crear directorio de salida
        os.makedirs(output_dir, exist_ok=True)
        
        # Configurar idiomas
        if direction == 'es2agr':
            self.src_lang, self.tgt_lang = 'es', 'agr'
        else:
            self.src_lang, self.tgt_lang = 'agr', 'es'
        
        # Cargar predictor
        self.predictor = NLLBPredictor(
            model_path=model_path,
            direction=direction,
            config=config
        )
        
        # Cargar datos
        self.config['data']['dataset_version'] = dataset_version
        self.data_loader = AwajunDataLoader(self.config)
        
    def load_evaluation_data(self, split='dev', sample_size=None):
        """Cargar datos para evaluación"""
        if split == 'dev':
            df_train, df_eval = self.data_loader.load_data()
        else:
            # Para test, asumir que existe test.agr, test.es, test.source
            try:
                df_eval = self.data_loader.load_parallel_files('test')
            except FileNotFoundError:
                print("Archivo de test no encontrado, usando dev")
                df_train, df_eval = self.data_loader.load_data()
        
        # Aplicar sample_size si se especifica
        if sample_size and sample_size < len(df_eval):
            df_eval = df_eval.sample(sample_size, random_state=42)
            print(f"Usando muestra de {sample_size} ejemplos")
        
        return df_eval
    
    def evaluate_model(self, df_eval, batch_size=16, save_predictions=False):
        """Evaluar modelo en dataset"""
        print(f"Evaluando modelo en {len(df_eval)} ejemplos...")
        
        start_time = time.time()
        
        # Generar predicciones
        sources = df_eval[self.src_lang].tolist()
        references = df_eval[self.tgt_lang].tolist()
        
        predictions = self.predictor.translate_batch(
            sources, 
            batch_size=batch_size, 
            show_progress=True
        )
        
        # Calcular métricas usando sacrebleu
        from sacrebleu.metrics import CHRF, BLEU
        
        chrf = CHRF(word_order=2)
        bleu = BLEU()
        
        chrf_score = chrf.corpus_score(predictions, [references]).score
        bleu_score = bleu.corpus_score(predictions, [references]).score
        
        # Métricas adicionales
        exact_matches = sum(1 for p, r in zip(predictions, references) 
                          if p.strip().lower() == r.strip().lower())
        
        elapsed = time.time() - start_time
        
        results = {
            'model_path': self.model_path,
            'direction': self.direction,
            'dataset_version': self.dataset_version,
            'eval_samples': len(df_eval),
            'chrf_score': chrf_score,
            'bleu_score': bleu_score,
            'exact_matches': exact_matches,
            'exact_match_rate': exact_matches / len(df_eval) * 100,
            'avg_src_length': sum(len(s) for s in sources) / len(sources),
            'avg_tgt_length': sum(len(r) for r in references) / len(references),
            'avg_pred_length': sum(len(p) for p in predictions) / len(predictions),
            'evaluation_time': elapsed,
            'samples_per_second': len(df_eval) / elapsed
        }
        
        # Guardar predicciones si se solicita
        if save_predictions:
            predictions_df = pd.DataFrame({
                'source': sources,
                'reference': references,
                'prediction': predictions,
                'domain': df_eval.get('source', ['unknown'] * len(df_eval))
            })
            
            pred_file = os.path.join(self.output_dir, 'predictions.csv')
            predictions_df.to_csv(pred_file, index=False)
            results['predictions_file'] = pred_file
        
        return results, predictions
    
    def analyze_by_domain(self, df_eval, predictions):
        """Análisis detallado por dominio"""
        if 'source' not in df_eval.columns:
            return {}
        
        domain_results = {}
        
        for domain in df_eval['source'].unique():
            domain_mask = df_eval['source'] == domain
            domain_df = df_eval[domain_mask]
            domain_preds = [predictions[i] for i in domain_df.index]
            domain_refs = domain_df[self.tgt_lang].tolist()
            
            # Calcular métricas por dominio
            from sacrebleu.metrics import CHRF, BLEU
            
            chrf = CHRF(word_order=2)
            bleu = BLEU()
            
            domain_chrf = chrf.corpus_score(domain_preds, [domain_refs]).score
            domain_bleu = bleu.corpus_score(domain_preds, [domain_refs]).score
            
            domain_results[domain] = {
                'samples': len(domain_df),
                'chrf': domain_chrf,
                'bleu': domain_bleu,
                'avg_src_length': domain_df[self.src_lang].str.len().mean(),
                'avg_tgt_length': domain_df[self.tgt_lang].str.len().mean(),
                'avg_pred_length': sum(len(p) for p in domain_preds) / len(domain_preds)
            }
        
        return domain_results
    
    def get_sample_translations(self, df_eval, predictions, n_samples=10):
        """Obtener muestras de traducciones para inspección"""
        if n_samples > len(df_eval):
            n_samples = len(df_eval)
        
        # Seleccionar muestras diversas
        sample_indices = []
        
        # Por dominio si está disponible
        if 'source' in df_eval.columns:
            domains = df_eval['source'].unique()
            samples_per_domain = max(1, n_samples // len(domains))
            
            for domain in domains:
                domain_indices = df_eval[df_eval['source'] == domain].index.tolist()
                selected = pd.Series(domain_indices).sample(
                    min(samples_per_domain, len(domain_indices)), 
                    random_state=42
                ).tolist()
                sample_indices.extend(selected)
        else:
            sample_indices = df_eval.sample(n_samples, random_state=42).index.tolist()
        
        # Limitar al número solicitado
        sample_indices = sample_indices[:n_samples]
        
        samples = []
        for idx in sample_indices:
            samples.append({
                'source': df_eval.loc[idx, self.src_lang],
                'reference': df_eval.loc[idx, self.tgt_lang],
                'prediction': predictions[list(df_eval.index).index(idx)],
                'domain': df_eval.loc[idx, 'source'] if 'source' in df_eval.columns else 'unknown'
            })
        
        return samples
    
    def save_results(self, results, domain_results, sample_translations):
        """Guardar resultados de evaluación"""
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        
        # Archivo principal de resultados
        results_file = os.path.join(self.output_dir, f'evaluation_results_{timestamp}.json')
        
        full_results = {
            'main_results': results,
            'domain_analysis': domain_results,
            'sample_translations': sample_translations,
            'model_info': self.predictor.get_model_info(),
            'timestamp': timestamp
        }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(full_results, f, indent=2, ensure_ascii=False)
        
        # Resumen en texto
        summary_file = os.path.join(self.output_dir, f'evaluation_summary_{timestamp}.txt')
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"Evaluación de Modelo NLLB\n")
            f.write(f"========================\n\n")
            f.write(f"Modelo: {results['model_path']}\n")
            f.write(f"Dirección: {results['direction']}\n")
            f.write(f"Dataset: {results['dataset_version']}\n")
            f.write(f"Muestras: {results['eval_samples']}\n\n")
            
            f.write(f"Resultados Principales:\n")
            f.write(f"----------------------\n")
            f.write(f"CHRF++: {results['chrf_score']:.2f}\n")
            f.write(f"BLEU: {results['bleu_score']:.2f}\n")
            f.write(f"Matches exactos: {results['exact_matches']}/{results['eval_samples']} ({results['exact_match_rate']:.1f}%)\n")
            f.write(f"Tiempo: {format_time(results['evaluation_time'])}\n")
            f.write(f"Velocidad: {results['samples_per_second']:.1f} samples/sec\n\n")
            
            if domain_results:
                f.write(f"Análisis por Dominio:\n")
                f.write(f"--------------------\n")
                for domain, metrics in domain_results.items():
                    f.write(f"{domain}: CHRF={metrics['chrf']:.2f}, BLEU={metrics['bleu']:.2f}, N={metrics['samples']}\n")
                f.write(f"\n")
            
            f.write(f"Ejemplos de Traducción:\n")
            f.write(f"----------------------\n")
            for i, sample in enumerate(sample_translations, 1):
                f.write(f"Ejemplo {i} ({sample['domain']}):\n")
                f.write(f"  Origen: {sample['source']}\n")
                f.write(f"  Referencia: {sample['reference']}\n")
                f.write(f"  Predicción: {sample['prediction']}\n\n")
        
        return results_file, summary_file

def main():
    """Función principal"""
    args = parse_args()
    
    # Setup logging
    setup_logging()
    
    # Verificar que el modelo existe
    if not os.path.exists(args.model_path):
        print(f"Error: Modelo no encontrado en {args.model_path}")
        return
    
    # Cargar configuración
    config = load_config(args.config)
    
    # Crear evaluador
    print(f"Iniciando evaluación de modelo...")
    print(f"Modelo: {args.model_path}")
    print(f"Dirección: {args.direction}")
    print(f"Dataset: {args.dataset_version}")
    
    evaluator = ModelEvaluator(
        model_path=args.model_path,
        direction=args.direction,
        dataset_version=args.dataset_version,
        config=config,
        output_dir=args.output_dir
    )
    
    # Cargar datos de evaluación
    df_eval = evaluator.load_evaluation_data(
        split=args.eval_split, 
        sample_size=args.sample_size
    )
    
    print(f"Datos cargados: {len(df_eval)} ejemplos")
    
    # Evaluar modelo
    results, predictions = evaluator.evaluate_model(
        df_eval,
        batch_size=args.batch_size,
        save_predictions=args.save_predictions
    )
    
    # Análisis por dominio si se solicita
    domain_results = {}
    if args.detailed_analysis:
        print("Realizando análisis detallado por dominio...")
        domain_results = evaluator.analyze_by_domain(df_eval, predictions)
    
    # Muestras de traducción
    sample_translations = evaluator.get_sample_translations(
        df_eval, predictions, args.sample_translations
    )
    
    # Guardar resultados
    results_file, summary_file = evaluator.save_results(
        results, domain_results, sample_translations
    )
    
    # Mostrar resultados
    print(f"\nResultados de Evaluación:")
    print(f"========================")
    print(f"CHRF++: {results['chrf_score']:.2f}")
    print(f"BLEU: {results['bleu_score']:.2f}")
    print(f"Matches exactos: {results['exact_matches']}/{results['eval_samples']} ({results['exact_match_rate']:.1f}%)")
    print(f"Tiempo: {format_time(results['evaluation_time'])}")
    print(f"Velocidad: {results['samples_per_second']:.1f} samples/sec")
    
    if domain_results:
        print(f"\nAnálisis por Dominio:")
        for domain, metrics in domain_results.items():
            print(f"  {domain}: CHRF={metrics['chrf']:.2f}, BLEU={metrics['bleu']:.2f}, N={metrics['samples']}")
    
    print(f"\nArchivos generados:")
    print(f"  Resultados: {results_file}")
    print(f"  Resumen: {summary_file}")
    if args.save_predictions:
        print(f"  Predicciones: {results.get('predictions_file', 'N/A')}")
    
    print(f"\nEjemplos de traducción:")
    for i, sample in enumerate(sample_translations[:3], 1):
        print(f"  {i}. [{sample['domain']}] {sample['source'][:50]}...")
        print(f"     → {sample['prediction'][:50]}...")

if __name__ == "__main__":
    main()