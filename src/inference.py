"""
Sistema de inferencia para modelos NLLB fine-tuneados
"""

import torch
import logging
from transformers import NllbTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import time
from src.utils import get_device
from src.dataset import TextPreprocessor

logger = logging.getLogger(__name__)

class NLLBPredictor:
    """Predictor para modelos NLLB fine-tuneados"""
    
    def __init__(self, model_path, direction, config, max_length=256, num_beams=4, length_penalty=1.0):
        self.model_path = model_path
        self.direction = direction
        self.config = config
        self.max_length = max_length
        self.num_beams = num_beams
        self.length_penalty = length_penalty
        self.device = get_device()
        
        # Configurar tokens de idioma
        self.lang_tokens = {
            'agr': 'agr_Latn',
            'es': 'spa_Latn'
        }
        
        # Obtener dirección de traducción
        self._setup_direction()
        
        # Preprocesador
        self.preprocessor = TextPreprocessor()
        
        # Cargar modelo y tokenizer
        self._load_model()
        
        logger.info(f"✅ Predictor listo - {self.src_lang.upper()} → {self.tgt_lang.upper()}")
    
    def _setup_direction(self):
        """Configurar dirección de traducción"""
        if self.direction == 'es2agr':
            self.src_lang = 'es'
            self.tgt_lang = 'agr'
            self.src_token = self.lang_tokens['es']
            self.tgt_token = self.lang_tokens['agr']
        else:  # agr2es
            self.src_lang = 'agr'
            self.tgt_lang = 'es'
            self.src_token = self.lang_tokens['agr']
            self.tgt_token = self.lang_tokens['es']
    
    def _load_model(self):
        """Cargar modelo y tokenizer desde el checkpoint"""
        logger.info(f"🔄 Cargando modelo desde: {self.model_path}")
        
        try:
            # Cargar tokenizer y modelo
            self.tokenizer = NllbTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path)
            
            # Mover a dispositivo
            self.model.to(self.device)
            self.model.eval()
            
            # Información del modelo
            param_count = sum(p.numel() for p in self.model.parameters())
            model_size = sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024 * 1024)
            
            logger.info(f"📊 Parámetros: {param_count:,}")
            logger.info(f"📊 Tamaño: {model_size:.1f} MB")
            logger.info(f"🎯 Dispositivo: {self.device}")
            
        except Exception as e:
            logger.error(f"❌ Error cargando modelo: {e}")
            raise
    
    def preprocess_text(self, text):
        """Preprocesar texto de entrada"""
        if isinstance(text, list):
            return [self.preprocessor.preprocess(t) for t in text]
        else:
            return self.preprocessor.preprocess(text)
    
    def tokenize_input(self, texts):
        """Tokenizar texto(s) de entrada"""
        if isinstance(texts, str):
            texts = [texts]
        
        # Configurar idioma fuente
        self.tokenizer.src_lang = self.src_token
        
        # Tokenizar
        inputs = self.tokenizer(
            texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.max_length
        )
        
        return inputs.to(self.device)
    
    def generate_translation(self, inputs):
        """Generar traducción usando el modelo"""
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=self.max_length,
                num_beams=self.num_beams,
                length_penalty=self.length_penalty,
                early_stopping=True,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decodificar
        translations = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return [t.strip() for t in translations]
    
    def translate_single(self, text):
        """Traducir un solo texto"""
        if not text or not text.strip():
            return ""
        
        # Preprocesar
        processed_text = self.preprocess_text(text)
        
        # Tokenizar
        inputs = self.tokenize_input(processed_text)
        
        # Generar traducción
        translations = self.generate_translation(inputs)
        
        return translations[0]
    
    def translate_batch(self, texts, batch_size=16, show_progress=True):
        """Traducir lote de textos"""
        if not texts:
            return []
        
        # Preprocesar todos los textos
        processed_texts = self.preprocess_text(texts)
        
        translations = []
        
        # Procesar en batches
        iterator = range(0, len(processed_texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Traduciendo")
        
        for i in iterator:
            batch = processed_texts[i:i + batch_size]
            
            # Tokenizar batch
            inputs = self.tokenize_input(batch)
            
            # Generar traducciones
            batch_translations = self.generate_translation(inputs)
            translations.extend(batch_translations)
            
            # Limpiar memoria GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return translations
    
    def translate_file(self, input_file, output_file, batch_size=16):
        """Traducir archivo completo"""
        logger.info(f"📂 Procesando archivo: {input_file}")
        
        # Leer archivo
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        if not lines:
            logger.warning("⚠️ Archivo vacío")
            return []
        
        logger.info(f"📊 Traduciendo {len(lines)} líneas...")
        start_time = time.time()
        
        # Traducir
        translations = self.translate_batch(lines, batch_size=batch_size)
        
        # Guardar resultado
        with open(output_file, 'w', encoding='utf-8') as f:
            for translation in translations:
                f.write(translation + '\n')
        
        elapsed = time.time() - start_time
        logger.info(f"✅ Traducción completada en {elapsed:.2f}s")
        logger.info(f"💾 Resultado guardado en: {output_file}")
        
        return translations
    
    def evaluate_samples(self, samples):
        """Evaluar muestras con referencia (para testing)"""
        if not samples:
            return {}
        
        sources = [s['source'] for s in samples]
        references = [s['reference'] for s in samples]
        predictions = self.translate_batch(sources, show_progress=False)
        
        # Calcular métricas básicas
        exact_matches = sum(1 for p, r in zip(predictions, references) if p.strip().lower() == r.strip().lower())
        
        results = {
            'total_samples': len(samples),
            'exact_matches': exact_matches,
            'exact_match_rate': exact_matches / len(samples) if samples else 0,
            'avg_prediction_length': sum(len(p) for p in predictions) / len(predictions) if predictions else 0,
            'avg_reference_length': sum(len(r) for r in references) / len(references) if references else 0
        }
        
        return results, list(zip(sources, references, predictions))
    
    def get_model_info(self):
        """Obtener información del modelo"""
        param_count = sum(p.numel() for p in self.model.parameters())
        model_size = sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024 * 1024)
        
        return {
            'model_path': self.model_path,
            'direction': self.direction,
            'src_lang': self.src_lang,
            'tgt_lang': self.tgt_lang,
            'src_token': self.src_token,
            'tgt_token': self.tgt_token,
            'parameters': param_count,
            'size_mb': model_size,
            'device': str(self.device),
            'max_length': self.max_length,
            'num_beams': self.num_beams,
            'length_penalty': self.length_penalty
        }

class BatchPredictor:
    """Predictor optimizado para grandes volúmenes de datos"""
    
    def __init__(self, model_path, direction, config, batch_size=32, max_length=256):
        self.predictor = NLLBPredictor(model_path, direction, config, max_length=max_length)
        self.batch_size = batch_size
        
    def predict_large_file(self, input_file, output_file, progress_callback=None):
        """Procesar archivos grandes con callback de progreso"""
        logger.info(f"🚀 Procesamiento masivo: {input_file}")
        
        # Contar líneas total
        with open(input_file, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for line in f if line.strip())
        
        logger.info(f"📊 Total de líneas: {total_lines:,}")
        
        processed = 0
        
        with open(input_file, 'r', encoding='utf-8') as f_in, \
             open(output_file, 'w', encoding='utf-8') as f_out:
            
            batch = []
            
            for line in tqdm(f_in, total=total_lines, desc="Procesando"):
                line = line.strip()
                if not line:
                    continue
                
                batch.append(line)
                
                if len(batch) >= self.batch_size:
                    # Procesar batch
                    translations = self.predictor.translate_batch(
                        batch, batch_size=self.batch_size, show_progress=False
                    )
                    
                    # Escribir resultados
                    for translation in translations:
                        f_out.write(translation + '\n')
                    
                    processed += len(batch)
                    batch = []
                    
                    # Callback de progreso
                    if progress_callback:
                        progress_callback(processed, total_lines)
                    
                    # Limpiar memoria
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            # Procesar último batch
            if batch:
                translations = self.predictor.translate_batch(
                    batch, batch_size=len(batch), show_progress=False
                )
                
                for translation in translations:
                    f_out.write(translation + '\n')
                
                processed += len(batch)
        
        logger.info(f"✅ Procesamiento completado: {processed:,} líneas")
        
        return processed