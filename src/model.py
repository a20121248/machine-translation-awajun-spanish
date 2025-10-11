"""
Configuración del modelo y tokenizer NLLB
"""

import torch
import logging
from transformers import (
    NllbTokenizer,
    AutoModelForSeq2SeqLM
)
from src.utils import get_device, count_parameters, get_model_size_mb

logger = logging.getLogger(__name__)

class NLLBModel:
    """Wrapper para modelo NLLB con configuración específica"""
    
    def __init__(self, config):
        self.config = config
        self.device = get_device()
        
        # Configurar tokens de idioma
        self.lang_tokens = {
            'agr': 'agr_Latn',
            'es': 'spa_Latn'
        }
        
        # Inicializar modelo y tokenizer
        self._load_model_and_tokenizer()
        self._setup_optimizer_and_scheduler()
        
    def _load_model_and_tokenizer(self):
        """Cargar modelo y tokenizer desde HuggingFace"""
        model_name = self.config['model']['name']
        logger.info(f"🤖 Cargando modelo: {model_name}")
        
        try:
            self.tokenizer = NllbTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            
            # Añadir token especial para Awajún si no existe
            lang_code = self.config['model']['lang_code']
            if lang_code not in self.tokenizer.get_vocab():
                self.tokenizer.add_special_tokens(
                    {"additional_special_tokens": [lang_code]}, 
                    replace_additional_special_tokens=False
                )
                self.model.resize_token_embeddings(len(self.tokenizer))
                logger.info(f"✅ Token añadido: {lang_code}")
            
            # Por ahora usar solo 1 GPU - DataParallel causa problemas
            self.device = get_device()
            logger.info(f"🚀 Usando dispositivo: {self.device}")
            
            # Mover modelo a dispositivo
            self.model.to(self.device)
            
            # Información del modelo
            model_for_params = self.model.module if hasattr(self.model, 'module') else self.model
            param_count = count_parameters(model_for_params)
            model_size = get_model_size_mb(model_for_params)
            logger.info(f"📊 Parámetros: {param_count:,}")
            logger.info(f"📊 Tamaño: {model_size:.1f} MB")
            
        except Exception as e:
            logger.error(f"❌ Error cargando modelo: {e}")
            raise
    
    def _setup_optimizer_and_scheduler(self):
        """Configurar optimizador y scheduler"""
        # Asegurar que learning_rate es float
        lr = float(self.config['training']['learning_rate'])
        wd = float(self.config['training']['weight_decay'])
        
        # Usar AdamW en lugar de Adafactor para evitar problemas
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=wd,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Scheduler simple que funciona
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=1000,  # No cambia el LR realmente
            gamma=1.0
        )
        
        logger.info(f"✅ Optimizador configurado - LR: {lr}, Weight decay: {wd}")
    
    def get_language_tokens(self, direction):
        """Obtener tokens de idioma según la dirección"""
        if direction == 'es2agr':
            src_token = self.lang_tokens['es']
            tgt_token = self.lang_tokens['agr']
            src_lang = 'es'
            tgt_lang = 'agr'
        else:  # agr2es
            src_token = self.lang_tokens['agr']
            tgt_token = self.lang_tokens['es']
            src_lang = 'agr'
            tgt_lang = 'es'
            
        return src_token, tgt_token, src_lang, tgt_lang
    
    def tokenize_batch(self, texts, lang_token, max_length=None):
        """Tokenizar batch de textos"""
        if max_length is None:
            max_length = self.config['model']['max_length']
            
        self.tokenizer.src_lang = lang_token
        
        return self.tokenizer(
            texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(self.device)
    
    def generate_translation(self, src_text, src_token, max_length=None):
        """Generar traducción para un texto"""
        if max_length is None:
            max_length = self.config['model']['max_length']
            
        inputs = self.tokenize_batch([src_text], src_token, max_length)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=4,
                length_penalty=1.0,
                early_stopping=True
            )
        
        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        return decoded.strip()
    
    def train_step(self, src_texts, tgt_texts, src_token, tgt_token):
        """Realizar un paso de entrenamiento"""
        # Tokenizar entrada
        src_inputs = self.tokenize_batch(src_texts, src_token)
        tgt_inputs = self.tokenize_batch(tgt_texts, tgt_token)
        
        # Preparar labels (reemplazar pad_token_id con -100)
        labels = tgt_inputs.input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        # Forward pass
        outputs = self.model(**src_inputs, labels=labels)
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        
        # Actualizar parámetros
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()
        
        return loss.item()
    
    def save_model(self, save_path):
        """Guardar modelo y tokenizer"""
        logger.info(f"💾 Guardando modelo en: {save_path}")
        
        # Modelo simple (no DataParallel)
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
    
    def get_current_lr(self):
        """Obtener learning rate actual"""
        return self.scheduler.get_last_lr()[0]