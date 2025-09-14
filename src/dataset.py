"""
Carga y preprocesamiento de datos para traducción Awajún-Español
"""

import os
import re
import sys
import unicodedata
import pandas as pd
import logging
import torch
from sklearn.utils.class_weight import compute_sample_weight
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sacremoses import MosesPunctNormalizer

logger = logging.getLogger(__name__)

class TranslationDataset(Dataset):
    """Dataset para pares de traducción"""
    
    def __init__(self, df, src_lang, tgt_lang, tokenizer, src_token, tgt_token, max_length=128):
        self.df = df
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.tokenizer = tokenizer
        self.src_token = src_token
        self.tgt_token = tgt_token
        self.max_length = max_length
        
        # Preprocesador
        self.preprocessor = TextPreprocessor()
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        src_text = self.preprocessor.preprocess(row[self.src_lang])
        tgt_text = self.preprocessor.preprocess(row[self.tgt_lang])
        return src_text, tgt_text

class TextPreprocessor:
    """Preprocesamiento de texto"""
    
    def __init__(self):
        self.mpn = MosesPunctNormalizer(lang="en")
        self.mpn.substitutions = [(re.compile(r), sub) for r, sub in self.mpn.substitutions]
        self.normalize_func = self._create_normalizer()
    
    def _create_normalizer(self):
        """Crear función para remover caracteres no imprimibles"""
        nonprint_map = {
            ord(c): " " 
            for c in (chr(i) for i in range(sys.maxunicode)) 
            if unicodedata.category(c).startswith("C")
        }
        return lambda text: text.translate(nonprint_map)
    
    def preprocess(self, text):
        """Preprocesar texto completo"""
        if pd.isna(text):
            return ""
        text = str(text)
        text = self.mpn.normalize(text)
        text = self.normalize_func(text)
        text = unicodedata.normalize("NFKC", text)
        return text.strip()

class AwajunDataLoader:
    """Cargador de datos para entrenamiento"""
    
    def __init__(self, config):
        self.config = config
        self.data_path = os.path.join(
            config['data']['base_path'],
            f"awajun-spanish-{config['data']['dataset_version']}"
        )
        
    def load_parallel_files(self, prefix):
        """Cargar archivos paralelos (agr, es, source)"""
        agr_path = os.path.join(self.data_path, f"{prefix}.agr")
        es_path = os.path.join(self.data_path, f"{prefix}.es")
        source_path = os.path.join(self.data_path, f"{prefix}.source")
        
        if not all(os.path.exists(p) for p in [agr_path, es_path, source_path]):
            raise FileNotFoundError(f"Archivos no encontrados en {self.data_path}")
        
        with open(agr_path, 'r', encoding='utf-8') as f:
            agr_lines = f.read().splitlines()
        with open(es_path, 'r', encoding='utf-8') as f:
            es_lines = f.read().splitlines()
        with open(source_path, 'r', encoding='utf-8') as f:
            source_lines = f.read().splitlines()
            
        return pd.DataFrame({
            'agr': agr_lines,
            'es': es_lines,
            'source': source_lines
        })
    
    def load_data(self):
        """Cargar datos de entrenamiento y desarrollo"""
        logger.info(f"📂 Cargando datos desde: {self.data_path}")
        
        df_train = self.load_parallel_files("train")
        df_dev = self.load_parallel_files("dev")
        
        logger.info(f"📊 Train: {len(df_train)} samples, Dev: {len(df_dev)} samples")
        
        # Mostrar distribución por fuente
        source_dist = df_train['source'].value_counts()
        logger.info("📈 Distribución por fuente:")
        for source, count in source_dist.items():
            logger.info(f"  {source}: {count}")
        
        # Modo de prueba rápida
        if self.config['testing']['quick_test']:
            train_samples = self.config['testing']['test_train_samples']
            dev_samples = self.config['testing']['test_dev_samples']
            
            df_train = df_train.sample(min(train_samples, len(df_train)), random_state=42)
            df_dev = df_dev.sample(min(dev_samples, len(df_dev)), random_state=42)
            
            logger.info(f"🧪 Modo prueba: Train={len(df_train)}, Dev={len(df_dev)}")
        
        return df_train, df_dev
    
    def create_weighted_sampler(self, df):
        """Crear sampler con pesos balanceados"""
        if self.config['data']['balance_method'] != 'weighted':
            return None
            
        logger.info("⚖️  Creando weighted sampler...")
        sample_weights = compute_sample_weight('balanced', df['source'].values)
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(df),
            replacement=True
        )
        return sampler
    
    def create_dataloader(self, dataset, sampler=None, shuffle=True):
        """Crear DataLoader"""
        if sampler is not None:
            shuffle = False  # No shuffle cuando hay sampler
            
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=shuffle,
            sampler=sampler,
            num_workers=0,  # Para evitar problemas con multiprocessing
            pin_memory=torch.cuda.is_available()
        )