"""
Utilidades para logging, early stopping y configuración
"""

import os
import random
import logging
import numpy as np
import torch
from datetime import datetime

def setup_logging():
    """Configurar logging global"""
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def set_random_seed(seed=42):
    """Fijar semilla para reproducibilidad"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device():
    """Obtener device disponible"""
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"🚀 Usando GPU 0: {torch.cuda.get_device_name(0)}")
        if torch.cuda.device_count() > 1:
            print(f"ℹ️  {torch.cuda.device_count()} GPUs detectadas, usando solo GPU 0")
    else:
        device = torch.device("cpu")
        print("💻 Usando CPU")
    return device

def create_run_dir(base_dir, experiment_name):
    """Crear directorio para la corrida"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(base_dir, f'{experiment_name}_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

class EarlyStopping:
    """Early stopping para detener entrenamiento cuando no hay mejora"""
    
    def __init__(self, patience=3, min_delta=0.1, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
            
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
            
        if improved:
            self.best_score = score
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
            return False
    
    def get_status(self):
        """Obtener estado actual del early stopping"""
        return {
            'counter': self.counter,
            'best_score': self.best_score,
            'patience': self.patience,
            'early_stop': self.early_stop
        }

def format_time(seconds):
    """Formatear tiempo en formato legible"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"

def count_parameters(model):
    """Contar parámetros del modelo"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_size_mb(model):
    """Obtener tamaño del modelo en MB"""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 * 1024)