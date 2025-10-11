#!/usr/bin/env python3
"""
Script principal para fine-tuning NLLB Awajún-Español
Uso: python train.py --direction es2agr --dataset_version v1
Soporta datasets sintéticos: v3-top20, v3-top40, v3-top60, v3-top80, v3
"""

import argparse
import yaml
from src.training import Trainer
from src.utils import setup_logging, set_random_seed

def load_config(config_path="config.yaml"):
    """Cargar configuración desde YAML"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def parse_args():
    """Argumentos de línea de comandos"""
    parser = argparse.ArgumentParser(description="Fine-tuning NLLB para Awajún-Español")
    
    # Parámetros principales
    parser.add_argument('--direction', type=str, required=True, 
                       choices=['es2agr', 'agr2es'], 
                       help='Dirección de traducción')
    parser.add_argument('--dataset_version', type=str, default='v1',
                       help='Versión del dataset (v1, v2, v3-top20, v3-top40, v3-top60, v3-top80, v3)')
    
    # Overrides de configuración
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Archivo de configuración')
    parser.add_argument('--model_name', type=str, default=None,
                       help='Modelo a usar (ej: facebook/nllb-200-1.3B)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Número de épocas (override config)')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Tamaño de batch (override config)')
    parser.add_argument('--learning_rate', type=float, default=None,
                       help='Learning rate (override config)')
    parser.add_argument('--patience', type=int, default=10,
                       help='Paciencia para early stopping (default: 10)')
    parser.add_argument('--eval_frequency', type=int, default=10,
                       help='Evaluar chrF++ cada N épocas (default: 10)')
    
    # Modos especiales
    parser.add_argument('--test_mode', action='store_true',
                       help='Modo de prueba rápida (2 épocas, pocos samples)')
    parser.add_argument('--quick_eval', action='store_true',
                       help='Evaluación rápida (500 samples en lugar de todo dev)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Reanudar desde checkpoint')
    
    return parser.parse_args()

def main():
    """Función principal"""
    args = parse_args()
    
    # Cargar configuración
    config = load_config(args.config)
    
    # Aplicar overrides de CLI
    if args.model_name is not None:
        config['model']['name'] = args.model_name
        # Auto-detectar display_name desde model_name
        if 'distilled-600M' in args.model_name:
            config['model']['display_name'] = 'nllb-600M-distilled'
        elif 'distilled-1.3B' in args.model_name:
            config['model']['display_name'] = 'nllb-1.3B-distilled'
        elif '3.3B' in args.model_name:
            config['model']['display_name'] = 'nllb-3.3B'
        else:
            config['model']['display_name'] = args.model_name.split('/')[-1]
    
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    if args.learning_rate is not None:
        config['training']['learning_rate'] = args.learning_rate
    if args.patience is not None:
        config['training']['patience'] = args.patience
    
    # NUEVO: Configurar frecuencia de evaluación
    config['evaluation']['eval_frequency'] = args.eval_frequency
    
    # Configurar modo de prueba
    if args.test_mode:
        config['testing']['quick_test'] = True
        config['training']['epochs'] = config['testing']['test_epochs']
        config['evaluation']['eval_sample_size'] = 50  # Evaluación súper rápida
        config['evaluation']['eval_frequency'] = 1  # Evaluar cada época en test
        print("🧪 Modo de prueba activado - Entrenamiento rápido")
    
    # Configurar evaluación rápida
    if args.quick_eval:
        config['evaluation']['eval_sample_size'] = 500
        print("⚡ Evaluación rápida activada - 500 samples")
    
    # Configurar paths y nombres
    config['data']['dataset_version'] = args.dataset_version
    config['experiment']['direction'] = args.direction
    config['experiment']['resume'] = args.resume
    
    # Validar que el dataset existe
    from pathlib import Path
    dataset_path = Path(f"data/awajun-spanish-{args.dataset_version}")
    if not dataset_path.exists():
        print(f"❌ Error: Dataset no encontrado en {dataset_path}")
        print(f"   Asegúrate de que exista el directorio con los archivos:")
        print(f"   - train.agr, train.es, train.source")
        print(f"   - dev.agr, dev.es, dev.source")
        exit(1)
    
    # Setup logging y semilla
    setup_logging()
    set_random_seed(42)
    
    # Mensaje informativo
    print(f"\n{'='*80}")
    print(f"🚀 INICIANDO ENTRENAMIENTO")
    print(f"{'='*80}")
    print(f"📦 Dataset: awajun-spanish-{args.dataset_version}")
    print(f"🔄 Dirección: {args.direction}")
    print(f"🤖 Modelo: {config['model']['display_name']}")
    print(f"📊 Épocas: {config['training']['epochs']}")
    print(f"📈 Eval cada: {config['evaluation']['eval_frequency']} épocas")
    print(f"⏸️  Paciencia: {config['training']['patience']} épocas sin mejora")
    print(f"{'='*80}\n")
    
    # Crear y ejecutar trainer
    trainer = Trainer(config)
    trainer.run()

if __name__ == "__main__":
    main()