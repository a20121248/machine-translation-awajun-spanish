#!/usr/bin/env python3
"""
Utilidades para gestión de modelos NLLB fine-tuneados
Uso: python model_utils.py --command list_models
"""

import argparse
import os
import json
import shutil
from pathlib import Path
import torch
from transformers import NllbTokenizer, AutoModelForSeq2SeqLM

def parse_args():
    """Argumentos de línea de comandos"""
    parser = argparse.ArgumentParser(description="Utilidades para modelos NLLB")
    
    parser.add_argument('--command', type=str, required=True,
                       choices=['list_models', 'model_info', 'convert_model', 'test_model', 'clean_cache'],
                       help='Comando a ejecutar')
    
    # Parámetros específicos
    parser.add_argument('--models_dir', type=str, default='runs',
                       help='Directorio de modelos')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Ruta específica del modelo')
    parser.add_argument('--output_path', type=str, default=None,
                       help='Ruta de salida')
    parser.add_argument('--test_text', type=str, default="Hola mundo",
                       help='Texto de prueba')
    
    return parser.parse_args()

def list_models(models_dir):
    """Listar modelos disponibles"""
    print(f"Modelos en {models_dir}:")
    print("=" * 50)
    
    if not os.path.exists(models_dir):
        print("Directorio no encontrado")
        return
    
    found_models = []
    
    for root, dirs, files in os.walk(models_dir):
        # Buscar directorios que contengan config.json y pytorch_model.bin
        if 'config.json' in files and any(f.startswith('pytorch_model') for f in files):
            rel_path = os.path.relpath(root, models_dir)
            model_info = get_model_info(root)
            found_models.append((rel_path, root, model_info))
    
    if not found_models:
        print("No se encontraron modelos")
        return
    
    for i, (rel_path, full_path, info) in enumerate(found_models, 1):
        print(f"{i}. {rel_path}")
        print(f"   Ruta: {full_path}")
        if info:
            print(f"   Tamaño: {info.get('size_mb', 'N/A'):.1f} MB")
            print(f"   Parámetros: {info.get('parameters', 'N/A'):,}")
            print(f"   Modificado: {info.get('modified', 'N/A')}")
        print()

def get_model_info(model_path):
    """Obtener información básica del modelo"""
    try:
        # Información del directorio
        stat = os.stat(model_path)
        modified = os.path.getctime(model_path)
        
        # Calcular tamaño total
        total_size = 0
        for root, dirs, files in os.walk(model_path):
            for file in files:
                file_path = os.path.join(root, file)
                total_size += os.path.getsize(file_path)
        
        # Leer config si existe
        config_path = os.path.join(model_path, 'config.json')
        parameters = None
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                # Estimar parámetros basado en configuración común de NLLB
                if 'd_model' in config and 'encoder_layers' in config:
                    # Estimación aproximada para NLLB
                    d_model = config['d_model']
                    layers = config['encoder_layers'] + config.get('decoder_layers', config['encoder_layers'])
                    vocab_size = config.get('vocab_size', 256000)
                    parameters = vocab_size * d_model + layers * d_model * d_model * 4
            except:
                pass
        
        return {
            'size_mb': total_size / (1024 * 1024),
            'parameters': parameters,
            'modified': time.ctime(modified)
        }
    except Exception as e:
        return {'error': str(e)}

def get_detailed_model_info(model_path):
    """Obtener información detallada del modelo"""
    print(f"Información detallada del modelo: {model_path}")
    print("=" * 60)
    
    if not os.path.exists(model_path):
        print("Modelo no encontrado")
        return
    
    try:
        # Cargar configuración
        config_path = os.path.join(model_path, 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            print("Configuración del modelo:")
            print(f"  Arquitectura: {config.get('model_type', 'N/A')}")
            print(f"  Vocabulario: {config.get('vocab_size', 'N/A'):,}")
            print(f"  Dimensiones: {config.get('d_model', 'N/A')}")
            print(f"  Capas encoder: {config.get('encoder_layers', 'N/A')}")
            print(f"  Capas decoder: {config.get('decoder_layers', 'N/A')}")
            print(f"  Heads atención: {config.get('encoder_attention_heads', 'N/A')}")
        
        # Información de archivos
        print("\nArchivos del modelo:")
        total_size = 0
        for file in sorted(os.listdir(model_path)):
            file_path = os.path.join(model_path, file)
            if os.path.isfile(file_path):
                size = os.path.getsize(file_path)
                total_size += size
                print(f"  {file}: {size / (1024*1024):.1f} MB")
        
        print(f"\nTamaño total: {total_size / (1024*1024):.1f} MB")
        
        # Intentar cargar el modelo para obtener info exacta
        try:
            print("\nCargando modelo para análisis detallado...")
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
            
            param_count = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            print(f"Parámetros totales: {param_count:,}")
            print(f"Parámetros entrenables: {trainable_params:,}")
            print(f"Memoria estimada: {param_count * 4 / (1024*1024):.1f} MB (FP32)")
            
            # Verificar tokenizer
            tokenizer_path = os.path.join(model_path, 'tokenizer.json')
            if os.path.exists(tokenizer_path):
                tokenizer = NllbTokenizer.from_pretrained(model_path)
                print(f"Tokenizer: {len(tokenizer)} tokens")
                
                # Verificar tokens especiales
                special_tokens = []
                if hasattr(tokenizer, 'added_tokens_decoder'):
                    for token_id, token in tokenizer.added_tokens_decoder.items():
                        if 'agr_Latn' in str(token):
                            special_tokens.append(str(token))
                
                if special_tokens:
                    print(f"Tokens Awajún: {', '.join(special_tokens)}")
            
        except Exception as e:
            print(f"Error cargando modelo: {e}")
            
    except Exception as e:
        print(f"Error obteniendo información: {e}")

def test_model(model_path, test_text="Hola mundo", direction="es2agr"):
    """Probar modelo con texto de ejemplo"""
    print(f"Probando modelo: {model_path}")
    print(f"Texto de entrada: {test_text}")
    print(f"Dirección: {direction}")
    print("=" * 50)
    
    try:
        from src.inference import NLLBPredictor
        import yaml
        
        # Cargar configuración básica
        config = {
            'model': {'lang_code': 'agr_Latn'},
            'data': {'base_path': 'data'}
        }
        
        predictor = NLLBPredictor(
            model_path=model_path,
            direction=direction,
            config=config
        )
        
        # Realizar traducción
        import time
        start = time.time()
        translation = predictor.translate_single(test_text)
        elapsed = time.time() - start
        
        print(f"Traducción: {translation}")
        print(f"Tiempo: {elapsed:.2f}s")
        
        # Información del modelo
        model_info = predictor.get_model_info()
        print(f"\nInformación del modelo:")
        print(f"  Parámetros: {model_info['parameters']:,}")
        print(f"  Dispositivo: {model_info['device']}")
        print(f"  Tokens: {model_info['src_token']} → {model_info['tgt_token']}")
        
    except Exception as e:
        print(f"Error probando modelo: {e}")

def convert_model(input_path, output_path, format_type="huggingface"):
    """Convertir modelo a diferentes formatos"""
    print(f"Convirtiendo modelo de {input_path} a {output_path}")
    print(f"Formato: {format_type}")
    
    if not os.path.exists(input_path):
        print("Modelo de entrada no encontrado")
        return
    
    if format_type == "huggingface":
        # Ya está en formato HuggingFace, solo copiar
        try:
            shutil.copytree(input_path, output_path)
            print("Modelo copiado exitosamente")
        except Exception as e:
            print(f"Error copiando modelo: {e}")
    
    elif format_type == "onnx":
        print("Conversión a ONNX no implementada aún")
    
    elif format_type == "torchscript":
        print("Conversión a TorchScript no implementada aún")
    
    else:
        print(f"Formato {format_type} no soportado")

def clean_cache():
    """Limpiar cache de modelos y temporales"""
    print("Limpiando cache...")
    
    # Cache de HuggingFace
    cache_dir = os.path.expanduser("~/.cache/huggingface")
    if os.path.exists(cache_dir):
        try:
            size_before = get_directory_size(cache_dir)
            # No eliminar todo, solo archivos temporales
            for root, dirs, files in os.walk(cache_dir):
                for file in files:
                    if file.endswith(('.tmp', '.lock', '.incomplete')):
                        os.remove(os.path.join(root, file))
            
            size_after = get_directory_size(cache_dir)
            print(f"Cache HuggingFace: {(size_before - size_after) / (1024*1024):.1f} MB liberados")
        except Exception as e:
            print(f"Error limpiando cache HuggingFace: {e}")
    
    # Cache de PyTorch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("Cache GPU liberado")
    
    # Archivos temporales del proyecto
    temp_patterns = ['*.tmp', '*.temp', '__pycache__']
    cleaned = 0
    
    for pattern in temp_patterns:
        for file_path in Path('.').rglob(pattern):
            try:
                if file_path.is_file():
                    file_path.unlink()
                    cleaned += 1
                elif file_path.is_dir():
                    shutil.rmtree(file_path)
                    cleaned += 1
            except Exception as e:
                print(f"Error eliminando {file_path}: {e}")
    
    print(f"Archivos temporales eliminados: {cleaned}")

def get_directory_size(directory):
    """Obtener tamaño total de un directorio"""
    total = 0
    try:
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                if os.path.exists(file_path):
                    total += os.path.getsize(file_path)
    except:
        pass
    return total

def main():
    """Función principal"""
    args = parse_args()
    
    if args.command == 'list_models':
        list_models(args.models_dir)
    
    elif args.command == 'model_info':
        if not args.model_path:
            print("Se requiere --model_path para model_info")
            return
        get_detailed_model_info(args.model_path)
    
    elif args.command == 'test_model':
        if not args.model_path:
            print("Se requiere --model_path para test_model")
            return
        test_model(args.model_path, args.test_text)
    
    elif args.command == 'convert_model':
        if not args.model_path or not args.output_path:
            print("Se requiere --model_path y --output_path para convert_model")
            return
        convert_model(args.model_path, args.output_path)
    
    elif args.command == 'clean_cache':
        clean_cache()

if __name__ == "__main__":
    import time
    main()