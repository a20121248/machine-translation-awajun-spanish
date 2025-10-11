#!/usr/bin/env python3
"""
Traductor de archivos de texto línea por línea (versión mejorada)
Soporta modelos fragmentados y carga flexible
"""

import os
import torch
import argparse
from transformers import NllbTokenizer, AutoModelForSeq2SeqLM
from pathlib import Path

def load_specific_model(model_path):
    print(f"Cargando modelo desde: {model_path}")
    
    try:
        # Verificar que existe config.json (mínimo necesario)
        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            print(f"Error: No se encuentra config.json en {model_path}")
            return None, None, None
        
        # Verificar que hay algún archivo de modelo
        model_files = [
            "model.safetensors",
            "pytorch_model.bin",
            "model.safetensors.index.json",  # Para modelos fragmentados
            "pytorch_model.bin.index.json"
        ]
        
        has_model = any(os.path.exists(os.path.join(model_path, f)) for f in model_files)
        
        if not has_model:
            print(f"Error: No se encuentra archivo de modelo en {model_path}")
            print(f"Archivos buscados: {model_files}")
            return None, None, None
        
        print("Cargando modelo y tokenizer...")
        
        # Cargar modelo y tokenizer (from_pretrained maneja fragmentación automáticamente)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        tokenizer = NllbTokenizer.from_pretrained(model_path)
        
        # Configurar device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        
        print(f"✅ Modelo cargado exitosamente en: {device}")
        return model, tokenizer, device
        
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def translate_batch(texts, direction, model, tokenizer, device, batch_size=8):
    """Traducir múltiples textos en un solo batch"""
    # Tokens de idioma
    lang_tokens = {
        'agr': 'agr_Latn',
        'es': 'spa_Latn'
    }
    
    if direction == 'es2agr':
        src_token = lang_tokens['es']
    elif direction == 'agr2es':
        src_token = lang_tokens['agr']
    else:
        raise ValueError("direction debe ser 'es2agr' o 'agr2es'")
    
    # Filtrar textos vacíos pero mantener índices
    non_empty_texts = []
    text_indices = []
    
    for i, text in enumerate(texts):
        if text.strip():
            non_empty_texts.append(text.strip())
            text_indices.append(i)
    
    if not non_empty_texts:
        return [""] * len(texts)
    
    # Tokenizar batch
    tokenizer.src_lang = src_token
    inputs = tokenizer(
        non_empty_texts,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=256
    ).to(device)
    
    # Generar traducciones
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=256,
            num_beams=4,
            length_penalty=1.0,
            early_stopping=True,
            do_sample=False
        )
    
    # Decodificar
    translations = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    # Reconstruir lista completa con textos vacíos en su lugar
    result = [""] * len(texts)
    for i, translation in enumerate(translations):
        original_index = text_indices[i]
        result[original_index] = translation.strip()
    
    return result

def translate_file(model_path, direction, input_file, output_file, batch_size=8, resume=False):
    """
    Traducir archivo de texto línea por línea usando batches
    """
    
    # Verificar archivos
    if not os.path.exists(input_file):
        print(f"Error: Archivo de entrada no existe: {input_file}")
        return False
    
    if not os.path.exists(model_path):
        print(f"Error: Modelo no existe: {model_path}")
        return False
    
    # Manejo de archivos existentes
    start_line = 0
    if os.path.exists(output_file):
        if resume:
            with open(output_file, 'r', encoding='utf-8') as f:
                start_line = sum(1 for _ in f)
            print(f"Archivo de salida existe. Resumiendo desde línea {start_line + 1}")
        else:
            import time
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            backup_file = f"{output_file}.backup_{timestamp}"
            
            print(f"ADVERTENCIA: El archivo {output_file} ya existe.")
            print(f"Se creará backup en: {backup_file}")
            
            import shutil
            shutil.copy2(output_file, backup_file)
            print(f"Backup creado: {backup_file}")
            
            start_line = 0
    
    # Cargar modelo
    model, tokenizer, device = load_specific_model(model_path)
    if not model:
        print("Error: No se pudo cargar el modelo")
        return False
    
    # Procesar archivo
    print(f"Traduciendo archivo: {input_file}")
    print(f"Dirección: {direction}")
    print(f"Salida: {output_file}")
    print(f"Batch size: {batch_size}")
    if start_line > 0:
        print(f"Resumiendo desde línea: {start_line + 1}")
    print("-" * 50)
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f_in:
            lines = [line.rstrip('\n\r') for line in f_in.readlines()]
        
        total_lines = len(lines)
        remaining_lines = total_lines - start_line
        
        print(f"Total de líneas: {total_lines}")
        if start_line > 0:
            print(f"Líneas ya procesadas: {start_line}")
            print(f"Líneas restantes: {remaining_lines}")
        
        translated_lines = []
        errors = 0
        
        import time
        start_time = time.time()
        
        file_mode = 'a' if resume and start_line > 0 else 'w'
        
        with open(output_file, file_mode, encoding='utf-8') as f_out:
            for batch_start in range(start_line, total_lines, batch_size):
                batch_end = min(batch_start + batch_size, total_lines)
                batch_lines = lines[batch_start:batch_end]
                
                try:
                    batch_translations = translate_batch(batch_lines, direction, model, tokenizer, device, batch_size)
                    
                    for translation in batch_translations:
                        f_out.write(translation + '\n')
                    f_out.flush()
                    
                    translated_lines.extend(batch_translations)
                    
                    current_line = batch_end
                    processed_lines = current_line - start_line
                    elapsed = time.time() - start_time
                    rate = processed_lines / elapsed if elapsed > 0 else 0
                    remaining = (total_lines - current_line) / rate if rate > 0 else 0
                    
                    percentage = (current_line / total_lines) * 100
                    
                    # Progress bar simple (sobreescribe la misma línea)
                    print(f"\r[{percentage:5.1f}%] {current_line:5d}/{total_lines} | "
                          f"{rate:.1f} líneas/seg | "
                          f"ETA: {remaining/60:.1f}min", end='', flush=True)
                    
                    if batch_start == start_line:
                        print()  # Nueva línea después del primer batch
                        print("Ejemplos de traducción:")
                        for i in range(min(3, len(batch_lines))):
                            if batch_lines[i].strip() and batch_translations[i].strip():
                                print(f"  {i+1}. Original:  {batch_lines[i][:80]}{'...' if len(batch_lines[i]) > 80 else ''}")
                                print(f"     Traducido: {batch_translations[i][:80]}{'...' if len(batch_translations[i]) > 80 else ''}")
                        print()
                    
                except Exception as e:
                    print(f"\nERROR en batch {batch_start}-{batch_end}: {e}")
                    error_lines = [f"[ERROR: {line}]" for line in batch_lines]
                    for error_line in error_lines:
                        f_out.write(error_line + '\n')
                    f_out.flush()
                    
                    translated_lines.extend(error_lines)
                    errors += len(batch_lines)
        
        print()  # Nueva línea al final de la barra de progreso
        print("-" * 50)
        print(f"Traducción completada!")
        print(f"Líneas procesadas: {total_lines}")
        print(f"Errores: {errors}")
        print(f"Tiempo total: {(time.time() - start_time)/60:.1f} minutos")
        print(f"Archivo guardado: {output_file}")
        
        return True
        
    except Exception as e:
        print(f"Error procesando archivo: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        if 'model' in locals():
            del model, tokenizer
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

def main():
    parser = argparse.ArgumentParser(description="Traductor de archivos de texto")
    
    parser.add_argument('--model_path', required=True, 
                       help='Path al modelo (directorio best_model)')
    parser.add_argument('--direction', required=True, choices=['agr2es', 'es2agr'],
                       help='Dirección de traducción')
    parser.add_argument('--input_file', required=True,
                       help='Archivo de entrada')
    parser.add_argument('--output_file', required=True,
                       help='Archivo de salida')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Tamaño del batch (default: 16)')
    parser.add_argument('--resume', action='store_true',
                       help='Continuar desde donde se quedó')
    
    args = parser.parse_args()
    
    print("=== Traductor de Archivos Awajún-Español ===")
    print(f"Modelo: {args.model_path}")
    print(f"Dirección: {args.direction}")
    print(f"Entrada: {args.input_file}")
    print(f"Salida: {args.output_file}")
    print(f"Batch size: {args.batch_size}")
    print(f"Modo resume: {'Sí' if args.resume else 'No'}")
    print()
    
    success = translate_file(
        args.model_path,
        args.direction, 
        args.input_file,
        args.output_file,
        args.batch_size,
        args.resume
    )
    
    if success:
        print("\n✅ Traducción exitosa!")
    else:
        print("\n❌ Traducción falló!")
        exit(1)

if __name__ == "__main__":
    main()