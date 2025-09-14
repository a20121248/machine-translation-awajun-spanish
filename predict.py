#!/usr/bin/env python3
"""
Script principal para inferencia con modelos NLLB fine-tuneados
Uso: python predict.py --model_path runs/best_model --direction es2agr --input "Hola mundo"
"""

import argparse
import yaml
import sys
import os
from src.inference import NLLBPredictor
from src.utils import setup_logging


def load_config(config_path="config.yaml"):
    """Cargar configuración desde YAML"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def parse_args():
    """Argumentos de línea de comandos"""
    parser = argparse.ArgumentParser(description="Inferencia NLLB para Awajún-Español")

    # Parámetros principales
    parser.add_argument('--model_path', type=str, required=True,
                        help='Ruta al modelo fine-tuneado')
    parser.add_argument('--direction', type=str, required=True,
                        choices=['es2agr', 'agr2es'],
                        help='Dirección de traducción')

    # Entrada
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--input', type=str,
                       help='Texto a traducir')
    group.add_argument('--input_file', type=str,
                       help='Archivo con textos a traducir')
    group.add_argument('--interactive', action='store_true',
                       help='Modo interactivo')

    # Salida
    parser.add_argument('--output_file', type=str, default=None,
                        help='Archivo de salida (default: stdout)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Tamaño de batch para procesamiento')

    # Parámetros de generación
    parser.add_argument('--max_length', type=int, default=256,
                        help='Longitud máxima de salida')
    parser.add_argument('--num_beams', type=int, default=4,
                        help='Número de beams para beam search')
    parser.add_argument('--length_penalty', type=float, default=1.0,
                        help='Penalización por longitud')

    # Configuración
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Archivo de configuración')
    parser.add_argument('--verbose', action='store_true',
                        help='Output detallado')

    return parser.parse_args()


def interactive_mode(predictor, direction):
    """Modo interactivo de traducción"""
    lang_map = {'es2agr': ('Español', 'Awajún'), 'agr2es': ('Awajún', 'Español')}
    src_lang, tgt_lang = lang_map[direction]

    print(f"\n🌍 Traductor {src_lang} → {tgt_lang}")
    print("Escribe 'quit' o 'salir' para terminar\n")

    while True:
        try:
            text = input(f"{src_lang}: ").strip()

            if text.lower() in ['quit', 'exit', 'salir', 'q']:
                print("¡Hasta luego!")
                break

            if not text:
                continue

            translation = predictor.translate_single(text)
            print(f"{tgt_lang}: {translation}\n")

        except KeyboardInterrupt:
            print("\n¡Hasta luego!")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    """Función principal"""
    args = parse_args()

    # Setup logging
    if args.verbose:
        setup_logging()

    # Cargar configuración
    config = load_config(args.config)

    # Verificar que el modelo existe
    if not os.path.exists(args.model_path):
        print(f"❌ Error: Modelo no encontrado en {args.model_path}")
        sys.exit(1)

    # Crear predictor
    print(f"🤖 Cargando modelo desde: {args.model_path}")
    predictor = NLLBPredictor(
        model_path=args.model_path,
        direction=args.direction,
        config=config,
        max_length=args.max_length,
        num_beams=args.num_beams,
        length_penalty=args.length_penalty
    )

    try:
        if args.interactive:
            # Modo interactivo
            interactive_mode(predictor, args.direction)

        elif args.input:
            # Texto único
            print(f"📝 Traduciendo: {args.input}")
            translation = predictor.translate_single(args.input)

            if args.output_file:
                with open(args.output_file, 'w', encoding='utf-8') as f:
                    f.write(translation)
                print(f"💾 Traducción guardada en: {args.output_file}")
            else:
                print(f"🔄 Traducción: {translation}")

        elif args.input_file:
            # Archivo de entrada
            if not os.path.exists(args.input_file):
                print(f"❌ Error: Archivo no encontrado: {args.input_file}")
                sys.exit(1)

            print(f"📂 Procesando archivo: {args.input_file}")

            with open(args.input_file, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]

            if not lines:
                print("⚠️ El archivo está vacío")
                sys.exit(1)

            print(f"📊 Traduciendo {len(lines)} líneas...")
            translations = predictor.translate_batch(lines, batch_size=args.batch_size)

            if args.output_file:
                with open(args.output_file, 'w', encoding='utf-8') as f:
                    for translation in translations:
                        f.write(translation + '\n')
                print(f"💾 Traducciones guardadas en: {args.output_file}")
            else:
                for i, (original, translation) in enumerate(zip(lines, translations)):
                    print(f"{i + 1:3d} | {original}")
                    print(f"    → {translation}\n")

    except Exception as e:
        print(f"❌ Error durante la traducción: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()