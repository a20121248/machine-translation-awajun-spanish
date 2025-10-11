#!/usr/bin/env python3
"""
Back-translation simple con dos modelos separados
Lee archivo con columnas, añade traducciones y métricas
"""

import os
import argparse
import subprocess
import sacrebleu
from pathlib import Path
from datetime import datetime

def extract_text_column(input_file, output_file, text_col=2, has_header=True):
    """Extrae solo la columna de texto para traducir"""
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        lines = f_in.readlines()
        start = 1 if has_header else 0
        
        for line in lines[start:]:
            parts = line.strip().split('|')
            if len(parts) > text_col:
                f_out.write(parts[text_col] + '\n')
            else:
                f_out.write('\n')
    
    print(f"✅ Texto extraído: {output_file}")

def translate_file(model_path, direction, input_file, output_file, batch_size=16):
    """Llama al script de traducción"""
    print(f"\n🔄 Traduciendo {direction}...")
    
    cmd = [
        "python", "translate_file.py",
        "--model_path", model_path,
        "--direction", direction,
        "--input_file", input_file,
        "--output_file", output_file,
        "--batch_size", str(batch_size)
    ]
    
    subprocess.run(cmd, check=True)
    print(f"✅ Completado: {output_file}")

def calculate_chrf(original_file, backtrans_file):
    """Calcula chrF++ entre original y back-translated"""
    with open(original_file, 'r', encoding='utf-8') as f:
        references = [line.strip() for line in f.readlines()]
    
    with open(backtrans_file, 'r', encoding='utf-8') as f:
        hypotheses = [line.strip() for line in f.readlines()]
    
    # Filtrar líneas vacías
    valid_pairs = [(ref, hyp) for ref, hyp in zip(references, hypotheses) 
                   if ref.strip() and hyp.strip()]
    
    refs = [pair[0] for pair in valid_pairs]
    hyps = [pair[1] for pair in valid_pairs]
    
    # Calcular métricas globales
    bleu = sacrebleu.corpus_bleu(hyps, [refs])
    chrf = sacrebleu.corpus_chrf(hyps, [refs])
    
    # Calcular chrF++ por línea
    chrf_scores = []
    for ref, hyp in zip(references, hypotheses):
        if ref.strip() and hyp.strip():
            score = sacrebleu.sentence_chrf(hyp, [ref]).score
            chrf_scores.append(score)
        else:
            chrf_scores.append(0.0)
    
    print(f"\n📊 Métricas globales:")
    print(f"   BLEU:   {bleu.score:.2f}")
    print(f"   chrF++: {chrf.score:.2f}")
    
    return chrf_scores, bleu.score, chrf.score

def create_output_file(input_file, spanish_file, backtrans_file, chrf_scores, 
                       output_file, global_bleu, global_chrf):
    """Crea archivo de salida con todas las columnas + archivo JSON con métricas"""
    
    with open(input_file, 'r', encoding='utf-8') as f:
        input_lines = f.readlines()
    
    with open(spanish_file, 'r', encoding='utf-8') as f:
        spanish_lines = [line.strip() for line in f.readlines()]
    
    with open(backtrans_file, 'r', encoding='utf-8') as f:
        backtrans_lines = [line.strip() for line in f.readlines()]
    
    # Crear archivo de datos
    with open(output_file, 'w', encoding='utf-8') as f_out:
        # Cabecera
        header = input_lines[0].strip()
        f_out.write(f"{header}|spanish_synthetic|awajun_backtranslated|chrf_score\n")
        
        # Datos
        for i, line in enumerate(input_lines[1:]):
            parts = line.strip().split('|')
            
            spanish = spanish_lines[i] if i < len(spanish_lines) else ""
            backtrans = backtrans_lines[i] if i < len(backtrans_lines) else ""
            chrf = f"{chrf_scores[i]:.2f}" if i < len(chrf_scores) else "0.00"
            
            f_out.write(f"{line.strip()}|{spanish}|{backtrans}|{chrf}\n")
    
    # Crear archivo JSON con métricas
    import json
    metrics_file = output_file.replace('.txt', '_metrics.json')
    
    metrics_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_file": str(Path(input_file).name),
        "output_file": str(Path(output_file).name),
        "global_metrics": {
            "BLEU": round(global_bleu, 2),
            "chrF++": round(global_chrf, 2)
        },
        "statistics": {
            "total_lines": len(chrf_scores),
            "avg_chrf": round(sum(chrf_scores) / len(chrf_scores), 2) if chrf_scores else 0,
            "min_chrf": round(min(chrf_scores), 2) if chrf_scores else 0,
            "max_chrf": round(max(chrf_scores), 2) if chrf_scores else 0,
            "lines_above_50": sum(1 for s in chrf_scores if s >= 50),
            "lines_above_60": sum(1 for s in chrf_scores if s >= 60),
            "lines_above_70": sum(1 for s in chrf_scores if s >= 70)
        },
        "interpretation": {
            "quality": "EXCELENTE" if global_chrf >= 70 else 
                      "BUENA" if global_chrf >= 60 else 
                      "ACEPTABLE" if global_chrf >= 50 else 
                      "MEJORABLE",
            "recommendation": "Corpus sintético de alta calidad" if global_chrf >= 60 else
                             "Considerar filtrado adicional" if global_chrf >= 50 else
                             "Revisar pipeline de traducción"
        }
    }
    
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Archivo de datos: {output_file}")
    print(f"✅ Archivo de métricas: {metrics_file}")
    print(f"   Columnas: document_id | segment_id | awajun_original | spanish_synthetic | awajun_backtranslated | chrf_score")

def main():
    parser = argparse.ArgumentParser(description="Back-translation con dos modelos")
    
    parser.add_argument('--model_agr2es', required=True,
                       help='Modelo para Awajún → Español')
    parser.add_argument('--model_es2agr', required=True,
                       help='Modelo para Español → Awajún')
    parser.add_argument('--input_file', required=True,
                       help='Archivo con columnas: document_id|segment_id|awajun_text')
    parser.add_argument('--output_file', 
                       help='Archivo de salida (default: input_file con sufijo _evaluated)')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size (default: 16)')
    
    args = parser.parse_args()
    
    # Determinar output_file
    if not args.output_file:
        input_path = Path(args.input_file)
        args.output_file = str(input_path.parent / f"{input_path.stem}_synthetic{input_path.suffix}")
    
    # Archivos temporales
    temp_dir = Path(args.input_file).parent / "temp_backtrans"
    temp_dir.mkdir(exist_ok=True)
    
    agr_text = temp_dir / "original.agr"
    es_text = temp_dir / "synthetic.es"
    agr_back = temp_dir / "backtranslated.agr"
    
    print("=" * 80)
    print("BACK-TRANSLATION Y EVALUACIÓN")
    print("=" * 80)
    print(f"Modelo AGR→ES: {args.model_agr2es}")
    print(f"Modelo ES→AGR: {args.model_es2agr}")
    print(f"Entrada:       {args.input_file}")
    print(f"Salida:        {args.output_file}")
    print()
    
    try:
        # Paso 1: Extraer texto
        print("PASO 1: Extrayendo texto original...")
        extract_text_column(args.input_file, agr_text)
        
        # Paso 2: Traducir AGR → ES
        print("\nPASO 2: Traduciendo Awajún → Español")
        translate_file(args.model_agr2es, "agr2es", agr_text, es_text, args.batch_size)
        
        # Paso 3: Back-traducir ES → AGR
        print("\nPASO 3: Back-traduciendo Español → Awajún")
        translate_file(args.model_es2agr, "es2agr", es_text, agr_back, args.batch_size)
        
        # Paso 4: Calcular métricas
        print("\nPASO 4: Calculando métricas...")
        chrf_scores, global_bleu, global_chrf = calculate_chrf(agr_text, agr_back)
        
        # Paso 5: Crear archivo final
        print("\nPASO 5: Creando archivo de salida...")
        create_output_file(args.input_file, es_text, agr_back, chrf_scores,
                          args.output_file, global_bleu, global_chrf)
        
        # Limpiar archivos temporales
        import shutil
        shutil.rmtree(temp_dir)
        print(f"\n🗑️  Archivos temporales eliminados")
        
        print("\n" + "=" * 80)
        print("✅ PROCESO COMPLETADO")
        print("=" * 80)
        print(f"📊 BLEU global:   {global_bleu:.2f}")
        print(f"📊 chrF++ global: {global_chrf:.2f}")
        print(f"📁 Datos:         {args.output_file}")
        print(f"📊 Métricas:      {args.output_file.replace('.txt', '_metrics.json')}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

if __name__ == "__main__":
    main()