#!/usr/bin/env python3
"""
Generador de datasets combinados: bilingüe + sintético filtrado
Crea múltiples versiones con diferentes thresholds de calidad
"""

import pandas as pd
from pathlib import Path
import shutil
import json

def create_filtered_datasets(
    synthetic_file,
    base_dataset_dir,
    output_base_dir="data",
    thresholds=None
):
    """
    Genera datasets combinando corpus bilingüe base + corpus sintético filtrado
    
    Args:
        synthetic_file: Ruta al archivo awajun_corpus_monolingue_synthetic.txt
        base_dataset_dir: Ruta a ./data/awajun-spanish-v1/
        output_base_dir: Directorio base de salida (default: "data")
        thresholds: Dict con {nombre: percentil} o None para usar defaults
    """
    
    if thresholds is None:
        thresholds = {
            'top20': 0.80,  # Top 20% = percentil 80
            'top40': 0.60,  # Top 40% = percentil 60
            'top60': 0.40,  # Top 60% = percentil 40
            'top80': 0.20,  # Top 80% = percentil 20
            'complete': 0.00  # 100% = sin filtro
        }
    
    print("=" * 80)
    print("GENERADOR DE DATASETS SINTÉTICOS FILTRADOS")
    print("=" * 80)
    
    # 1. Leer corpus sintético con lectura manual (evita problemas de pandas con comillas)
    print("\n📊 Cargando corpus sintético con lectura manual...")
    
    synthetic_data = []
    with open(synthetic_file, 'r', encoding='utf-8') as f:
        header = f.readline().strip().split('|')
        
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split('|')
            
            if len(parts) == 6:
                synthetic_data.append(parts)
            elif len(parts) > 6:
                # Reconstruir si hay pipes extra en el texto
                fixed = [
                    parts[0],              # document_id
                    parts[1],              # segment_id
                    '|'.join(parts[2:-3]), # awajun_text
                    parts[-3],             # spanish_synthetic
                    parts[-2],             # awajun_backtranslated
                    parts[-1]              # chrf_score
                ]
                synthetic_data.append(fixed)
            # Si tiene menos de 6 columnas, la saltamos silenciosamente
    
    print(f"   Líneas leídas manualmente: {len(synthetic_data):,}")
    
    # Crear DataFrame
    df_synthetic = pd.DataFrame(synthetic_data, columns=header)
    df_synthetic['chrf_score'] = pd.to_numeric(df_synthetic['chrf_score'], errors='coerce')
    df_synthetic = df_synthetic.dropna(subset=['chrf_score'])
    
    total_synthetic = len(df_synthetic)
    print(f"✅ Corpus sintético cargado: {total_synthetic:,} líneas")
    print(f"   chrF++ medio: {df_synthetic['chrf_score'].mean():.2f}")
    
    # 2. Leer corpus bilingüe base
    print(f"\n📂 Cargando corpus bilingüe base desde: {base_dataset_dir}")
    base_path = Path(base_dataset_dir)
    
    with open(base_path / 'train.agr', 'r', encoding='utf-8') as f:
        base_agr = [line.strip() for line in f.readlines()]
    
    with open(base_path / 'train.es', 'r', encoding='utf-8') as f:
        base_es = [line.strip() for line in f.readlines()]
    
    with open(base_path / 'train.source', 'r', encoding='utf-8') as f:
        base_source = [line.strip() for line in f.readlines()]
    
    print(f"✅ Corpus bilingüe cargado: {len(base_agr):,} pares")
    
    # 3. Generar datasets filtrados
    print("\n🔧 Generando datasets filtrados...")
    print("-" * 80)
    
    results_summary = []
    
    for name, percentile in thresholds.items():
        print(f"\n📦 Procesando: {name.upper()}")
        
        # Calcular threshold
        if percentile > 0:
            threshold = df_synthetic['chrf_score'].quantile(percentile)
            df_filtered = df_synthetic[df_synthetic['chrf_score'] >= threshold]
        else:
            threshold = 0.0
            df_filtered = df_synthetic
        
        synthetic_count = len(df_filtered)
        avg_chrf = df_filtered['chrf_score'].mean()
        
        print(f"   Threshold chrF++: {threshold:.2f}")
        print(f"   Líneas sintéticas: {synthetic_count:,} ({(synthetic_count/total_synthetic)*100:.1f}%)")
        print(f"   chrF++ promedio: {avg_chrf:.2f}")
        
        # Extraer textos sintéticos
        synthetic_agr = df_filtered['awajun_text'].astype(str).tolist()
        synthetic_es = df_filtered['spanish_synthetic'].astype(str).tolist()
        
        # Extraer source: todas las líneas sintéticas son "Education"
        # (materiales educativos del Ministerio de Educación)
        synthetic_source = ['Education'] * synthetic_count
        
        # Combinar: base + sintético
        combined_agr = base_agr + synthetic_agr
        combined_es = base_es + synthetic_es
        combined_source = base_source + synthetic_source
        
        total_combined = len(combined_agr)
        
        print(f"   Total combinado: {total_combined:,} pares")
        print(f"   Proporción sintética: {(synthetic_count/total_combined)*100:.1f}%")
        
        # Crear directorio de salida
        if name == 'complete':
            output_dir = Path(output_base_dir) / 'awajun-spanish-v3'
        else:
            output_dir = Path(output_base_dir) / f'awajun-spanish-v3-{name}'
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Guardar archivos
        with open(output_dir / 'train.agr', 'w', encoding='utf-8') as f:
            f.write('\n'.join(combined_agr) + '\n')
        
        with open(output_dir / 'train.es', 'w', encoding='utf-8') as f:
            f.write('\n'.join(combined_es) + '\n')
        
        with open(output_dir / 'train.source', 'w', encoding='utf-8') as f:
            f.write('\n'.join(combined_source) + '\n')
        
        # Copiar archivos dev
        for ext in ['agr', 'es', 'source']:
            src_file = base_path / f'dev.{ext}'
            dst_file = output_dir / f'dev.{ext}'
            if src_file.exists():
                shutil.copy(src_file, dst_file)
        
        # Verificar que todos tengan el mismo número de líneas
        agr_lines = len(combined_agr)
        es_lines = len(combined_es)
        source_lines = len(combined_source)
        
        if agr_lines == es_lines == source_lines:
            print(f"   ✅ Verificado: {agr_lines:,} líneas en todos los archivos")
        else:
            print(f"   ❌ ERROR: agr={agr_lines}, es={es_lines}, source={source_lines}")
        
        # Guardar metadata
        metadata = {
            'dataset_name': output_dir.name,
            'base_corpus': str(base_path),
            'synthetic_corpus': str(synthetic_file),
            'threshold_percentile': percentile,
            'threshold_chrf': round(threshold, 2),
            'base_pairs': len(base_agr),
            'synthetic_pairs': synthetic_count,
            'total_pairs': total_combined,
            'synthetic_percentage': round((synthetic_count/total_combined)*100, 2),
            'avg_chrf_synthetic': round(avg_chrf, 2)
        }
        
        with open(output_dir / 'dataset_info.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"   ✅ Guardado en: {output_dir}")
        
        results_summary.append({
            'Dataset': output_dir.name,
            'Threshold': f'{threshold:.2f}',
            'Base': f'{len(base_agr):,}',
            'Sintético': f'{synthetic_count:,}',
            'Total': f'{total_combined:,}',
            '% Sintético': f'{(synthetic_count/total_combined)*100:.1f}%',
            'chrF++ Avg': f'{avg_chrf:.2f}'
        })
    
    # 4. Resumen final
    print("\n" + "=" * 80)
    print("✅ PROCESO COMPLETADO")
    print("=" * 80)
    print("\n📊 RESUMEN DE DATASETS GENERADOS:\n")
    
    # Tabla resumen
    df_summary = pd.DataFrame(results_summary)
    print(df_summary.to_string(index=False))
    
    # Guardar resumen
    summary_file = Path(output_base_dir) / 'synthetic_datasets_summary.txt'
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("RESUMEN DE DATASETS SINTÉTICOS GENERADOS\n")
        f.write("=" * 80 + "\n\n")
        f.write(df_summary.to_string(index=False))
        f.write("\n\n")
        f.write("ARCHIVOS GENERADOS POR DATASET:\n")
        f.write("-" * 80 + "\n")
        for result in results_summary:
            f.write(f"\n{result['Dataset']}/\n")
            f.write(f"  ├── train.agr ({result['Total']} líneas)\n")
            f.write(f"  ├── train.es ({result['Total']} líneas)\n")
            f.write(f"  ├── train.source ({result['Total']} líneas)\n")
            f.write(f"  ├── dev.agr, dev.es, dev.source\n")
            f.write(f"  └── dataset_info.json\n")
    
    print(f"\n📄 Resumen guardado en: {summary_file}")
    print(f"\nDatasets generados en: {output_base_dir}/awajun-spanish-v3*")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Genera datasets combinados con corpus sintético filtrado"
    )
    
    parser.add_argument(
        '--synthetic_file',
        default='data/monolingual-corpus/awajun_corpus_monolingue_synthetic.txt',
        help='Archivo de corpus sintético con métricas'
    )
    
    parser.add_argument(
        '--base_dataset',
        default='data/awajun-spanish-v1',
        help='Directorio del corpus bilingüe base'
    )
    
    parser.add_argument(
        '--output_dir',
        default='data',
        help='Directorio base de salida'
    )
    
    args = parser.parse_args()
    
    create_filtered_datasets(
        synthetic_file=args.synthetic_file,
        base_dataset_dir=args.base_dataset,
        output_base_dir=args.output_dir
    )