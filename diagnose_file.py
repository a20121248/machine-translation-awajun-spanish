#!/usr/bin/env python3
"""
Diagnóstico de líneas problemáticas en archivo de backtranslation
Identifica y muestra líneas con formato incorrecto
"""

import sys
from pathlib import Path

def diagnose_file(input_file, output_file=None, show_samples=True):
    """Diagnostica problemas en el archivo"""
    
    print("=" * 80)
    print("DIAGNÓSTICO DE ARCHIVO DE BACKTRANSLATION")
    print("=" * 80)
    print(f"Archivo: {input_file}\n")
    
    if not Path(input_file).exists():
        print(f"❌ Error: Archivo no encontrado")
        return
    
    good_lines = []
    bad_lines = []
    line_stats = {
        'total': 0,
        'header': 0,
        'good': 0,
        'bad': 0,
        'empty': 0
    }
    
    column_count = {}
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, start=1):
            line_stats['total'] += 1
            
            # Saltar línea vacía
            if not line.strip():
                line_stats['empty'] += 1
                continue
            
            # Header
            if line_num == 1:
                line_stats['header'] += 1
                expected_header = line.strip()
                print(f"📋 Header detectado:")
                print(f"   {expected_header}")
                print(f"   Columnas esperadas: 6")
                print()
                continue
            
            # Analizar línea
            parts = line.strip().split('|')
            num_cols = len(parts)
            
            # Contar distribución de columnas
            column_count[num_cols] = column_count.get(num_cols, 0) + 1
            
            if num_cols == 6:
                line_stats['good'] += 1
                good_lines.append((line_num, line, parts))
            else:
                line_stats['bad'] += 1
                bad_lines.append((line_num, line, parts, num_cols))
    
    # Mostrar estadísticas
    print("📊 ESTADÍSTICAS:")
    print(f"   Total líneas: {line_stats['total']}")
    print(f"   Header: {line_stats['header']}")
    print(f"   Líneas buenas: {line_stats['good']} ✅")
    print(f"   Líneas malas: {line_stats['bad']} ❌")
    print(f"   Líneas vacías: {line_stats['empty']}")
    print()
    
    # Distribución de columnas
    print("📈 DISTRIBUCIÓN DE COLUMNAS:")
    for num_cols in sorted(column_count.keys()):
        count = column_count[num_cols]
        pct = (count / (line_stats['total'] - line_stats['header'] - line_stats['empty'])) * 100
        status = "✅" if num_cols == 6 else "❌"
        print(f"   {num_cols} columnas: {count:6,} líneas ({pct:5.1f}%) {status}")
    print()
    
    # Mostrar muestras de líneas malas
    if bad_lines and show_samples:
        print("=" * 80)
        print("🔍 MUESTRAS DE LÍNEAS PROBLEMÁTICAS:")
        print("=" * 80)
        
        # Mostrar primeras 10 líneas malas
        for i, (line_num, line, parts, num_cols) in enumerate(bad_lines[:10], start=1):
            print(f"\n🔴 Línea {line_num} (tiene {num_cols} columnas en lugar de 6):")
            print(f"   Contenido: {line[:150]}{'...' if len(line) > 150 else ''}")
            print(f"   Columnas detectadas:")
            for j, part in enumerate(parts):
                preview = part[:60] + "..." if len(part) > 60 else part
                print(f"      [{j}] {preview}")
        
        if len(bad_lines) > 10:
            print(f"\n... y {len(bad_lines) - 10} líneas más con problemas")
    
    # Guardar líneas malas a archivo
    if output_file and bad_lines:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# LÍNEAS PROBLEMÁTICAS\n")
            f.write(f"# Total: {len(bad_lines)} líneas\n")
            f.write("# Formato: LINEA_NUM | NUM_COLUMNAS | CONTENIDO\n\n")
            
            for line_num, line, parts, num_cols in bad_lines:
                f.write(f"{line_num}|{num_cols}|{line}")
        
        print(f"\n💾 Líneas problemáticas guardadas en: {output_file}")
    
    # Análisis de patrones
    print("\n" + "=" * 80)
    print("🔍 ANÁLISIS DE PATRONES:")
    print("=" * 80)
    
    if bad_lines:
        # Analizar dónde están los pipes extra
        pipe_positions = []
        for line_num, line, parts, num_cols in bad_lines[:100]:  # Analizar primeras 100
            # Intentar identificar cuál columna tiene el problema
            if num_cols > 6:
                extra_pipes = num_cols - 6
                pipe_positions.append(extra_pipes)
        
        if pipe_positions:
            from collections import Counter
            pipe_dist = Counter(pipe_positions)
            print(f"\n📍 Pipes extra por línea:")
            for extra, count in pipe_dist.most_common(5):
                print(f"   {extra} pipe(s) extra: {count} líneas")
        
        # Verificar si el problema es consistente
        if len(set(num_cols for _, _, _, num_cols in bad_lines)) == 1:
            consistent_cols = bad_lines[0][3]
            print(f"\n⚠️  PROBLEMA CONSISTENTE: Todas las líneas malas tienen {consistent_cols} columnas")
            print(f"   Esto sugiere un problema sistemático en la generación del archivo")
        else:
            print(f"\n⚠️  PROBLEMA VARIABLE: Las líneas tienen diferente número de columnas")
            print(f"   Esto sugiere pipes '|' dentro del contenido de texto")
    else:
        print("✅ No se encontraron patrones problemáticos")
    
    # Recomendaciones
    print("\n" + "=" * 80)
    print("💡 RECOMENDACIONES:")
    print("=" * 80)
    
    if not bad_lines:
        print("✅ El archivo está perfecto, no requiere corrección")
    elif line_stats['bad'] < line_stats['good'] * 0.01:  # Menos del 1%
        print("⚠️  Pocas líneas problemáticas (<1%), puedes:")
        print("   1. Ignorarlas (usar on_bad_lines='skip' en pandas)")
        print("   2. Corregirlas manualmente")
    else:
        print("❌ Muchas líneas problemáticas, opciones:")
        print("   1. Si el patrón es consistente: crear script de corrección automática")
        print("   2. Revisar el proceso de generación del archivo")
        print("   3. Usar el script de limpieza que puedo generarte")
    
    print("\n" + "=" * 80)
    
    return {
        'stats': line_stats,
        'good_lines': len(good_lines),
        'bad_lines': len(bad_lines),
        'column_distribution': column_count
    }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python diagnose_file.py <archivo> [archivo_salida_errores]")
        print("\nEjemplo:")
        print("  python diagnose_file.py data/synthetic.txt")
        print("  python diagnose_file.py data/synthetic.txt bad_lines.txt")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    diagnose_file(input_file, output_file)