#!/usr/bin/env python3
"""
Análisis del corpus v1 para entender ratios de longitud awajún-español
"""

import numpy as np
import matplotlib.pyplot as plt
import os

def analyze_corpus_ratios(agr_file, es_file, dataset_name=""):
    """Analizar ratios de longitud en corpus existente"""
    
    print(f"Analizando corpus: {dataset_name}")
    print("=" * 50)
    
    # Leer archivos
    with open(agr_file, 'r', encoding='utf-8') as f:
        agr_lines = [line.strip() for line in f.readlines()]
    
    with open(es_file, 'r', encoding='utf-8') as f:
        es_lines = [line.strip() for line in f.readlines()]
    
    print(f"Líneas awajún: {len(agr_lines)}")
    print(f"Líneas español: {len(es_lines)}")
    
    if len(agr_lines) != len(es_lines):
        print("ADVERTENCIA: Número de líneas no coincide")
        min_lines = min(len(agr_lines), len(es_lines))
        agr_lines = agr_lines[:min_lines]
        es_lines = es_lines[:min_lines]
    
    # Calcular estadísticas
    char_ratios = []
    word_ratios = []
    agr_lengths = []
    es_lengths = []
    agr_word_counts = []
    es_word_counts = []
    
    for agr, es in zip(agr_lines, es_lines):
        if agr.strip() and es.strip():
            # Caracteres
            agr_chars = len(agr)
            es_chars = len(es)
            char_ratio = es_chars / agr_chars
            
            # Palabras
            agr_words = len(agr.split())
            es_words = len(es.split())
            word_ratio = es_words / agr_words if agr_words > 0 else 0
            
            # Almacenar
            char_ratios.append(char_ratio)
            word_ratios.append(word_ratio)
            agr_lengths.append(agr_chars)
            es_lengths.append(es_chars)
            agr_word_counts.append(agr_words)
            es_word_counts.append(es_words)
    
    # Estadísticas de caracteres
    print("\n📏 ANÁLISIS DE CARACTERES")
    print("-" * 30)
    print(f"Promedio awajún: {np.mean(agr_lengths):.1f} caracteres")
    print(f"Promedio español: {np.mean(es_lengths):.1f} caracteres")
    print(f"Ratio promedio (es/agr): {np.mean(char_ratios):.2f}")
    print(f"Mediana ratio: {np.median(char_ratios):.2f}")
    print(f"Desviación estándar: {np.std(char_ratios):.2f}")
    print(f"Percentiles 5-95: {np.percentile(char_ratios, 5):.2f} - {np.percentile(char_ratios, 95):.2f}")
    print(f"Min-Max ratio: {min(char_ratios):.2f} - {max(char_ratios):.2f}")
    
    # Estadísticas de palabras
    print("\n🔤 ANÁLISIS DE PALABRAS")
    print("-" * 30)
    print(f"Promedio awajún: {np.mean(agr_word_counts):.1f} palabras")
    print(f"Promedio español: {np.mean(es_word_counts):.1f} palabras")
    print(f"Ratio promedio (es/agr): {np.mean(word_ratios):.2f}")
    print(f"Mediana ratio: {np.median(word_ratios):.2f}")
    print(f"Desviación estándar: {np.std(word_ratios):.2f}")
    print(f"Percentiles 5-95: {np.percentile(word_ratios, 5):.2f} - {np.percentile(word_ratios, 95):.2f}")
    print(f"Min-Max ratio: {min(word_ratios):.2f} - {max(word_ratios):.2f}")
    
    # Filtros recomendados
    print("\n🎯 FILTROS RECOMENDADOS")
    print("-" * 30)
    
    # Filtro conservador (percentiles 10-90)
    char_p10 = np.percentile(char_ratios, 10)
    char_p90 = np.percentile(char_ratios, 90)
    word_p10 = np.percentile(word_ratios, 10)
    word_p90 = np.percentile(word_ratios, 90)
    
    print(f"Filtro conservador caracteres: {char_p10:.2f} - {char_p90:.2f}")
    print(f"Filtro conservador palabras: {word_p10:.2f} - {word_p90:.2f}")
    
    # Filtro moderado (percentiles 5-95)
    char_p05 = np.percentile(char_ratios, 5)
    char_p95 = np.percentile(char_ratios, 95)
    word_p05 = np.percentile(word_ratios, 5)
    word_p95 = np.percentile(word_ratios, 95)
    
    print(f"Filtro moderado caracteres: {char_p05:.2f} - {char_p95:.2f}")
    print(f"Filtro moderado palabras: {word_p05:.2f} - {word_p95:.2f}")
    
    # Ejemplos extremos
    print("\n🔍 EJEMPLOS EXTREMOS")
    print("-" * 30)
    
    # Ratios muy bajos (awajún mucho más largo)
    low_ratios = [(i, char_ratios[i]) for i in range(len(char_ratios)) if char_ratios[i] < 0.5]
    if low_ratios:
        print("Ratios muy bajos (awajún >> español):")
        for i, ratio in low_ratios[:3]:
            print(f"  Ratio {ratio:.2f}: AGR='{agr_lines[i][:60]}...' ES='{es_lines[i][:60]}...'")
    
    # Ratios muy altos (español mucho más largo)
    high_ratios = [(i, char_ratios[i]) for i in range(len(char_ratios)) if char_ratios[i] > 2.0]
    if high_ratios:
        print("Ratios muy altos (español >> awajún):")
        for i, ratio in high_ratios[:3]:
            print(f"  Ratio {ratio:.2f}: AGR='{agr_lines[i][:60]}...' ES='{es_lines[i][:60]}...'")
    
    return {
        'char_ratios': char_ratios,
        'word_ratios': word_ratios,
        'agr_lengths': agr_lengths,
        'es_lengths': es_lengths,
        'filters': {
            'char_conservative': (char_p10, char_p90),
            'char_moderate': (char_p05, char_p95),
            'word_conservative': (word_p10, word_p90),
            'word_moderate': (word_p05, word_p95)
        }
    }

def create_visualization(stats, output_file="corpus_analysis.png"):
    """Crear gráficos de distribución"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Histograma de ratios de caracteres
    ax1.hist(stats['char_ratios'], bins=50, alpha=0.7, color='blue')
    ax1.set_xlabel('Ratio Caracteres (ES/AGR)')
    ax1.set_ylabel('Frecuencia')
    ax1.set_title('Distribución Ratio Caracteres')
    ax1.axvline(np.mean(stats['char_ratios']), color='red', linestyle='--', label='Promedio')
    ax1.legend()
    
    # Histograma de ratios de palabras
    ax2.hist(stats['word_ratios'], bins=50, alpha=0.7, color='green')
    ax2.set_xlabel('Ratio Palabras (ES/AGR)')
    ax2.set_ylabel('Frecuencia')
    ax2.set_title('Distribución Ratio Palabras')
    ax2.axvline(np.mean(stats['word_ratios']), color='red', linestyle='--', label='Promedio')
    ax2.legend()
    
    # Scatter plot longitudes
    ax3.scatter(stats['agr_lengths'], stats['es_lengths'], alpha=0.5)
    ax3.set_xlabel('Longitud Awajún (caracteres)')
    ax3.set_ylabel('Longitud Español (caracteres)')
    ax3.set_title('Correlación Longitudes')
    
    # Línea diagonal para referencia
    max_len = max(max(stats['agr_lengths']), max(stats['es_lengths']))
    ax3.plot([0, max_len], [0, max_len], 'r--', alpha=0.5, label='Igual longitud')
    ax3.legend()
    
    # Box plot de ratios
    ax4.boxplot([stats['char_ratios'], stats['word_ratios']], 
               labels=['Caracteres', 'Palabras'])
    ax4.set_ylabel('Ratio (ES/AGR)')
    ax4.set_title('Distribución de Ratios')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Gráfico guardado en: {output_file}")

def main():
    # Paths para train y dev
    base_path = "/home/jmonzon/machine-translation-awajun-spanish_backup/data/awajun-spanish-v1/"
    
    print("ANÁLISIS COMPLETO DEL CORPUS V1")
    print("=" * 60)
    
    # Analizar training set
    train_stats = analyze_corpus_ratios(
        os.path.join(base_path, "train.agr"),
        os.path.join(base_path, "train.es"),
        "Training Set"
    )
    
    print("\n" + "=" * 60)
    
    # Analizar dev set
    dev_stats = analyze_corpus_ratios(
        os.path.join(base_path, "dev.agr"),
        os.path.join(base_path, "dev.es"),
        "Development Set"
    )
    
    # Crear visualización
    create_visualization(train_stats, "train_corpus_analysis.png")
    
    # Generar código de filtros recomendados
    print("\n" + "=" * 60)
    print("CÓDIGO PYTHON PARA FILTROS")
    print("=" * 60)
    
    char_cons = train_stats['filters']['char_conservative']
    char_mod = train_stats['filters']['char_moderate']
    word_cons = train_stats['filters']['word_conservative']
    word_mod = train_stats['filters']['word_moderate']
    
    print("# Filtros basados en tu corpus v1:")
    print(f"""
def filter_conservative_chars(agr, es):
    ratio = len(es) / len(agr)
    return {char_cons[0]:.2f} <= ratio <= {char_cons[1]:.2f}

def filter_moderate_chars(agr, es):
    ratio = len(es) / len(agr)
    return {char_mod[0]:.2f} <= ratio <= {char_mod[1]:.2f}

def filter_conservative_words(agr, es):
    agr_words = len(agr.split())
    es_words = len(es.split())
    ratio = es_words / agr_words if agr_words > 0 else 0
    return {word_cons[0]:.2f} <= ratio <= {word_cons[1]:.2f}

def filter_moderate_words(agr, es):
    agr_words = len(agr.split())
    es_words = len(es.split())
    ratio = es_words / agr_words if agr_words > 0 else 0
    return {word_mod[0]:.2f} <= ratio <= {word_mod[1]:.2f}
""")

if __name__ == "__main__":
    main()