#!/usr/bin/env python3
"""
Análisis estadístico comparativo entre datasets V1 (base) y V3 (base + sintético)
"""

import json
from pathlib import Path
from collections import Counter

def analyze_corpus(file_path, name):
    """Analizar estadísticas de un corpus"""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    # Estadísticas de palabras
    all_words = []
    sentence_lengths = []
    char_lengths = []
    
    for line in lines:
        words = line.split()
        all_words.extend(words)
        sentence_lengths.append(len(words))
        char_lengths.append(len(line))
    
    # Calcular estadísticas
    total_sentences = len(lines)
    total_words = len(all_words)
    unique_words = len(set(all_words))
    avg_words_per_sentence = total_words / total_sentences if total_sentences > 0 else 0
    avg_chars_per_sentence = sum(char_lengths) / total_sentences if total_sentences > 0 else 0
    median_words = sorted(sentence_lengths)[total_sentences // 2] if total_sentences > 0 else 0
    
    # Distribución por longitud
    length_dist = Counter(sentence_lengths)
    
    stats = {
        'name': name,
        'total_sentences': total_sentences,
        'total_words': total_words,
        'unique_words': unique_words,
        'avg_words_per_sentence': round(avg_words_per_sentence, 2),
        'median_words': median_words,
        'avg_chars_per_sentence': round(avg_chars_per_sentence, 2),
        'min_words': min(sentence_lengths) if sentence_lengths else 0,
        'max_words': max(sentence_lengths) if sentence_lengths else 0,
        'vocab_size': unique_words,
        'length_distribution': {
            '4_words': sum(v for k, v in length_dist.items() if k == 4),
            '6_10_words': sum(v for k, v in length_dist.items() if 6 <= k <= 10),
            '11_20_words': sum(v for k, v in length_dist.items() if 11 <= k <= 20),
            'over_20_words': sum(v for k, v in length_dist.items() if k > 20)
        }
    }
    
    return stats

def compare_datasets(v1_agr, v1_es, v3_agr, v3_es):
    """Comparar datasets V1 (base) y V3 (base + sintético)"""
    
    print("="*80)
    print("ANÁLISIS COMPARATIVO: DATASET V1 (BASE) vs V3 (BASE + SINTÉTICO)")
    print("="*80)
    print()
    
    # Analizar V1
    print("📦 DATASET V1 (CORPUS BASE ORIGINAL)")
    print("-"*80)
    stats_v1_agr = analyze_corpus(v1_agr, "V1 - Awajún")
    stats_v1_es = analyze_corpus(v1_es, "V1 - Español")
    
    for stats in [stats_v1_agr, stats_v1_es]:
        print(f"\n📊 {stats['name']}")
        print(f"   Total oraciones: {stats['total_sentences']:,}")
        print(f"   Total palabras: {stats['total_words']:,}")
        print(f"   Vocabulario único: {stats['vocab_size']:,}")
        print(f"   Promedio palabras/oración: {stats['avg_words_per_sentence']}")
        print(f"   Mediana: {stats['median_words']} palabras")
        print(f"   Rango: {stats['min_words']}-{stats['max_words']} palabras")
        print(f"   Promedio caracteres/oración: {stats['avg_chars_per_sentence']}")
        
        dist = stats['length_distribution']
        total = stats['total_sentences']
        print(f"\n   Distribución:")
        print(f"   - 4 palabras: {dist['4_words']} ({dist['4_words']/total*100:.1f}%)")
        print(f"   - 6-10 palabras: {dist['6_10_words']} ({dist['6_10_words']/total*100:.1f}%)")
        print(f"   - 11-20 palabras: {dist['11_20_words']} ({dist['11_20_words']/total*100:.1f}%)")
        print(f"   - Más de 20 palabras: {dist['over_20_words']} ({dist['over_20_words']/total*100:.1f}%)")
    
    print("\n" + "="*80)
    print()
    
    # Analizar V3
    print("📦 DATASET V3 (BASE + DATOS SINTÉTICOS)")
    print("-"*80)
    stats_v3_agr = analyze_corpus(v3_agr, "V3 - Awajún")
    stats_v3_es = analyze_corpus(v3_es, "V3 - Español")
    
    for stats in [stats_v3_agr, stats_v3_es]:
        print(f"\n📊 {stats['name']}")
        print(f"   Total oraciones: {stats['total_sentences']:,}")
        print(f"   Total palabras: {stats['total_words']:,}")
        print(f"   Vocabulario único: {stats['vocab_size']:,}")
        print(f"   Promedio palabras/oración: {stats['avg_words_per_sentence']}")
        print(f"   Mediana: {stats['median_words']} palabras")
        print(f"   Rango: {stats['min_words']}-{stats['max_words']} palabras")
        print(f"   Promedio caracteres/oración: {stats['avg_chars_per_sentence']}")
        
        dist = stats['length_distribution']
        total = stats['total_sentences']
        print(f"\n   Distribución:")
        print(f"   - 4 palabras: {dist['4_words']} ({dist['4_words']/total*100:.1f}%)")
        print(f"   - 6-10 palabras: {dist['6_10_words']} ({dist['6_10_words']/total*100:.1f}%)")
        print(f"   - 11-20 palabras: {dist['11_20_words']} ({dist['11_20_words']/total*100:.1f}%)")
        print(f"   - Más de 20 palabras: {dist['over_20_words']} ({dist['over_20_words']/total*100:.1f}%)")
    
    print("\n" + "="*80)
    print()
    
    # Comparación directa
    print("📈 COMPARACIÓN V1 → V3 (IMPACTO DEL AUMENTO DE DATOS)")
    print("-"*80)
    
    print("\n🔤 AWAJÚN:")
    sent_inc_agr = stats_v3_agr['total_sentences'] - stats_v1_agr['total_sentences']
    word_inc_agr = stats_v3_agr['total_words'] - stats_v1_agr['total_words']
    vocab_inc_agr = stats_v3_agr['vocab_size'] - stats_v1_agr['vocab_size']
    avg_diff_agr = stats_v3_agr['avg_words_per_sentence'] - stats_v1_agr['avg_words_per_sentence']
    
    print(f"   Incremento oraciones: {sent_inc_agr:,} ({sent_inc_agr/stats_v1_agr['total_sentences']*100:.1f}%)")
    print(f"   Incremento palabras: {word_inc_agr:,} ({word_inc_agr/stats_v1_agr['total_words']*100:.1f}%)")
    print(f"   Incremento vocabulario: {vocab_inc_agr:,} ({vocab_inc_agr/stats_v1_agr['vocab_size']*100:.1f}%)")
    print(f"   Diferencia promedio palabras/oración: {avg_diff_agr:.2f}")
    
    print("\n🇪🇸 ESPAÑOL:")
    sent_inc_es = stats_v3_es['total_sentences'] - stats_v1_es['total_sentences']
    word_inc_es = stats_v3_es['total_words'] - stats_v1_es['total_words']
    vocab_inc_es = stats_v3_es['vocab_size'] - stats_v1_es['vocab_size']
    avg_diff_es = stats_v3_es['avg_words_per_sentence'] - stats_v1_es['avg_words_per_sentence']
    
    print(f"   Incremento oraciones: {sent_inc_es:,} ({sent_inc_es/stats_v1_es['total_sentences']*100:.1f}%)")
    print(f"   Incremento palabras: {word_inc_es:,} ({word_inc_es/stats_v1_es['total_words']*100:.1f}%)")
    print(f"   Incremento vocabulario: {vocab_inc_es:,} ({vocab_inc_es/stats_v1_es['vocab_size']*100:.1f}%)")
    print(f"   Diferencia promedio palabras/oración: {avg_diff_es:.2f}")
    
    # Guardar en JSON
    output = {
        'dataset_v1': {
            'awajun': stats_v1_agr,
            'espanol': stats_v1_es
        },
        'dataset_v3': {
            'awajun': stats_v3_agr,
            'espanol': stats_v3_es
        },
        'comparison': {
            'awajun': {
                'sentence_increase': sent_inc_agr,
                'sentence_increase_pct': round(sent_inc_agr/stats_v1_agr['total_sentences']*100, 1),
                'word_increase': word_inc_agr,
                'word_increase_pct': round(word_inc_agr/stats_v1_agr['total_words']*100, 1),
                'vocab_increase': vocab_inc_agr,
                'vocab_increase_pct': round(vocab_inc_agr/stats_v1_agr['vocab_size']*100, 1),
                'avg_length_diff': round(avg_diff_agr, 2)
            },
            'espanol': {
                'sentence_increase': sent_inc_es,
                'sentence_increase_pct': round(sent_inc_es/stats_v1_es['total_sentences']*100, 1),
                'word_increase': word_inc_es,
                'word_increase_pct': round(word_inc_es/stats_v1_es['total_words']*100, 1),
                'vocab_increase': vocab_inc_es,
                'vocab_increase_pct': round(vocab_inc_es/stats_v1_es['vocab_size']*100, 1),
                'avg_length_diff': round(avg_diff_es, 2)
            }
        }
    }
    
    output_file = 'dataset_v1_v3_comparison.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Estadísticas guardadas en: {output_file}")
    print("="*80)

if __name__ == "__main__":
    # Rutas de los datasets
    v1_agr = "./data/awajun-spanish-v1/train.agr"
    v1_es = "./data/awajun-spanish-v1/train.es"
    v3_agr = "./data/awajun-spanish-v3/train.agr"
    v3_es = "./data/awajun-spanish-v3/train.es"
    
    # Verificar que existen los archivos
    for file in [v1_agr, v1_es, v3_agr, v3_es]:
        if not Path(file).exists():
            print(f"❌ Error: No se encuentra el archivo {file}")
            exit(1)
    
    # Realizar comparación
    compare_datasets(v1_agr, v1_es, v3_agr, v3_es)