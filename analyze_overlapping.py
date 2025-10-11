#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Análisis de corpus paralelo Awajún-Español para fine-tuning NLLB
Compara overlaps entre train/dev y analiza datos nuevos
"""

import os
from collections import Counter
import re

def clean_text(text):
    """Limpia y normaliza texto para comparación"""
    # Convertir a minúsculas y quitar espacios extra
    text = text.lower().strip()
    # Opcional: quitar signos de puntuación para análisis más limpio
    # text = re.sub(r'[^\w\s]', ' ', text)
    return text

def analyze_file(filepath):
    """Analiza un archivo y retorna estadísticas básicas"""
    if not os.path.exists(filepath):
        print(f"⚠️  Archivo no encontrado: {filepath}")
        return None
    
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = [clean_text(line) for line in f.readlines() if line.strip()]
    
    # Contar oraciones
    num_sentences = len(lines)
    
    # Contar palabras totales y únicas
    all_words = []
    for line in lines:
        words = line.split()
        all_words.extend(words)
    
    total_words = len(all_words)
    unique_words = set(all_words)
    num_unique_words = len(unique_words)
    
    return {
        'sentences': num_sentences,
        'total_words': total_words,
        'unique_words': unique_words,
        'num_unique_words': num_unique_words,
        'lines': lines
    }

def compare_datasets(train_data, dev_data, dataset_name=""):
    """Compara dos datasets y muestra estadísticas de overlap"""
    print(f"\n{'='*60}")
    print(f"COMPARACIÓN: {dataset_name}")
    print(f"{'='*60}")
    
    # Estadísticas básicas
    print(f"\n📊 ESTADÍSTICAS BÁSICAS:")
    print(f"Train - Oraciones: {train_data['sentences']:,}, Palabras totales: {train_data['total_words']:,}, Palabras únicas: {train_data['num_unique_words']:,}")
    print(f"Dev   - Oraciones: {dev_data['sentences']:,}, Palabras totales: {dev_data['total_words']:,}, Palabras únicas: {dev_data['num_unique_words']:,}")
    
    # Análisis de overlap a nivel de oraciones
    train_sentences = set(train_data['lines'])
    dev_sentences = set(dev_data['lines'])
    
    sentence_overlap = train_sentences.intersection(dev_sentences)
    sentence_overlap_pct = (len(sentence_overlap) / len(dev_sentences)) * 100 if dev_sentences else 0
    
    print(f"\n🔄 OVERLAP DE ORACIONES:")
    print(f"Oraciones idénticas: {len(sentence_overlap)} ({sentence_overlap_pct:.1f}% del dev set)")
    
    # Análisis de overlap a nivel de palabras
    word_overlap = train_data['unique_words'].intersection(dev_data['unique_words'])
    word_overlap_pct = (len(word_overlap) / len(dev_data['unique_words'])) * 100 if dev_data['unique_words'] else 0
    
    print(f"\n📝 OVERLAP DE VOCABULARIO:")
    print(f"Palabras compartidas: {len(word_overlap)} ({word_overlap_pct:.1f}% del vocabulario de dev)")
    
    # Palabras nuevas en dev
    new_words_in_dev = dev_data['unique_words'] - train_data['unique_words']
    new_words_pct = (len(new_words_in_dev) / len(dev_data['unique_words'])) * 100 if dev_data['unique_words'] else 0
    
    print(f"Palabras nuevas en dev: {len(new_words_in_dev)} ({new_words_pct:.1f}% del vocabulario de dev)")
    
    if len(new_words_in_dev) > 0 and len(new_words_in_dev) <= 20:
        print(f"Palabras nuevas: {sorted(list(new_words_in_dev))}")
    
    return {
        'sentence_overlap': len(sentence_overlap),
        'sentence_overlap_pct': sentence_overlap_pct,
        'word_overlap': len(word_overlap),
        'word_overlap_pct': word_overlap_pct,
        'new_words_in_dev': len(new_words_in_dev),
        'new_words_pct': new_words_pct
    }

def main():
    # Ruta base del proyecto
    base_path = "/home/jmonzon/machine-translation-awajun-spanish_backup"
    
    print("🔍 ANÁLISIS DE CORPUS AWAJÚN-ESPAÑOL PARA FINE-TUNING NLLB")
    print("="*70)
    
    # PASO 1: Comparar v1 train.agr vs dev.agr
    print("\n📁 PASO 1: Analizando datos base (v1)")
    
    v1_train_path = os.path.join(base_path, "data", "awajun-spanish-v1", "train.agr")
    v1_dev_path = os.path.join(base_path, "data", "awajun-spanish-v1", "dev.agr")
    
    v1_train_data = analyze_file(v1_train_path)
    v1_dev_data = analyze_file(v1_dev_path)
    
    if v1_train_data and v1_dev_data:
        v1_comparison = compare_datasets(v1_train_data, v1_dev_data, "V1: Train vs Dev")
    
    # PASO 2: Analizar v2 y comparar con v1
    print("\n📁 PASO 2: Analizando datos nuevos (v2)")
    
    v2_train_path = os.path.join(base_path, "data", "awajun-spanish-v2", "train.agr")
    v2_train_data = analyze_file(v2_train_path)
    
    if v1_dev_data and v2_train_data:
        # Comparar v2 train con v1 dev (para ver overlap con datos de evaluación)
        v2_vs_v1_dev = compare_datasets(v2_train_data, v1_dev_data, "V2 Train vs V1 Dev")
        
        # También comparar v2 train con v1 train (para ver cuánto dato nuevo hay)
        if v1_train_data:
            print(f"\n🆕 ANÁLISIS DE DATOS NUEVOS (V2 vs V1):")
            print(f"{'='*60}")
            
            # Palabras nuevas en v2 respecto a v1 train
            new_words_v2 = v2_train_data['unique_words'] - v1_train_data['unique_words']
            new_words_v2_pct = (len(new_words_v2) / len(v2_train_data['unique_words'])) * 100
            
            print(f"Palabras totales en V2: {v2_train_data['num_unique_words']:,}")
            print(f"Palabras nuevas respecto a V1: {len(new_words_v2):,} ({new_words_v2_pct:.1f}%)")
            
            # Oraciones nuevas en v2 respecto a v1 train
            v1_train_sentences = set(v1_train_data['lines'])
            v2_train_sentences = set(v2_train_data['lines'])
            new_sentences_v2 = v2_train_sentences - v1_train_sentences
            new_sentences_v2_pct = (len(new_sentences_v2) / len(v2_train_sentences)) * 100
            
            print(f"Oraciones totales en V2: {v2_train_data['sentences']:,}")
            print(f"Oraciones nuevas respecto a V1: {len(new_sentences_v2):,} ({new_sentences_v2_pct:.1f}%)")
    
    # CONCLUSIONES
    print(f"\n🎯 CONCLUSIONES PARA FINE-TUNING:")
    print(f"{'='*60}")
    
    if v1_train_data and v1_dev_data:
        if v1_comparison['word_overlap_pct'] < 70:
            print(f"✅ Bajo overlap de vocabulario entre train y dev ({v1_comparison['word_overlap_pct']:.1f}%)")
            print(f"   → Las métricas en dev deberían ser estables durante fine-tuning")
        else:
            print(f"⚠️  Alto overlap de vocabulario entre train y dev ({v1_comparison['word_overlap_pct']:.1f}%)")
            print(f"   → Posible overfitting, las métricas podrían mejorar artificialmente")
        
        if v1_comparison['sentence_overlap_pct'] > 5:
            print(f"⚠️  Hay {v1_comparison['sentence_overlap_pct']:.1f}% de oraciones duplicadas entre train y dev")
            print(f"   → Considerar limpiar duplicados para evaluación más robusta")
    
    if v2_train_data and v1_dev_data:
        if v2_vs_v1_dev['word_overlap_pct'] < 70:
            print(f"✅ Los datos V2 tienen bajo overlap con dev ({v2_vs_v1_dev['word_overlap_pct']:.1f}%)")
            print(f"   → El fine-tuning con V2 no debería inflar las métricas artificialmente")

if __name__ == "__main__":
    main()