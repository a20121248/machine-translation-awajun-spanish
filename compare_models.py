#!/usr/bin/env python3
"""
Script para comparar múltiples modelos NLLB fine-tuneados
Uso: python compare_models.py --models model1 model2 model3 --direction es2agr --dataset_version v1
"""

import argparse
import yaml
import os
import json
import pandas as pd
from src.inference import NLLBPredictor
from src.dataset import AwajunDataLoader
from src.utils import setup_logging, format_time
import time
from sacrebleu.metrics import CHRF, BLEU

def load_config(config_path="config.yaml"):
    """Cargar configuración desde YAML"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def parse_args():
    """Argumentos de línea de comandos"""
    parser = argparse.ArgumentParser(description="Comparación de modelos NLLB")
    
    # Modelos a comparar
    parser.add_argument('--models', type=str, nargs='+', required=True,
                       help='Rutas a los modelos a comparar')
    parser.add_argument('--model_names', type=str, nargs='+', default=None,
                       help='Nombres personalizados para los modelos')
    
    # Configuración
    parser.add_argument('--direction', type=str, required=True, 
                       choices=['es2agr', 'agr2es'], 
                       help='Dirección de traducción')
    parser.add_argument('--dataset_version', type=str, default='v1',
                       choices=['v1', 'v2'],
                       help='Versión del dataset')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Archivo de configuración')
    
    # Evaluación
    parser.add_argument('--sample_size', type=int, default=None,
                       help='Número de muestras (None = todo el dev set)')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Tamaño de batch')
    parser.add_argument('--output_dir', type=str, default='model_comparison',
                       help='Directorio de salida')
    
    # Opciones adicionales
    parser.add_argument('--save_translations', action='store_true',
                       help='Guardar todas las traducciones')
    parser.add_argument('--head_to_head', action='store_true',
                       help='Comparación cabeza a cabeza ejemplo por ejemplo')
    
    return parser.parse_args()

class ModelComparator:
    """Comparador de múltiples modelos"""
    
    def __init__(self, model_paths, direction, dataset_version, config, output_dir, model_names=None):
        self.model_paths = model_paths
        self.direction = direction
        self.dataset_version = dataset_version
        self.config = config
        self.output_dir = output_dir
        self.model_names = model_names or [f"Model_{i+1}" for i in range(len(model_paths))]
        
        # Crear directorio de salida
        os.makedirs(output_dir, exist_ok=True)
        
        # Configurar idiomas
        if direction == 'es2agr':
            self.src_lang, self.tgt_lang = 'es', 'agr'
        else:
            self.src_lang, self.tgt_lang = 'agr', 'es'
        
        # Cargar datos
        self.config['data']['dataset_version'] = dataset_version
        self.data_loader = AwajunDataLoader(self.config)
        
        # Cargar predictores
        self.predictors = {}
        print("Cargando modelos...")
        for i, model_path in enumerate(model_paths):
            model_name = self.model_names[i]
            print(f"  {model_name}: {model_path}")
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
            
            self.predictors[model_name] = NLLBPredictor(
                model_path=model_path,
                direction=direction,
                config=config
            )
    
    def load_evaluation_data(self, sample_size=None):
        """Cargar datos para evaluación"""
        df_train, df_eval = self.data_loader.load_data()
        
        if sample_size and sample_size < len(df_eval):
            df_eval = df_eval.sample(sample_size, random_state=42)
            print(f"Usando muestra de {sample_size} ejemplos")
        
        return df_eval
    
    def evaluate_all_models(self, df_eval, batch_size=16):
        """Evaluar todos los modelos"""
        sources = df_eval[self.src_lang].tolist()
        references = df_eval[self.tgt_lang].tolist()
        
        results = {}
        all_predictions = {}
        
        chrf_metric = CHRF(word_order=2)
        bleu_metric = BLEU()
        
        for model_name, predictor in self.predictors.items():
            print(f"\nEvaluando {model_name}...")
            start_time = time.time()
            
            # Generar predicciones
            predictions = predictor.translate_batch(
                sources, 
                batch_size=batch_size, 
                show_progress=True
            )
            
            all_predictions[model_name] = predictions
            
            # Calcular métricas
            chrf_score = chrf_metric.corpus_score(predictions, [references]).score
            bleu_score = bleu_metric.corpus_score(predictions, [references]).score
            
            elapsed = time.time() - start_time
            
            # Métricas adicionales
            exact_matches = sum(1 for p, r in zip(predictions, references) 
                              if p.strip().lower() == r.strip().lower())
            
            results[model_name] = {
                'chrf': chrf_score,
                'bleu': bleu_score,
                'exact_matches': exact_matches,
                'exact_match_rate': exact_matches / len(df_eval) * 100,
                'avg_pred_length': sum(len(p) for p in predictions) / len(predictions),
                'evaluation_time': elapsed,
                'samples_per_second': len(df_eval) / elapsed,
                'model_info': predictor.get_model_info()
            }
        
        return results, all_predictions, sources, references
    
    def analyze_head_to_head(self, all_predictions, sources, references):
        """Análisis cabeza a cabeza"""
        model_names = list(all_predictions.keys())
        comparisons = {}
        
        chrf_metric = CHRF(word_order=2)
        
        # Comparar cada par de modelos
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names):
                if i >= j:  # Evitar duplicados y auto-comparación
                    continue
                
                pair_key = f"{model1}_vs_{model2}"
                
                # Contar victorias por ejemplo
                model1_wins = 0
                model2_wins = 0
                ties = 0
                
                pred1 = all_predictions[model1]
                pred2 = all_predictions[model2]
                
                for p1, p2, ref in zip(pred1, pred2, references):
                    # CHRF por ejemplo individual
                    score1 = chrf_metric.sentence_score(p1, [ref]).score
                    score2 = chrf_metric.sentence_score(p2, [ref]).score
                    
                    if score1 > score2:
                        model1_wins += 1
                    elif score2 > score1:
                        model2_wins += 1
                    else:
                        ties += 1
                
                comparisons[pair_key] = {
                    'model1': model1,
                    'model2': model2,
                    'model1_wins': model1_wins,
                    'model2_wins': model2_wins,
                    'ties': ties,
                    'total': len(references),
                    'model1_win_rate': model1_wins / len(references) * 100,
                    'model2_win_rate': model2_wins / len(references) * 100
                }
        
        return comparisons
    
    def create_comparison_table(self, results):
        """Crear tabla de comparación"""
        df = pd.DataFrame(results).T
        
        # Ordenar por CHRF
        df = df.sort_values('chrf', ascending=False)
        
        # Formatear columnas
        df['chrf'] = df['chrf'].round(2)
        df['bleu'] = df['bleu'].round(2)
        df['exact_match_rate'] = df['exact_match_rate'].round(1)
        df['avg_pred_length'] = df['avg_pred_length'].round(1)
        df['samples_per_second'] = df['samples_per_second'].round(1)
        
        return df
    
    def save_results(self, results, comparisons, all_predictions, sources, references, save_translations):
        """Guardar resultados de comparación"""
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        
        # Resultados principales
        results_file = os.path.join(self.output_dir, f'comparison_results_{timestamp}.json')
        
        full_results = {
            'comparison_config': {
                'models': list(zip(self.model_names, self.model_paths)),
                'direction': self.direction,
                'dataset_version': self.dataset_version,
                'eval_samples': len(sources),
                'timestamp': timestamp
            },
            'model_results': results,
            'head_to_head': comparisons
        }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(full_results, f, indent=2, ensure_ascii=False)
        
        # Tabla de comparación
        comparison_table = self.create_comparison_table(results)
        table_file = os.path.join(self.output_dir, f'comparison_table_{timestamp}.csv')
        comparison_table.to_csv(table_file)
        
        # Resumen en texto
        summary_file = os.path.join(self.output_dir, f'comparison_summary_{timestamp}.txt')
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"Comparación de Modelos NLLB\n")
            f.write(f"===========================\n\n")
            f.write(f"Configuración:\n")
            f.write(f"  Dirección: {self.direction}\n")
            f.write(f"  Dataset: {self.dataset_version}\n")
            f.write(f"  Muestras evaluadas: {len(sources)}\n")
            f.write(f"  Modelos comparados: {len(self.model_names)}\n\n")
            
            f.write(f"Ranking por CHRF++:\n")
            f.write(f"==================\n")
            for i, (model_name, row) in enumerate(comparison_table.iterrows(), 1):
                f.write(f"{i}. {model_name}:\n")
                f.write(f"   CHRF++: {row['chrf']:.2f}\n")
                f.write(f"   BLEU: {row['bleu']:.2f}\n")
                f.write(f"   Exact Match: {row['exact_match_rate']:.1f}%\n")
                f.write(f"   Velocidad: {row['samples_per_second']:.1f} samples/sec\n\n")
            
            if comparisons:
                f.write(f"Comparación Cabeza a Cabeza:\n")
                f.write(f"===========================\n")
                for pair_key, comp in comparisons.items():
                    f.write(f"{comp['model1']} vs {comp['model2']}:\n")
                    f.write(f"  {comp['model1']}: {comp['model1_wins']} victorias ({comp['model1_win_rate']:.1f}%)\n")
                    f.write(f"  {comp['model2']}: {comp['model2_wins']} victorias ({comp['model2_win_rate']:.1f}%)\n")
                    f.write(f"  Empates: {comp['ties']}\n\n")
        
        # Guardar traducciones si se solicita
        if save_translations:
            translations_file = os.path.join(self.output_dir, f'all_translations_{timestamp}.csv')
            
            # Crear DataFrame con todas las traducciones
            data = {
                'source': sources,
                'reference': references
            }
            
            for model_name, predictions in all_predictions.items():
                data[f'prediction_{model_name}'] = predictions
            
            translations_df = pd.DataFrame(data)
            translations_df.to_csv(translations_file, index=False)
            
            return results_file, summary_file, table_file, translations_file
        
        return results_file, summary_file, table_file, None
    
    def find_interesting_examples(self, all_predictions, sources, references, n_examples=10):
        """Encontrar ejemplos donde los modelos difieren más"""
        model_names = list(all_predictions.keys())
        chrf_metric = CHRF(word_order=2)
        
        example_scores = []
        
        for i, (source, reference) in enumerate(zip(sources, references)):
            # Calcular CHRF para cada modelo en este ejemplo
            scores = {}
            for model_name in model_names:
                pred = all_predictions[model_name][i]
                score = chrf_metric.sentence_score(pred, [reference]).score
                scores[model_name] = score
            
            # Calcular varianza en los scores
            score_values = list(scores.values())
            variance = sum((s - sum(score_values)/len(score_values))**2 for s in score_values) / len(score_values)
            
            example_scores.append({
                'index': i,
                'source': source,
                'reference': reference,
                'scores': scores,
                'predictions': {name: all_predictions[name][i] for name in model_names},
                'variance': variance,
                'max_score': max(score_values),
                'min_score': min(score_values),
                'score_range': max(score_values) - min(score_values)
            })
        
        # Ordenar por varianza (casos más interesantes)
        example_scores.sort(key=lambda x: x['variance'], reverse=True)
        
        return example_scores[:n_examples]

def main():
    """Función principal"""
    args = parse_args()
    
    # Setup logging
    setup_logging()
    
    # Verificar que todos los modelos existen
    for model_path in args.models:
        if not os.path.exists(model_path):
            print(f"Error: Modelo no encontrado en {model_path}")
            return
    
    # Cargar configuración
    config = load_config(args.config)
    
    # Crear comparador
    print(f"Iniciando comparación de {len(args.models)} modelos")
    print(f"Dirección: {args.direction}")
    print(f"Dataset: {args.dataset_version}")
    
    comparator = ModelComparator(
        model_paths=args.models,
        direction=args.direction,
        dataset_version=args.dataset_version,
        config=config,
        output_dir=args.output_dir,
        model_names=args.model_names
    )
    
    # Cargar datos
    df_eval = comparator.load_evaluation_data(sample_size=args.sample_size)
    print(f"Datos cargados: {len(df_eval)} ejemplos")
    
    # Evaluar todos los modelos
    results, all_predictions, sources, references = comparator.evaluate_all_models(
        df_eval, batch_size=args.batch_size
    )
    
    # Análisis cabeza a cabeza si se solicita
    comparisons = {}
    if args.head_to_head:
        print("\nRealizando análisis cabeza a cabeza...")
        comparisons = comparator.analyze_head_to_head(all_predictions, sources, references)
    
    # Guardar resultados
    saved_files = comparator.save_results(
        results, comparisons, all_predictions, sources, references, args.save_translations
    )
    
    # Mostrar resultados
    print(f"\nResultados de Comparación:")
    print(f"=========================")
    
    # Crear y mostrar tabla
    comparison_table = comparator.create_comparison_table(results)
    print(comparison_table[['chrf', 'bleu', 'exact_match_rate', 'samples_per_second']])
    
    # Mejor modelo
    best_model = comparison_table.index[0]
    best_chrf = comparison_table.loc[best_model, 'chrf']
    print(f"\n🏆 Mejor modelo: {best_model} (CHRF++ {best_chrf:.2f})")
    
    if comparisons:
        print(f"\nComparación Cabeza a Cabeza:")
        for pair_key, comp in comparisons.items():
            print(f"  {comp['model1']} vs {comp['model2']}: "
                  f"{comp['model1_wins']}-{comp['model2_wins']}-{comp['ties']} "
                  f"({comp['model1_win_rate']:.1f}% vs {comp['model2_win_rate']:.1f}%)")
    
    # Ejemplos interesantes
    interesting = comparator.find_interesting_examples(all_predictions, sources, references, 3)
    print(f"\nEjemplos donde los modelos más difieren:")
    for i, example in enumerate(interesting, 1):
        print(f"  {i}. {example['source'][:50]}...")
        for model_name, pred in example['predictions'].items():
            score = example['scores'][model_name]
            print(f"     {model_name}: {pred[:50]}... (CHRF: {score:.1f})")
    
    print(f"\nArchivos generados:")
    for i, file_path in enumerate(saved_files):
        if file_path:
            print(f"  {i+1}. {file_path}")

if __name__ == "__main__":
    main()