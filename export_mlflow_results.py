#!/usr/bin/env python3
"""
Script para exportar resultados de experimentos MLflow
Genera tabla CSV con todos los experimentos para la tesis
"""

import mlflow
import pandas as pd
from datetime import datetime

def export_mlflow_experiments(experiment_names, output_file="mlflow_results.csv"):
    """
    Exportar resultados de experimentos MLflow a CSV
    
    Args:
        experiment_names: Lista de nombres de experimentos (ej: ['awajun_translation_agr2es', 'awajun_translation_es2agr'])
        output_file: Nombre del archivo CSV de salida
    """
    
    # Configurar MLflow
    mlflow.set_tracking_uri("./mlruns")
    
    all_runs = []
    
    for exp_name in experiment_names:
        print(f"📂 Procesando experimento: {exp_name}")
        
        try:
            # Obtener experimento
            experiment = mlflow.get_experiment_by_name(exp_name)
            if experiment is None:
                print(f"   ⚠️ No se encontró el experimento: {exp_name}")
                continue
            
            # Buscar todas las runs del experimento
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string="",
                order_by=["start_time DESC"]
            )
            
            print(f"   ✅ Encontradas {len(runs)} runs")
            
            # Procesar cada run
            for idx, run in runs.iterrows():
                try:
                    run_data = {
                        # Identificación
                        'run_id': run['run_id'],
                        'experiment_name': exp_name,
                        'run_name': run.get('tags.mlflow.runName', 'N/A'),
                        'status': run['status'],
                        'start_time': run['start_time'],
                        
                        # Configuración (tags)
                        'direction': run.get('tags.direction', 'N/A'),
                        'dataset_version': run.get('tags.dataset_version', 'N/A'),
                        'model': run.get('tags.model', 'N/A'),
                        'balance_method': run.get('tags.balance_method', 'N/A'),
                        
                        # Parámetros
                        'learning_rate': run.get('params.learning_rate', 'N/A'),
                        'batch_size': run.get('params.batch_size', 'N/A'),
                        'epochs': run.get('params.epochs', 'N/A'),
                        'patience': run.get('params.patience', 'N/A'),
                        'eval_frequency': run.get('params.eval_frequency', 'N/A'),
                        
                        # Métricas principales
                        'best_chrf': run.get('metrics.best_chrf', None),
                        'best_epoch': run.get('metrics.best_epoch', None),
                        'eval_chrf': run.get('metrics.eval_chrf', None),
                        'eval_bleu': run.get('metrics.eval_bleu', None),
                        
                        # Información adicional
                        'total_training_time_minutes': run.get('metrics.total_training_time_minutes', None),
                        'early_stopped': run.get('metrics.early_stopped', None),
                        'final_epoch': run.get('metrics.final_epoch', None),
                    }
                    
                    all_runs.append(run_data)
                    
                except Exception as e:
                    print(f"   ⚠️ Error procesando run {idx}: {e}")
                    continue
        
        except Exception as e:
            print(f"   ❌ Error con experimento {exp_name}: {e}")
            continue
    
    # Crear DataFrame
    df = pd.DataFrame(all_runs)
    
    # Mostrar todas las runs antes de filtrar
    print(f"\n📊 Total de runs encontradas: {len(df)}")
    if not df.empty:
        status_counts = df['status'].value_counts()
        print("\nEstatus de runs:")
        for status, count in status_counts.items():
            print(f"   {status}: {count}")
    
    # Filtrar runs FINISHED y RUNNING (excluir solo FAILED, KILLED, etc.)
    df_valid = df[df['status'].isin(['FINISHED', 'RUNNING'])].copy()
    
    if len(df_valid) < len(df):
        print(f"\n⚠️  Excluyendo runs fallidas/canceladas: {len(df)} → {len(df_valid)}")
        excluded = df[~df['status'].isin(['FINISHED', 'RUNNING'])][['run_name', 'status', 'dataset_version']]
        if not excluded.empty:
            print("    Runs excluidas:")
            for _, row in excluded.iterrows():
                print(f"      - {row['run_name']} ({row['status']}) - {row['dataset_version']}")
    
    # Marcar runs en curso
    if 'RUNNING' in df_valid['status'].values:
        running_count = (df_valid['status'] == 'RUNNING').sum()
        print(f"\n🔄 Runs en curso incluidas: {running_count}")
    
    df = df_valid
    
    # Ordenar por dataset y chrF++
    df = df.sort_values(['dataset_version', 'best_chrf'], ascending=[True, False])
    
    # Guardar a CSV
    df.to_csv(output_file, index=False)
    
    print(f"\n✅ Resultados exportados a: {output_file}")
    print(f"📊 Total de runs exitosas: {len(df)}")
    
    # Mostrar resumen
    print("\n" + "="*80)
    print("RESUMEN DE EXPERIMENTOS")
    print("="*80)
    
    # Agrupar por dataset
    if not df.empty:
        summary = df.groupby('dataset_version').agg({
            'best_chrf': ['count', 'mean', 'max'],
            'eval_bleu': 'max'
        }).round(2)
        
        print("\nPor dataset:")
        print(summary)
        
        # Mejor run overall
        best_run = df.loc[df['best_chrf'].idxmax()]
        print(f"\n🏆 MEJOR RESULTADO OVERALL:")
        print(f"   Dataset: {best_run['dataset_version']}")
        print(f"   Modelo: {best_run['model']}")
        print(f"   Dirección: {best_run['direction']}")
        print(f"   Balance: {best_run['balance_method']}")
        print(f"   chrF++: {best_run['best_chrf']:.2f}")
        print(f"   BLEU: {best_run['eval_bleu']:.2f}")
        print(f"   Mejor época: {best_run['best_epoch']:.0f}")
        
    return df

def create_thesis_tables(df, output_prefix="thesis_table"):
    """
    Crear tablas específicas para la tesis
    """
    
    # Tabla 1: Comparación por dataset (para RE4 vs RE5)
    table1 = df.groupby(['dataset_version', 'direction', 'model', 'balance_method']).agg({
        'best_chrf': 'max',
        'eval_bleu': 'max',
        'best_epoch': 'first',
        'total_training_time_minutes': 'first'
    }).reset_index()
    
    table1 = table1.round(2)
    table1.to_csv(f"{output_prefix}_by_dataset.csv", index=False)
    print(f"\n✅ Tabla por dataset guardada: {output_prefix}_by_dataset.csv")
    
    # Tabla 2: Solo V1 (baseline - RE4)
    table_v1 = df[df['dataset_version'] == 'v1'][
        ['model', 'direction', 'balance_method', 'best_chrf', 'eval_bleu', 
         'best_epoch', 'total_training_time_minutes']
    ].round(2)
    table_v1.to_csv(f"{output_prefix}_v1_baseline.csv", index=False)
    print(f"✅ Tabla V1 (baseline) guardada: {output_prefix}_v1_baseline.csv")
    
    # Tabla 3: V3 variantes (RE5)
    table_v3 = df[df['dataset_version'].str.startswith('v3')][
        ['dataset_version', 'model', 'direction', 'balance_method', 
         'best_chrf', 'eval_bleu', 'best_epoch', 'total_training_time_minutes']
    ].round(2)
    table_v3.to_csv(f"{output_prefix}_v3_synthetic.csv", index=False)
    print(f"✅ Tabla V3 (sintéticos) guardada: {output_prefix}_v3_synthetic.csv")
    
    # Tabla 4: Comparación V1 vs mejor V3 por dirección
    comparison = []
    for direction in df['direction'].unique():
        df_dir = df[df['direction'] == direction]
        
        # Mejor V1
        v1_data = df_dir[df_dir['dataset_version'] == 'v1']
        if not v1_data.empty:
            v1_best = v1_data['best_chrf'].max()
        else:
            v1_best = None
        
        # Mejor V3
        v3_data = df_dir[df_dir['dataset_version'].str.startswith('v3')]
        if not v3_data.empty:
            v3_best = v3_data['best_chrf'].max()
            v3_dataset = v3_data.loc[v3_data['best_chrf'].idxmax()]['dataset_version']
        else:
            v3_best = None
            v3_dataset = 'N/A'
        
        # Calcular mejora
        if v1_best is not None and v3_best is not None:
            improvement = v3_best - v1_best
            improvement_pct = (improvement / v1_best) * 100 if v1_best > 0 else 0
        else:
            improvement = None
            improvement_pct = None
        
        comparison.append({
            'direction': direction,
            'v1_best_chrf': v1_best,
            'v3_best_chrf': v3_best,
            'v3_best_dataset': v3_dataset,
            'improvement': improvement,
            'improvement_pct': improvement_pct
        })
    
    table_comparison = pd.DataFrame(comparison).round(2)
    table_comparison.to_csv(f"{output_prefix}_v1_vs_v3_comparison.csv", index=False)
    print(f"✅ Tabla comparativa V1 vs V3: {output_prefix}_v1_vs_v3_comparison.csv")

if __name__ == "__main__":
    # Experimentos a exportar
    experiments = [
        "awajun_translation_agr2es",
        "awajun_translation_es2agr"
    ]
    
    print("="*80)
    print("EXPORTADOR DE RESULTADOS MLFLOW PARA TESIS")
    print("="*80)
    print()
    
    # Exportar datos
    df = export_mlflow_experiments(experiments, "mlflow_all_results.csv")
    
    # Crear tablas para la tesis
    if not df.empty:
        create_thesis_tables(df, "thesis_table")
        
        print("\n" + "="*80)
        print("✅ EXPORTACIÓN COMPLETADA")
        print("="*80)
        print("\nArchivos generados:")
        print("  1. mlflow_all_results.csv - Todos los experimentos")
        print("  2. thesis_table_by_dataset.csv - Resumen por dataset")
        print("  3. thesis_table_v1_baseline.csv - Solo baseline V1")
        print("  4. thesis_table_v3_synthetic.csv - Solo experimentos V3")
        print("  5. thesis_table_v1_vs_v3_comparison.csv - Comparación V1 vs V3")
    else:
        print("\n⚠️ No se encontraron experimentos para exportar")