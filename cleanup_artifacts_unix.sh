#!/bin/bash

# Script para limpiar carpetas artifacts excepto corridas específicas
MLRUNS_DIR="/home/jmonzon/machine-translation-awajun-spanish_backup/mlruns"

# IDs de las corridas que queremos mantener
KEEP_RUNS=(
    "a61e1d9c451d4bb68cddfcc66a4d75f2"
    "32842abde26f4ec09d806c977819175d"
    "5526c343f7f9470aac421755c181b1d2"
    "a078ccfc24284d0fbe45605fff9cdb74"
)

# Función para verificar si un run_id debe mantenerse
should_keep() {
    local run_id="$1"
    for keep_id in "${KEEP_RUNS[@]}"; do
        if [[ "$run_id" == "$keep_id" ]]; then
            return 0
        fi
    done
    return 1
}

deleted_count=0
kept_count=0

echo "Iniciando limpieza de carpetas artifacts..."
echo "Directorio base: $MLRUNS_DIR"
echo ""

for experiment_dir in "$MLRUNS_DIR"/*; do
    if [[ -d "$experiment_dir" ]]; then
        experiment_id=$(basename "$experiment_dir")
        echo "Procesando experimento: $experiment_id"
        
        for run_dir in "$experiment_dir"/*; do
            if [[ -d "$run_dir" ]]; then
                run_id=$(basename "$run_dir")
                artifacts_dir="$run_dir/artifacts"
                
                if [[ -d "$artifacts_dir" ]]; then
                    if should_keep "$run_id"; then
                        echo "  Manteniendo: $run_id"
                        ((kept_count++))
                    else
                        echo "  Borrando artifacts de: $run_id"
                        rm -rf "$artifacts_dir"/*
                        ((deleted_count++))
                    fi
                fi
            fi
        done
    fi
done

echo ""
echo "Limpieza completada!"
echo "Corridas mantenidas: $kept_count"
echo "Carpetas artifacts limpiadas: $deleted_count"
