#!/bin/bash

# Script para limpiar carpetas artifacts excepto corridas específicas
# Mantener solo estas corridas:
# - a61e1d9c451d4bb68cddfcc66a4d75f2 (NLLB-1.3B | Sintético | P40 | Oversample)
# - 32842abde26f4ec09d806c977819175d (NLLB-1.3B | Sintético | P100 | Oversample)
# - 5526c343f7f9470aac421755c181b1d2 (NLLB-1.3B | Original | Ninguno)
# - a078ccfc24284d0fbe45605fff9cdb74 (1. NLLB-600M)

# Directorio base de MLflow
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

# Contador de operaciones
deleted_count=0
kept_count=0
total_size_deleted=0

echo "🧹 Iniciando limpieza de carpetas artifacts..."
echo "📁 Directorio base: $MLRUNS_DIR"
echo "🔒 Corridas a mantener: ${#KEEP_RUNS[@]}"
echo ""

# Buscar todos los experimentos
for experiment_dir in "$MLRUNS_DIR"/*; do
    if [[ -d "$experiment_dir" ]]; then
        experiment_id=$(basename "$experiment_dir")
        echo "🔍 Procesando experimento: $experiment_id"
        
        # Buscar todas las corridas en el experimento
        for run_dir in "$experiment_dir"/*; do
            if [[ -d "$run_dir" ]]; then
                run_id=$(basename "$run_dir")
                artifacts_dir="$run_dir/artifacts"
                
                # Verificar si existe la carpeta artifacts
                if [[ -d "$artifacts_dir" ]]; then
                    if should_keep "$run_id"; then
                        echo "  ✅ Manteniendo: $run_id"
                        ((kept_count++))
                    else
                        # Calcular tamaño antes de borrar
                        size_kb=$(du -sk "$artifacts_dir" 2>/dev/null | cut -f1)
                        if [[ -n "$size_kb" ]]; then
                            ((total_size_deleted += size_kb))
                        fi
                        
                        echo "  🗑️  Borrando artifacts de: $run_id"
                        rm -rf "$artifacts_dir"/*
                        ((deleted_count++))
                    fi
                else
                    echo "  ⚠️  No existe carpeta artifacts en: $run_id"
                fi
            fi
        done
        echo ""
    fi
done

# Convertir KB a MB para mejor legibilidad
total_size_mb=$((total_size_deleted / 1024))

echo "✨ Limpieza completada!"
echo "📊 Resumen:"
echo "  - Corridas mantenidas: $kept_count"
echo "  - Carpetas artifacts limpiadas: $deleted_count"
echo "  - Espacio liberado: ${total_size_mb} MB (~${total_size_deleted} KB)"
echo ""
echo "🔒 Las siguientes corridas conservan sus artifacts:"
for run_id in "${KEEP_RUNS[@]}"; do
    echo "  - $run_id"
done