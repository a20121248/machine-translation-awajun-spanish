#!/bin/bash

# Script para ejecutar experimentos NLLB Awajún-Español
# Solo 2 experimentos: es2agr y agr2es
# Dataset version como parámetro dentro de cada run

echo "🚀 Ejecutando experimentos NLLB Awajún-Español"
echo "==============================================="

# ===== PRUEBA RÁPIDA =====
echo ""
echo "🧪 PRUEBA RÁPIDA - Verificación de funcionamiento"
echo "---------------------------------------------------"

echo "1️⃣  Probando configuración básica..."
python3 train.py --direction es2agr --dataset_version v1 --test_mode --batch_size 16

echo ""
echo "✅ Prueba básica completada"
echo ""

# ===== EXPERIMENTOS ESPAÑOL → AWAJÚN =====
echo "🔬 EXPERIMENTO: ESPAÑOL → AWAJÚN"
echo "================================"

# NLLB-600M experiments
echo "1️⃣  NLLB-600M | Dataset V1 | Sin balance"
python3 train.py --direction es2agr --dataset_version v1 --model_name facebook/nllb-200-distilled-600M --epochs 12 --batch_size 24 --learning_rate 3e-5 --patience 2 --config <(sed 's/balance_method: "weighted"/balance_method: "none"/' config.yaml)

echo "2️⃣  NLLB-600M | Dataset V1 | Con balance"
python3 train.py --direction es2agr --dataset_version v1 --model_name facebook/nllb-200-distilled-600M --epochs 12 --batch_size 24 --learning_rate 3e-5 --patience 2

echo "3️⃣  NLLB-600M | Dataset V2 | Sin balance"
python3 train.py --direction es2agr --dataset_version v2 --model_name facebook/nllb-200-distilled-600M --epochs 15 --batch_size 24 --learning_rate 3e-5 --patience 2 --config <(sed 's/balance_method: "weighted"/balance_method: "none"/' config.yaml)

echo "4️⃣  NLLB-600M | Dataset V2 | Con balance"
python3 train.py --direction es2agr --dataset_version v2 --model_name facebook/nllb-200-distilled-600M --epochs 15 --batch_size 24 --learning_rate 3e-5 --patience 2

# NLLB-1.3B experiments
echo "5️⃣  NLLB-1.3B | Dataset V1 | Sin balance"
python3 train.py --direction es2agr --dataset_version v1 --model_name facebook/nllb-200-distilled-1.3B --epochs 10 --batch_size 16 --learning_rate 2e-5 --patience 2 --config <(sed 's/balance_method: "weighted"/balance_method: "none"/' config.yaml)

echo "6️⃣  NLLB-1.3B | Dataset V1 | Con balance"
python3 train.py --direction es2agr --dataset_version v1 --model_name facebook/nllb-200-distilled-1.3B --epochs 10 --batch_size 16 --learning_rate 2e-5 --patience 2

echo "7️⃣  NLLB-1.3B | Dataset V2 | Sin balance"
python3 train.py --direction es2agr --dataset_version v2 --model_name facebook/nllb-200-distilled-1.3B --epochs 12 --batch_size 16 --learning_rate 2e-5 --patience 2 --config <(sed 's/balance_method: "weighted"/balance_method: "none"/' config.yaml)

echo "8️⃣  NLLB-1.3B | Dataset V2 | Con balance"
python3 train.py --direction es2agr --dataset_version v2 --model_name facebook/nllb-200-distilled-1.3B --epochs 12 --batch_size 16 --learning_rate 2e-5 --patience 2

# ===== EXPERIMENTOS AWAJÚN → ESPAÑOL =====
echo ""
echo "🔬 EXPERIMENTO: AWAJÚN → ESPAÑOL"
echo "================================"

# NLLB-600M experiments
echo "9️⃣  NLLB-600M | Dataset V1 | Sin balance"
python3 train.py --direction agr2es --dataset_version v1 --model_name facebook/nllb-200-distilled-600M --epochs 12 --batch_size 24 --learning_rate 3e-5 --patience 2 --config <(sed 's/balance_method: "weighted"/balance_method: "none"/' config.yaml)

echo "🔟 NLLB-600M | Dataset V1 | Con balance"
python3 train.py --direction agr2es --dataset_version v1 --model_name facebook/nllb-200-distilled-600M --epochs 12 --batch_size 24 --learning_rate 3e-5 --patience 2

echo "1️⃣1️⃣ NLLB-600M | Dataset V2 | Sin balance"
python3 train.py --direction agr2es --dataset_version v2 --model_name facebook/nllb-200-distilled-600M --epochs 15 --batch_size 24 --learning_rate 3e-5 --patience 2 --config <(sed 's/balance_method: "weighted"/balance_method: "none"/' config.yaml)

echo "1️⃣2️⃣ NLLB-600M | Dataset V2 | Con balance"
python3 train.py --direction agr2es --dataset_version v2 --model_name facebook/nllb-200-distilled-600M --epochs 15 --batch_size 24 --learning_rate 3e-5 --patience 2

# NLLB-1.3B experiments
echo "1️⃣3️⃣ NLLB-1.3B | Dataset V1 | Sin balance"
python3 train.py --direction agr2es --dataset_version v1 --model_name facebook/nllb-200-distilled-1.3B --epochs 10 --batch_size 16 --learning_rate 2e-5 --patience 2 --config <(sed 's/balance_method: "weighted"/balance_method: "none"/' config.yaml)

echo "1️⃣4️⃣ NLLB-1.3B | Dataset V1 | Con balance"
python3 train.py --direction agr2es --dataset_version v1 --model_name facebook/nllb-200-distilled-1.3B --epochs 10 --batch_size 16 --learning_rate 2e-5 --patience 2

echo "1️⃣5️⃣ NLLB-1.3B | Dataset V2 | Sin balance"
python3 train.py --direction agr2es --dataset_version v2 --model_name facebook/nllb-200-distilled-1.3B --epochs 12 --batch_size 16 --learning_rate 2e-5 --patience 2 --config <(sed 's/balance_method: "weighted"/balance_method: "none"/' config.yaml)

echo "1️⃣6️⃣ NLLB-1.3B | Dataset V2 | Con balance"
python3 train.py --direction agr2es --dataset_version v2 --model_name facebook/nllb-200-distilled-1.3B --epochs 12 --batch_size 16 --learning_rate 2e-5 --patience 2

echo ""
echo "🎉 Todos los experimentos completados!"
echo "📊 Revisa los resultados en MLflow:"
echo "   - Experimento: awajun_translation_es2agr"
echo "   - Experimento: awajun_translation_agr2es"
echo "🔗 mlflow ui → http://localhost:5000"