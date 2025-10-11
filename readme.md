cd /home/jmonzon/machine-translation-awajun-spanish_backup/
source /home/jmonzon/envs/nllb-2/bin/activate


# 🚀 Comandos para Experimentos NLLB

## 📁 Estructura de directorios requerida
```
proyecto/
├── config.yaml
├── train.py
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── model.py
│   ├── evaluation.py
│   ├── training.py
│   └── utils.py
└── data/
    ├── awajun-spanish-v1/     # Dataset normal
    │   ├── train.agr
    │   ├── train.es
    │   ├── train.source
    │   ├── dev.agr
    │   ├── dev.es
    │   └── dev.source
    └── awajun-spanish-v2/     # Dataset extendido
        ├── train.agr
        ├── train.es
        ├── train.source
        ├── dev.agr
        ├── dev.es
        └── dev.source
```

## 🧪 Comandos de Prueba Rápida (1-2 minutos)

### Verificar que todo funciona
```bash
# Prueba básica es2agr
python train.py --direction es2agr --dataset_version v1 --test_mode

# Prueba básica agr2es  
python train.py --direction agr2es --dataset_version v1 --test_mode

# Prueba con dataset extendido
python train.py --direction es2agr --dataset_version v2 --test_mode
```

## 🔬 Experimentos Reales

### Configuración Base (recomendada)
```bash
# Español → Awajún (dataset normal)
python train.py --direction es2agr --dataset_version v1 --epochs 12 --batch_size 16 --learning_rate 3e-5 --patience 3

# Awajún → Español (dataset normal)
python train.py --direction agr2es --dataset_version v1 --epochs 12 --batch_size 16 --learning_rate 3e-5 --patience 3
```

### Con Dataset Extendido
```bash
# Español → Awajún (más datos)
python train.py --direction es2agr --dataset_version v2 --epochs 15 --batch_size 16 --learning_rate 3e-5 --patience 4

# Awajún → Español (más datos)  
python train.py --direction agr2es --dataset_version v2 --epochs 15 --batch_size 16 --learning_rate 3e-5 --patience 4
```

## ⚙️ Experimentos con Diferentes Hiperparámetros

### Learning Rate Conservador
```bash
python train.py --direction es2agr --dataset_version v1 --epochs 15 --batch_size 16 --learning_rate 1e-5 --patience 5
```

### Batch Size Grande (si tienes GPU potente)
```bash
python train.py --direction es2agr --dataset_version v1 --epochs 10 --batch_size 32 --learning_rate 3e-5 --patience 3
```

### Entrenamiento Largo
```bash
python train.py --direction agr2es --dataset_version v2 --epochs 25 --batch_size 16 --learning_rate 3e-5 --patience 7
```

### GPU con Poca Memoria
```bash
python train.py --direction es2agr --dataset_version v1 --epochs 12 --batch_size 4 --learning_rate 2e-5 --patience 4
```

## 📊 Monitoreo con MLflow

### Iniciar dashboard MLflow
```bash
mlflow ui
```
Luego abre: http://localhost:5000

### Ver experimentos específicos
```bash
# Filtrar por dataset version
mlflow ui --backend-store-uri file:./mlruns

# En la interfaz web, filtrar por:
# - Experiment name contiene "v1" o "v2"  
# - Direction: "es2agr" o "agr2es"
```

## 🛠️ Comandos de Utilidad

### Verificar estructura de datos
```bash
ls -la data/awajun-spanish-v1/
ls -la data/awajun-spanish-v2/
```

### Ejecutar todos los experimentos automáticamente
```bash
chmod +x run_experiments.sh
./run_experiments.sh
```

### Reanudar entrenamiento desde checkpoint
```bash
python train.py --direction es2agr --dataset_version v1 --resume runs/nllb_es2agr_20250715_143022
```

## 🎯 Configuración Recomendada por Caso

### Para tesis/demostración
```bash
python train.py --direction es2agr --dataset_version v1 --epochs 10 --batch_size 16 --learning_rate 3e-5 --patience 3
```

### Para publicación/paper
```bash
python train.py --direction es2agr --dataset_version v2 --epochs 20 --batch_size 16 --learning_rate 3e-5 --patience 5
```

### Para experimento rápido
```bash
python train.py --direction es2agr --dataset_version v1 --test_mode
```

## 📈 Interpretación de Resultados

### Métricas a observar en MLflow:
- **eval_chrf**: Métrica principal (mayor = mejor)
- **eval_bleu**: Métrica secundaria (mayor = mejor)  
- **best_epoch**: Época del mejor modelo
- **early_stopped**: Si se activó early stopping
- **total_training_time_minutes**: Tiempo total

### Buenos resultados esperados:
- **CHRF++ > 45**: Traducción decente
- **CHRF++ > 55**: Traducción buena
- **CHRF++ > 65**: Traducción excelente