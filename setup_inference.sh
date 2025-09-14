#!/bin/bash

# Script de configuración inicial para el sistema de inferencia NLLB
# Uso: bash setup_inference.sh

echo "🔮 Configurando Sistema de Inferencia NLLB Awajún-Español"
echo "=========================================================="

# Verificar Python
echo "📋 Verificando requisitos..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 no encontrado. Instala Python 3.8+"
    exit 1
fi

PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "✅ Python $PYTHON_VERSION detectado"

# Crear directorio de modelos si no existe
echo "📁 Configurando directorios..."
mkdir -p runs
mkdir -p evaluation_results
mkdir -p model_comparison
mkdir -p inference_outputs
echo "✅ Directorios creados"

# Verificar estructura de archivos
echo "🔍 Verificando estructura de archivos..."

required_files=(
    "predict.py"
    "evaluate.py" 
    "compare_models.py"
    "model_utils.py"
    "src/inference.py"
    "inference_config.yaml"
)

missing_files=()
for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        missing_files+=("$file")
    fi
done

if [ ${#missing_files[@]} -ne 0 ]; then
    echo "❌ Archivos faltantes:"
    printf '   %s\n' "${missing_files[@]}"
    exit 1
fi

echo "✅ Estructura de archivos verificada"

# Verificar dependencias de Python
echo "📦 Verificando dependencias..."

check_package() {
    python3 -c "import $1" 2>/dev/null
    return $?
}

required_packages=(
    "torch"
    "transformers" 
    "sacrebleu"
    "pandas"
    "tqdm"
)

missing_packages=()
for package in "${required_packages[@]}"; do
    if ! check_package "$package"; then
        missing_packages+=("$package")
    fi
done

if [ ${#missing_packages[@]} -ne 0 ]; then
    echo "❌ Paquetes Python faltantes:"
    printf '   %s\n' "${missing_packages[@]}"
    echo ""
    echo "Instala con: pip install ${missing_packages[*]}"
    exit 1
fi

echo "✅ Dependencias verificadas"

# Verificar GPU (opcional)
echo "🖥️ Verificando GPU..."
if python3 -c "import torch; print('CUDA disponible:', torch.cuda.is_available())" | grep -q "True"; then
    GPU_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count())")
    GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')")
    echo "✅ GPU detectada: $GPU_NAME (Total: $GPU_COUNT)"
else
    echo "ℹ️ GPU no disponible, usando CPU"
fi

# Verificar modelos existentes
echo "🤖 Verificando modelos disponibles..."
model_count=0
if [ -d "runs" ]; then
    model_count=$(find runs -name "config.json" | wc -l)
fi

if [ $model_count -gt 0 ]; then
    echo "✅ $model_count modelos encontrados en runs/"
    echo "   Usar: python model_utils.py --command list_models"
else
    echo "ℹ️ No hay modelos entrenados aún"
    echo "   Primero entrena modelos con train.py"
fi

# Crear archivos de ejemplo
echo "📝 Creando archivos de ejemplo..."

cat > test_es.txt << EOF
Hola mundo
Buenos días
¿Cómo estás?
Gracias por tu ayuda
El día está muy bonito
EOF

cat > test_agr.txt << EOF
Yama
Ame
Wararat ame
Anentaim
Tsawan pénker
EOF

echo "✅ Archivos de prueba creados (test_es.txt, test_agr.txt)"

# Crear script de prueba rápida
cat > quick_test.sh << 'EOF'
#!/bin/bash
echo "🧪 Prueba rápida del sistema de inferencia"

# Verificar que hay modelos
if ! ls runs/*/best_model/config.json &> /dev/null; then
    echo "❌ No hay modelos entrenados. Ejecuta train.py primero."
    exit 1
fi

# Encontrar primer modelo disponible
MODEL=$(find runs -name "best_model" -type d | head -n 1)
echo "📋 Usando modelo: $MODEL"

# Probar traducción simple
echo "🔄 Probando traducción..."
python3 predict.py \
    --model_path "$MODEL" \
    --direction es2agr \
    --input "Hola mundo" \
    --verbose

echo "✅ Prueba completada"
EOF

chmod +x quick_test.sh
echo "✅ Script de prueba rápida creado (quick_test.sh)"

# Verificar configuración
echo "⚙️ Verificando configuración..."
if [ -f "inference_config.yaml" ]; then
    echo "✅ Configuración de inferencia encontrada"
else
    echo "❌ inference_config.yaml no encontrado"
fi

# Resumen final
echo ""
echo "🎉 Configuración completada!"
echo "================================"
echo ""
echo "📋 Comandos disponibles:"
echo "  python predict.py --help           # Ver opciones de traducción"
echo "  python evaluate.py --help          # Ver opciones de evaluación"  
echo "  python compare_models.py --help    # Ver opciones de comparación"
echo "  python model_utils.py --help       # Ver utilidades de modelos"
echo ""
echo "🚀 Comandos de ejemplo:"
echo "  python model_utils.py --command list_models"
echo "  bash quick_test.sh"
echo ""
echo "📚 Documentación completa: README_INFERENCE.md"
echo ""

# Verificar si podemos hacer una prueba básica
if ls runs/*/best_model/config.json &> /dev/null; then
    echo "¿Ejecutar prueba rápida ahora? (y/n)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        bash quick_test.sh
    fi
else
    echo "ℹ️  Para usar el sistema, primero entrena algunos modelos:"
    echo "   python train.py --direction es2agr --dataset_version v1 --test_mode"
fi

echo ""
echo "✅ Sistema de inferencia listo para usar!"