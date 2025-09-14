# Makefile para el sistema de inferencia NLLB Awajún-Español

# Variables por defecto
MODEL_DIR = runs
BEST_MODEL = $(shell find $(MODEL_DIR) -name "best_model" -type d | head -n 1)
DATASET_VERSION = v1
BATCH_SIZE = 16

# Colores para output
RED = \033[31m
GREEN = \033[32m  
YELLOW = \033[33m
BLUE = \033[34m
RESET = \033[0m

.PHONY: help setup test list-models info clean
.DEFAULT_GOAL := help

help: ## Mostrar esta ayuda
	@echo "$(BLUE)Sistema de Inferencia NLLB Awajún-Español$(RESET)"
	@echo "============================================="
	@echo ""
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "$(GREEN)%-20s$(RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST)

setup: ## Configurar sistema inicial
	@echo "$(YELLOW)Configurando sistema...$(RESET)"
	@bash setup_inference.sh

test: ## Prueba rápida del sistema
	@echo "$(YELLOW)Ejecutando prueba rápida...$(RESET)"
	@if [ -d "$(BEST_MODEL)" ]; then \
		python3 predict.py --model_path "$(BEST_MODEL)" --direction es2agr --input "Hola mundo" --verbose; \
	else \
		echo "$(RED)Error: No hay modelos entrenados$(RESET)"; \
		echo "Ejecuta: make train-quick"; \
	fi

list-models: ## Listar modelos disponibles
	@echo "$(YELLOW)Modelos disponibles:$(RESET)"
	@python3 model_utils.py --command list_models

info: ## Información del primer modelo disponible
	@if [ -d "$(BEST_MODEL)" ]; then \
		echo "$(YELLOW)Información del modelo: $(BEST_MODEL)$(RESET)"; \
		python3 model_utils.py --command model_info --model_path "$(BEST_MODEL)"; \
	else \
		echo "$(RED)No hay modelos disponibles$(RESET)"; \
	fi

# Tareas de traducción
translate-text: ## Traducir texto (usar TEXT="tu texto" DIRECTION=es2agr)
	@if [ -z "$(TEXT)" ] || [ -z "$(DIRECTION)" ]; then \
		echo "$(RED)Uso: make translate-text TEXT=\"tu texto\" DIRECTION=es2agr$(RESET)"; \
	else \
		echo "$(YELLOW)Traduciendo: $(TEXT)$(RESET)"; \
		python3 predict.py --model_path "$(BEST_MODEL)" --direction "$(DIRECTION)" --input "$(TEXT)"; \
	fi

translate-file: ## Traducir archivo (usar INPUT_FILE=archivo.txt OUTPUT_FILE=salida.txt DIRECTION=es2agr)
	@if [ -z "$(INPUT_FILE)" ] || [ -z "$(OUTPUT_FILE)" ] || [ -z "$(DIRECTION)" ]; then \
		echo "$(RED)Uso: make translate-file INPUT_FILE=input.txt OUTPUT_FILE=output.txt DIRECTION=es2agr$(RESET)"; \
	else \
		echo "$(YELLOW)Traduciendo archivo: $(INPUT_FILE) → $(OUTPUT_FILE)$(RESET)"; \
		python3 predict.py --model_path "$(BEST_MODEL)" --direction "$(DIRECTION)" \
			--input_file "$(INPUT_FILE)" --output_file "$(OUTPUT_FILE)" --batch_size $(BATCH_SIZE); \
	fi

interactive: ## Modo interactivo (usar DIRECTION=es2agr)
	@if [ -z "$(DIRECTION)" ]; then \
		echo "$(RED)Uso: make interactive DIRECTION=es2agr$(RESET)"; \
	else \
		echo "$(YELLOW)Iniciando modo interactivo $(DIRECTION)...$(RESET)"; \
		python3 predict.py --model_path "$(BEST_MODEL)" --direction "$(DIRECTION)" --interactive; \
	fi

# Tareas de evaluación
evaluate: ## Evaluar modelo (usar DIRECTION=es2agr)
	@if [ -z "$(DIRECTION)" ]; then \
		echo "$(RED)Uso: make evaluate DIRECTION=es2agr$(RESET)"; \
	else \
		echo "$(YELLOW)Evaluando modelo $(DIRECTION)...$(RESET)"; \
		python3 evaluate.py --model_path "$(BEST_MODEL)" --direction "$(DIRECTION)" \
			--dataset_version $(DATASET_VERSION) --detailed_analysis; \
	fi

evaluate-quick: ## Evaluación rápida (500 muestras)
	@if [ -z "$(DIRECTION)" ]; then \
		echo "$(RED)Uso: make evaluate-quick DIRECTION=es2agr$(RESET)"; \
	else \
		echo "$(YELLOW)Evaluación rápida $(DIRECTION)...$(RESET)"; \
		python3 evaluate.py --model_path "$(BEST_MODEL)" --direction "$(DIRECTION)" \
			--sample_size 500 --batch_size $(BATCH_SIZE); \
	fi

compare-models: ## Comparar todos los modelos de una dirección (usar DIRECTION=es2agr)
	@if [ -z "$(DIRECTION)" ]; then \
		echo "$(RED)Uso: make compare-models DIRECTION=es2agr$(RESET)"; \
	else \
		echo "$(YELLOW)Comparando modelos $(DIRECTION)...$(RESET)"; \
		MODELS=$$(find $(MODEL_DIR) -path "*$(DIRECTION)*" -name "best_model" -type d); \
		if [ -n "$$MODELS" ]; then \
			python3 compare_models.py --models $$MODELS --direction "$(DIRECTION)" \
				--dataset_version $(DATASET_VERSION) --head_to_head; \
		else \
			echo "$(RED)No se encontraron modelos para $(DIRECTION)$(RESET)"; \
		fi; \
	fi

# Tareas de entrenamiento rápido (para pruebas)
train-quick-es2agr: ## Entrenamiento rápido Español → Awajún
	@echo "$(YELLOW)Entrenamiento rápido ES→AGR...$(RESET)"
	@python3 train.py --direction es2agr --dataset_version v1 --test_mode --batch_size 16

train-quick-agr2es: ## Entrenamiento rápido Awajún → Español  
	@echo "$(YELLOW)Entrenamiento rápido AGR→ES...$(RESET)"
	@python3 train.py --direction agr2es --dataset_version v1 --test_mode --batch_size 16

train-quick: train-quick-es2agr train-quick-agr2es ## Entrenamiento rápido ambas direcciones

# Ejemplos predefinidos
example-es2agr: ## Ejemplo Español → Awajún
	@echo "$(YELLOW)Ejemplo ES→AGR:$(RESET)"
	@echo "Hola mundo\nBuenos días\n¿Cómo estás?" | while read line; do \
		echo "ES: $$line"; \
		python3 predict.py --model_path "$(BEST_MODEL)" --direction es2agr --input "$$line" 2>/dev/null | grep -v "Cargando"; \
	done

example-agr2es: ## Ejemplo Awajún → Español
	@echo "$(YELLOW)Ejemplo AGR→ES:$(RESET)"
	@echo "Yama\nAme\nWararat ame" | while read line; do \
		echo "AGR: $$line"; \
		python3 predict.py --model_path "$(BEST_MODEL)" --direction agr2es --input "$$line" 2>/dev/null | grep -v "Cargando"; \
	done

# Benchmarks
benchmark: ## Benchmark completo del sistema
	@echo "$(YELLOW)Ejecutando benchmark completo...$(RESET)"
	@echo "1/4 Listando modelos..." && make list-models
	@echo "2/4 Evaluación rápida ES→AGR..." && make evaluate-quick DIRECTION=es2agr
	@echo "3/4 Evaluación rápida AGR→ES..." && make evaluate-quick DIRECTION=agr2es
	@echo "4/4 Ejemplos de traducción..." && make example-es2agr && make example-agr2es
	@echo "$(GREEN)Benchmark completado$(RESET)"

# Utilidades
clean: ## Limpiar archivos temporales y cache
	@echo "$(YELLOW)Limpiando cache...$(RESET)"
	@python3 model_utils.py --command clean_cache
	@rm -rf __pycache__ src/__pycache__
	@rm -f *.tmp *.temp
	@echo "$(GREEN)Cache limpiado$(RESET)"

install-deps: ## Instalar dependencias
	@echo "$(YELLOW)Instalando dependencias...$(RESET)"
	@pip install torch transformers sacrebleu pandas tqdm pyyaml matplotlib scikit-learn

check-deps: ## Verificar dependencias
	@echo "$(YELLOW)Verificando dependencias...$(RESET)"
	@python3 -c "import torch, transformers, sacrebleu, pandas, tqdm; print('✅ Todas las dependencias disponibles')" || echo "$(RED)❌ Faltan dependencias. Ejecuta: make install-deps$(RESET)"

status: ## Estado del sistema
	@echo "$(BLUE)Estado del Sistema NLLB$(RESET)"
	@echo "======================="
	@echo "Python: $$(python3 --version)"
	@echo "Modelos disponibles: $$(find $(MODEL_DIR) -name "best_model" -type d | wc -l)"
	@echo "GPU disponible: $$(python3 -c "import torch; print('Sí' if torch.cuda.is_available() else 'No')")"
	@echo "Espacio en disco (runs/): $$(du -sh $(MODEL_DIR) 2>/dev/null || echo 'N/A')"
	@make check-deps

# Tareas de desarrollo
dev-test: ## Pruebas para desarrollo
	@echo "$(YELLOW)Ejecutando pruebas de desarrollo...$(RESET)"
	@python3 -c "from src.inference import NLLBPredictor; print('✅ Módulo de inferencia OK')"
	@python3 -c "from src.evaluation import TranslationEvaluator; print('✅ Módulo de evaluación OK')"
	@echo "$(GREEN)Pruebas de desarrollo completadas$(RESET)"

# Documentación
docs: ## Abrir documentación
	@if command -v xdg-open >/dev/null; then \
		xdg-open README_INFERENCE.md; \
	elif command -v open >/dev/null; then \
		open README_INFERENCE.md; \
	else \
		echo "$(YELLOW)Ver documentación en: README_INFERENCE.md$(RESET)"; \
	fi

# Tareas combinadas útiles
quick-start: setup train-quick test ## Configuración completa desde cero
	@echo "$(GREEN)🎉 Sistema listo para usar!$(RESET)"

full-eval: ## Evaluación completa de todos los modelos
	@for direction in es2agr agr2es; do \
		echo "$(YELLOW)Evaluando dirección: $$direction$(RESET)"; \
		make evaluate DIRECTION=$$direction; \
	done
	@make compare-models DIRECTION=es2agr
	@make compare-models DIRECTION=agr2es