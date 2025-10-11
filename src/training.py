"""
Lógica principal de entrenamiento con MLflow tracking
Modificado para soportar evaluación cada N épocas
"""

import os
import time
import logging
import matplotlib.pyplot as plt
import pandas as pd
import mlflow
import mlflow.pytorch
from tqdm import tqdm

from src.dataset import AwajunDataLoader, TranslationDataset
from src.model import NLLBModel
from src.evaluation import TranslationEvaluator
from src.utils import EarlyStopping, create_run_dir, format_time

logger = logging.getLogger(__name__)

class Trainer:
    """Trainer principal para fine-tuning NLLB"""
    
    def __init__(self, config):
        self.config = config
        self.setup_experiment()
        self.setup_data()
        self.setup_model()
        self.setup_evaluation()
        self.setup_training_state()
        
    def setup_experiment(self):
        """Configurar experimento MLflow - Versión mejorada"""
        # Crear nombre de experimento más descriptivo
        direction = self.config['experiment']['direction']
        dataset_version = self.config['data']['dataset_version']
        
        # Experimento principal por dirección
        experiment_name = f"awajun_translation_{direction}"
        
        # Configurar MLflow
        mlflow.set_tracking_uri(self.config['experiment']['mlflow_uri'])
        mlflow.set_experiment(experiment_name)
        
        logger.info(f"🧪 Experimento: {experiment_name}")
        logger.info(f"📊 Dataset version: {dataset_version}")
        
        # El modelo se loggea como parámetro, no en el nombre
        self.model_display_name = self.config['model'].get('display_name', self.config['model']['name'])
        
    def setup_data(self):
        """Configurar datos de entrenamiento"""
        self.data_loader = AwajunDataLoader(self.config)
        self.df_train, self.df_dev = self.data_loader.load_data()
        
        # Obtener configuración de idiomas
        direction = self.config['experiment']['direction']
        if direction == 'es2agr':
            self.src_lang, self.tgt_lang = 'es', 'agr'
        else:
            self.src_lang, self.tgt_lang = 'agr', 'es'
            
        logger.info(f"🔄 Dirección: {self.src_lang.upper()} → {self.tgt_lang.upper()}")
        
        # Configurar tags para mejor organización
        self.run_tags = {
            "direction": direction,
            "dataset_version": self.config['data']['dataset_version'],
            "model": self.model_display_name,
            "balance_method": self.config['data']['balance_method'],
            "language_pair": f"{self.src_lang}-{self.tgt_lang}",
            "experiment_type": "fine_tuning"
        }
        
    def setup_model(self):
        """Configurar modelo y tokenizer"""
        self.model_wrapper = NLLBModel(self.config)
        
        # Obtener tokens de idioma
        self.src_token, self.tgt_token, _, _ = self.model_wrapper.get_language_tokens(
            self.config['experiment']['direction']
        )
        
        logger.info(f"🏷️  Tokens: {self.src_token} → {self.tgt_token}")
        
    def setup_evaluation(self):
        """Configurar evaluador"""
        self.evaluator = TranslationEvaluator(self.model_wrapper, self.config)
        
    def setup_training_state(self):
        """Configurar estado de entrenamiento"""
        self.early_stopping = EarlyStopping(
            patience=self.config['training']['patience'],
            min_delta=self.config['training']['min_improvement'],
            mode='max'
        )
        
        self.training_history = []
        self.best_metrics = {}
        self.start_time = None
        
        # Crear directorio para guardar modelo
        self.run_dir = create_run_dir("runs", f"nllb_{self.config['experiment']['direction']}")
        
        # NUEVO: Obtener frecuencia de evaluación
        self.eval_frequency = self.config.get('evaluation', {}).get('eval_frequency', 1)
        logger.info(f"📈 Evaluación chrF++ cada {self.eval_frequency} épocas")
        
    def log_dataset_info(self):
        """Log información del dataset a MLflow - Versión mejorada"""
        try:
            import mlflow.data
            from mlflow.data.sources import LocalArtifactDatasetSource

            # Crear nombre más descriptivo del dataset
            dataset_version = self.config['data']['dataset_version']
            direction = self.config['experiment']['direction']
            model_name = self.config['model']['display_name']

            train_dataset = mlflow.data.from_pandas(
                self.df_train,
                source=LocalArtifactDatasetSource(f"data/awajun-spanish-{dataset_version}/train.*"),
                name=f"{direction}-{dataset_version}-{self.config['data']['balance_method']}",
                targets=self.tgt_lang
            )
           
            dev_dataset = mlflow.data.from_pandas(
                self.df_dev,
                source=LocalArtifactDatasetSource(f"data/awajun-spanish-{dataset_version}/dev.*"),
                name=f"{direction}-{dataset_version}-{self.config['data']['balance_method']}",
                targets=self.tgt_lang
            )
            
            # Log datasets con contexto específico
            mlflow.log_input(train_dataset, context="training")
            mlflow.log_input(dev_dataset, context="validation")
            
            # Log dataset stats con prefijo de versión
            dataset_metrics = {
                f"dataset_{dataset_version}_train_size": len(self.df_train),
                f"dataset_{dataset_version}_dev_size": len(self.df_dev),
                f"dataset_{dataset_version}_domains": self.df_train['source'].nunique(),
                f"dataset_{dataset_version}_avg_src_length": self.df_train[self.src_lang].str.len().mean(),
                f"dataset_{dataset_version}_avg_tgt_length": self.df_train[self.tgt_lang].str.len().mean(),
                f"dataset_{dataset_version}_total_size": len(self.df_train) + len(self.df_dev),
            }
            
            mlflow.log_metrics(dataset_metrics)
            
            # Log domain distribution con prefijo de versión
            domain_dist = self.df_train['source'].value_counts()
            for domain, count in domain_dist.items():
                metric_name = f"dataset_{dataset_version}_domain_{domain.replace(' ', '_').replace('&', 'and')}_count"
                mlflow.log_metric(metric_name, count)
           
            
            # Log dataset summary as artifact
            self._create_dataset_summary_artifact(dataset_version, domain_dist)
            
            logger.info(f"✅ Datasets (version {dataset_version}) logged to MLflow")
            
        except Exception as e:
            logger.warning(f"⚠️ Error logging dataset: {e}")
            # No fallar el entrenamiento por esto
            
    def _create_dataset_summary_artifact(self, dataset_version, domain_dist):
        """Crear artifact con resumen del dataset"""
        try:
            import json
            
            summary = {
                "dataset_version": dataset_version,
                "direction": self.config['experiment']['direction'],
                "model": self.config['model']['display_name'],
                "language_pair": f"{self.src_lang}-{self.tgt_lang}",
                "train_size": len(self.df_train),
                "dev_size": len(self.df_dev),
                "total_size": len(self.df_train) + len(self.df_dev),
                "balance_method": self.config['data']['balance_method'],
                "domains": domain_dist.to_dict(),
                "avg_lengths": {
                    "source": self.df_train[self.src_lang].str.len().mean(),
                    "target": self.df_train[self.tgt_lang].str.len().mean()
                },
                "domain_distribution": {
                    "count": domain_dist.to_dict(),
                    "percentage": (domain_dist / domain_dist.sum() * 100).round(2).to_dict()
                }
            }
            
            # Guardar como archivo JSON
            summary_path = os.path.join(self.run_dir, f"dataset_summary_{dataset_version}.json")
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
                
            mlflow.log_artifact(summary_path, "dataset_info")
            
        except Exception as e:
            logger.warning(f"⚠️ Error creating dataset summary: {e}")
    
    def create_dataset_and_dataloader(self):
        """Crear dataset y dataloader"""
        # Dataset
        dataset = TranslationDataset(
            self.df_train,
            self.src_lang,
            self.tgt_lang,
            self.model_wrapper.tokenizer,
            self.src_token,
            self.tgt_token,
            self.config['model']['max_length']
        )
        
        # Sampler con pesos si está configurado
        sampler = self.data_loader.create_weighted_sampler(self.df_train)
        
        # DataLoader
        dataloader = self.data_loader.create_dataloader(
            dataset, sampler=sampler, shuffle=(sampler is None)
        )
        
        # Log información útil
        logger.info(f"📊 Dataset original: {len(self.df_train)} samples")
        logger.info(f"📊 Batch size: {self.config['training']['batch_size']}")
        logger.info(f"📊 Steps por época: {len(dataloader)}")
        logger.info(f"📊 Samples efectivos por época: {len(dataloader) * self.config['training']['batch_size']}")
        
        return dataloader
    
    def train_epoch(self, dataloader, epoch):
        """Entrenar una época"""
        logger.info(f"🚀 Iniciando época {epoch+1}...")
        self.model_wrapper.model.train()
        epoch_losses = []
        
        pbar = tqdm(dataloader, desc=f"Época {epoch+1}/{self.config['training']['epochs']}")
        
        for batch_idx, batch in enumerate(pbar):
            src_texts, tgt_texts = batch
            
            # Log del primer batch para debugging
            if batch_idx == 0:
                logger.info(f"🔍 Procesando primer batch con {len(src_texts)} samples")
            
            # Paso de entrenamiento
            loss = self.model_wrapper.train_step(
                src_texts, tgt_texts, self.src_token, self.tgt_token
            )
            
            epoch_losses.append(loss)
            
            # Actualizar barra de progreso
            current_lr = self.model_wrapper.get_current_lr()
            pbar.set_postfix({
                'loss': f"{loss:.3f}",
                'lr': f"{current_lr:.1e}"
            })
        
        return sum(epoch_losses) / len(epoch_losses)
    
    def evaluate_epoch(self, epoch):
        """Evaluar modelo en epoch actual"""
        metrics = self.evaluator.evaluate_model(
            self.df_dev,
            self.src_lang,
            self.tgt_lang,
            self.src_token,
            self.tgt_token
        )
        
        # Añadir métricas de entrenamiento
        metrics['train_lr'] = self.model_wrapper.get_current_lr()
        metrics['epoch'] = epoch
        
        return metrics
    
    def should_evaluate_this_epoch(self, epoch):
        """Determinar si se debe evaluar en esta época"""
        # Siempre evaluar en la primera época
        if epoch == 0:
            return True
        
        # Siempre evaluar en la última época
        if epoch == self.config['training']['epochs'] - 1:
            return True
        
        # Evaluar según frecuencia configurada
        if (epoch + 1) % self.eval_frequency == 0:
            return True
        
        return False
    
    def log_metrics(self, metrics, epoch):
        """Loggear métricas a MLflow"""
        # Log all metrics
        mlflow.log_metrics(metrics, step=epoch)
        
        # Actualizar mejores métricas si hay chrF++ en las métricas
        if 'eval_chrf' in metrics:
            chrf_score = metrics.get('eval_chrf', 0)
            if chrf_score > self.best_metrics.get('best_chrf', 0):
                self.best_metrics['best_chrf'] = chrf_score
                self.best_metrics['best_epoch'] = epoch
                self.save_best_model()
            
            # Log best metrics
            mlflow.log_metrics({
                'best_chrf': self.best_metrics.get('best_chrf', 0),
                'best_epoch': self.best_metrics.get('best_epoch', 0)
            }, step=epoch)
        
    def save_best_model(self):
        """Guardar mejor modelo"""
        best_model_path = os.path.join(self.run_dir, "best_model")
        self.model_wrapper.save_model(best_model_path)
        logger.info(f"💾 Mejor modelo guardado (chrF++: {self.best_metrics.get('best_chrf', 0):.2f})")
        
    def create_loss_plot(self, losses):
        """Crear gráfico de pérdida - Versión corregida"""
        if len(losses) < 2:
            logger.info("⚠️ Muy pocas pérdidas para crear gráfico")
            return None
            
        plt.figure(figsize=(10, 6))
        
        # Calcular span mínimo para suavizado
        min_span = max(1, min(50, len(losses)//4))  # Asegurar que span >= 1
        
        # Solo hacer suavizado si hay suficientes datos
        if len(losses) >= 4:
            loss_series = pd.Series(losses)
            smoothed = loss_series.ewm(span=min_span).mean()
            plt.plot(smoothed, label='Loss suavizado', linewidth=2, color='blue')
            plt.plot(losses, alpha=0.3, label='Loss original', color='lightblue')
        else:
            # Para muy pocos datos, solo mostrar la línea original
            plt.plot(losses, label='Loss', linewidth=2, color='blue', marker='o')
        
        plt.title(f'Pérdida de Entrenamiento - Dataset {self.config["data"]["dataset_version"]}')
        plt.xlabel('Épocas')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Añadir información del dataset
        plt.text(0.02, 0.98, f'Dataset: {self.config["data"]["dataset_version"]}', 
                 transform=plt.gca().transAxes, fontsize=10,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plot_path = os.path.join(self.run_dir, f"loss_plot_{self.config['data']['dataset_version']}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📊 Gráfico de pérdida guardado: {plot_path}")
        return plot_path
    
    def log_training_summary(self, total_time, stopped_early=False):
        """Loggear resumen final del entrenamiento"""
        summary = {
            'total_training_time_minutes': total_time / 60,
            'early_stopped': stopped_early,
            'final_epoch': len(self.training_history),
        }
        
        # Métricas de convergencia
        if self.training_history:
            convergence = self.evaluator.calculate_convergence_metrics(self.training_history)
            summary.update(convergence)
        
        mlflow.log_metrics(summary)
        
        # Log training config - Versión mejorada
        mlflow.log_params({
            'model_name': self.model_display_name,
            'model_path': self.config['model']['name'],
            'dataset_version': self.config['data']['dataset_version'],
            'direction': self.config['experiment']['direction'],
            'epochs': self.config['training']['epochs'],
            'batch_size': self.config['training']['batch_size'],
            'learning_rate': self.config['training']['learning_rate'],
            'patience': self.config['training']['patience'],
            'eval_frequency': self.eval_frequency,  # NUEVO
            'balance_method': self.config['data']['balance_method'],
            'max_length': self.config['model']['max_length'],
            'warmup_steps': self.config['training']['warmup_steps'],
            'weight_decay': self.config['training']['weight_decay'],
            'clip_threshold': self.config['training']['clip_threshold']
        })
        
    def print_epoch_summary(self, epoch, metrics, epoch_time, evaluated=True):
        """Imprimir resumen de época"""
        if not evaluated:
            # Solo mostrar loss si no se evaluó
            loss = metrics.get('train_loss_epoch', 0)
            print(f"📊 Época {epoch+1} completada - "
                  f"Loss: {loss:.4f} | "
                  f"Tiempo: {format_time(epoch_time)} "
                  f"[Eval programada cada {self.eval_frequency} épocas]")
            return
        
        # Mostrar métricas completas si se evaluó
        chrf = metrics.get('eval_chrf', 0)
        bleu = metrics.get('eval_bleu', 0)
        
        # Flechas de tendencia
        chrf_trend = "📈" if chrf > self.best_metrics.get('prev_chrf', 0) else "📉"
        bleu_trend = "📈" if bleu > self.best_metrics.get('prev_bleu', 0) else "📉"
        
        # Indicador de mejor modelo
        best_indicator = " ✨ MEJOR" if chrf == self.best_metrics.get('best_chrf', 0) else ""
        
        print(f"📊 Época {epoch+1} completada - "
              f"CHRF++: {chrf:.2f} {chrf_trend} | "
              f"BLEU: {bleu:.2f} {bleu_trend} | "
              f"Tiempo: {format_time(epoch_time)}{best_indicator}")
        
        # Guardar para próxima comparación
        self.best_metrics['prev_chrf'] = chrf
        self.best_metrics['prev_bleu'] = bleu
        
    def run(self):
        """Ejecutar entrenamiento completo"""
        logger.info("🚀 Iniciando entrenamiento...")
        
        # Crear dataloader
        dataloader = self.create_dataset_and_dataloader()
        
        # Información inicial
        total_steps = len(dataloader) * self.config['training']['epochs']
        logger.info(f"📊 Pasos totales: {total_steps}")
        logger.info(f"⚖️  Método de balanceo: {self.config['data']['balance_method']}")
        logger.info(f"📈 Evaluación chrF++ cada: {self.eval_frequency} épocas")
        
        epoch_losses = []
        self.start_time = time.time()
        
        # Iniciar run con tags mejorados
        with mlflow.start_run(tags=self.run_tags):
            # Log dataset information primero
            self.log_dataset_info()
            
            for epoch in range(self.config['training']['epochs']):
                epoch_start = time.time()
                
                # Entrenamiento
                avg_loss = self.train_epoch(dataloader, epoch)
                epoch_losses.append(avg_loss)
                
                # NUEVO: Decidir si evaluar en esta época
                should_eval = self.should_evaluate_this_epoch(epoch)
                
                if should_eval:
                    logger.info(f"📊 Evaluando modelo (época {epoch+1})...")
                    
                    # Evaluación completa
                    metrics = self.evaluate_epoch(epoch)
                    metrics['train_loss_epoch'] = avg_loss
                    
                    # Early stopping basado en chrF++
                    chrf_score = metrics.get('eval_chrf', 0)
                    should_stop = self.early_stopping(chrf_score)
                    
                    # Logging
                    self.log_metrics(metrics, epoch)
                    self.training_history.append(metrics)
                    
                    # Muestras de traducción
                    if epoch % (self.eval_frequency * 3) == 0:  # Cada 3 evaluaciones
                        samples = self.evaluator.get_sample_translations(
                            self.df_dev, self.src_lang, self.tgt_lang, self.src_token
                        )
                        self.evaluator.log_sample_translations(samples, epoch)
                    
                    # Mostrar progreso con métricas
                    epoch_time = time.time() - epoch_start
                    self.print_epoch_summary(epoch, metrics, epoch_time, evaluated=True)
                    
                    # Early stopping
                    if should_stop:
                        logger.info(f"⏰ Early stopping activado en época {epoch+1}")
                        logger.info(f"   Mejor chrF++: {self.best_metrics.get('best_chrf', 0):.2f} en época {self.best_metrics.get('best_epoch', 0)+1}")
                        break
                else:
                    # Solo loggear loss si no evaluamos
                    metrics = {'train_loss_epoch': avg_loss}
                    mlflow.log_metrics(metrics, step=epoch)
                    
                    # Mostrar progreso simplificado
                    epoch_time = time.time() - epoch_start
                    self.print_epoch_summary(epoch, metrics, epoch_time, evaluated=False)
            
            # Finalización
            total_time = time.time() - self.start_time
            
            # Crear y loggear gráfico de pérdida
            try:
                loss_plot_path = self.create_loss_plot(epoch_losses)
                if loss_plot_path and os.path.exists(loss_plot_path):
                    mlflow.log_artifact(loss_plot_path, "plots")
                    logger.info(f"📊 Gráfico de pérdida loggeado en MLflow")
            except Exception as e:
                logger.warning(f"⚠️ Error creando gráfico de pérdida: {e}")
            
            # Guardar modelo final
            final_model_path = os.path.join(self.run_dir, "final_model")
            self.model_wrapper.save_model(final_model_path)
            
            # Log artifacts
            try:
                mlflow.log_artifacts(self.run_dir, "models")
                logger.info(f"📁 Artifacts loggeados en MLflow")
            except Exception as e:
                logger.warning(f"⚠️ Error loggeando artifacts: {e}")
            
            # Log modelo a MLflow con nombre personalizado
            try:
                model_name = f"{self.model_display_name}-{self.config['data']['dataset_version']}"
                mlflow.pytorch.log_model(
                    self.model_wrapper.model, 
                    "model",
                    registered_model_name=model_name
                )
                logger.info(f"🤖 Modelo registrado como: {model_name}")
            except Exception as e:
                logger.warning(f"⚠️ Error registrando modelo: {e}")
            
            # Resumen final
            self.log_training_summary(total_time, self.early_stopping.early_stop)
            
            best_chrf = self.best_metrics.get('best_chrf', 0)
            best_epoch = self.best_metrics.get('best_epoch', 0)
            
            logger.info("✅ Entrenamiento completado!")
            logger.info(f"📊 Dataset: {self.config['data']['dataset_version']}")
            logger.info(f"🔄 Dirección: {self.config['experiment']['direction']}")
            logger.info(f"🏆 Mejor CHRF++: {best_chrf:.2f} en época {best_epoch+1}")
            logger.info(f"⏱️  Tiempo total: {format_time(total_time)}")
            logger.info(f"💾 Modelos guardados en: {self.run_dir}")
            
            return {
                'best_chrf': best_chrf,
                'best_epoch': best_epoch,
                'total_time': total_time,
                'dataset_version': self.config['data']['dataset_version'],
                'direction': self.config['experiment']['direction']
            }