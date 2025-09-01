import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import os
from datetime import datetime

class ModelTrainer:
    def __init__(self, config, category):
        self.config = config
        self.category = category
        self.model = None
        self.history = None
        self.base_model = None
        
    def create_callbacks(self, model_save_path):
        """Create training callbacks"""
        callbacks = []
        
        # Early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.config.get('model.early_stopping_patience'),
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
        
        # Model checkpoint
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=model_save_path,
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
        callbacks.append(checkpoint)
        
        # Reduce learning rate on plateau
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        callbacks.append(reduce_lr)
        
        # TensorBoard logging
        log_dir = Path(self.config.get('paths.logs_dir')) / f"{self.category}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        tensorboard = tf.keras.callbacks.TensorBoard(
            log_dir=str(log_dir),
            histogram_freq=1,
            write_graph=True,
            write_images=True
        )
        callbacks.append(tensorboard)
        
        return callbacks
    
    def calculate_class_weights(self, train_df):
        """Calculate class weights for imbalanced datasets"""
        from sklearn.utils.class_weight import compute_class_weight
        
        classes = np.unique(train_df['encoded_label'])
        class_weights = compute_class_weight(
            'balanced',
            classes=classes,
            y=train_df['encoded_label']
        )
        
        return dict(zip(classes, class_weights))
    
    def train_model(self, model, train_dataset, val_dataset, train_df=None):
        """Train the model"""
        # Create model save path
        models_dir = Path(self.config.get('paths.models_dir'))
        model_save_path = models_dir / f"{self.category}_best_model.h5"
        
        # Create callbacks
        callbacks = self.create_callbacks(str(model_save_path))
        
        # Calculate class weights if specified
        class_weight = None
        if self.config.get('training.class_weights') and train_df is not None:
            class_weight = self.calculate_class_weights(train_df)
            print(f"Using class weights: {class_weight}")
        
        # Train model
        epochs = self.config.get('model.epochs')
        
        print(f"Starting training for {self.category} classification...")
        print(f"Training for {epochs} epochs")
        
        history = model.fit(
            train_dataset,
            epochs=epochs,
            validation_data=val_dataset,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=1
        )
        
        self.history = history
        return history
    
    def fine_tune_model(self, model, base_model, train_dataset, val_dataset, train_df=None):
        """Fine-tune the model with unfrozen base layers"""
        print(f"Starting fine-tuning for {self.category}...")
        
        # Unfreeze base model
        base_model.trainable = True
        
        # Use a lower learning rate for fine-tuning
        fine_tune_lr = self.config.get('model.learning_rate') / 10
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=fine_tune_lr),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Create new callbacks for fine-tuning
        models_dir = Path(self.config.get('paths.models_dir'))
        model_save_path = models_dir / f"{self.category}_fine_tuned_model.h5"
        callbacks = self.create_callbacks(str(model_save_path))
        
        # Calculate class weights
        class_weight = None
        if self.config.get('training.class_weights') and train_df is not None:
            class_weight = self.calculate_class_weights(train_df)
        
        # Fine-tune for fewer epochs
        fine_tune_epochs = self.config.get('model.epochs') // 2
        
        history_fine = model.fit(
            train_dataset,
            epochs=fine_tune_epochs,
            validation_data=val_dataset,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=1
        )
        
        return history_fine
    
    def evaluate_model(self, model, test_dataset, test_df, class_names):
        """Evaluate model performance"""
        print(f"Evaluating {self.category} model...")
        
        # Get predictions
        predictions = model.predict(test_dataset, verbose=1)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Get true labels
        true_labels = test_df['encoded_label'].values
        
        # Classification report
        report = classification_report(
            true_labels, 
            predicted_classes, 
            target_names=class_names,
            output_dict=True
        )
        
        print("\nClassification Report:")
        print(classification_report(true_labels, predicted_classes, target_names=class_names))
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predicted_classes)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix - {self.category.title()}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Save confusion matrix
        results_dir = Path(self.config.get('paths.results_dir'))
        plt.savefig(results_dir / f'{self.category}_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return report, cm
    
    def plot_training_history(self, history, save_path=None):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(history.history['accuracy'], label='Training Accuracy')
        axes[0, 0].plot(history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss
        axes[0, 1].plot(history.history['loss'], label='Training Loss')
        axes[0, 1].plot(history.history['val_loss'], label='Validation Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # F1 Score (if available)
        if 'f1_score' in history.history:
            axes[1, 0].plot(history.history['f1_score'], label='Training F1')
            axes[1, 0].plot(history.history['val_f1_score'], label='Validation F1')
            axes[1, 0].set_title('F1 Score')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('F1 Score')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Learning Rate
        if 'lr' in history.history:
            axes[1, 1].plot(history.history['lr'], label='Learning Rate')
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_yscale('log')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return fig
    
    def save_training_results(self, history, report, category):
        """Save training results"""
        results_dir = Path(self.config.get('paths.results_dir'))
        
        # Save history
        history_dict = {
            'history': history.history,
            'category': category,
            'config': dict(self.config.config)
        }
        
        with open(results_dir / f'{category}_training_history.json', 'w') as f:
            json.dump(history_dict, f, indent=2, default=str)
        
        # Save classification report
        with open(results_dir / f'{category}_classification_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Training results saved to {results_dir}")
