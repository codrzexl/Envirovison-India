#!/usr/bin/env python3
"""
Model evaluation and comparison script
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path
import json

from utils.config_loader import config
from utils.data_utils import DataPreprocessor
from utils.grad_cam import GradCAM

def evaluate_all_models():
    """Evaluate all trained models"""
    
    categories = ['rainfall', 'heatwave', 'air_quality']
    models_dir = Path(config.get('paths.models_dir'))
    results_dir = Path(config.get('paths.results_dir'))
    
    preprocessor = DataPreprocessor(config)
    
    # Load label encoders
    encoder_path = models_dir / 'label_encoders.json'
    if encoder_path.exists():
        preprocessor.load_label_encoders(str(encoder_path))
    
    evaluation_results = {}
    
    for category in categories:
        print(f"\n{'='*50}")
        print(f"Evaluating {category.upper()} Model")
        print(f"{'='*50}")
        
        # Load model
        model_path = models_dir / f"{category}_best_model.h5"
        if not model_path.exists():
            print(f"❌ Model not found: {model_path}")
            continue
        
        try:
            model = tf.keras.models.load_model(str(model_path))
            print(f"✅ Loaded model: {model_path}")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            continue
        
        # Load test data
        try:
            data_dir = config.get('paths.data_dir')
            df = preprocessor.load_dataset(data_dir, category)
            df = preprocessor.encode_labels(df, category)
            
            # Split data (we need the test set)
            train_df, val_df, test_df = preprocessor.split_data(df)
            
            # Create test dataset
            test_dataset = preprocessor.create_tf_dataset(test_df, category, augment=False)
            
            print(f"Test set size: {len(test_df)} images")
            
        except Exception as e:
            print(f"❌ Failed to load test data: {e}")
            continue
        
        # Make predictions
        print("Making predictions...")
        predictions = model.predict(test_dataset, verbose=1)
        predicted_classes = np.argmax(predictions, axis=1)
        true_labels = test_df['encoded_label'].values
        
        # Get class names
        class_names = config.get(f'classes.{category}')
        
        # Calculate metrics
        report = classification_report(
            true_labels, 
            predicted_classes, 
            target_names=class_names,
            output_dict=True
        )
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(true_labels, predicted_classes, target_names=class_names))
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predicted_classes)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix - {category.title()}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(results_dir / f'{category}_evaluation_confusion_matrix.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # Store results
        evaluation_results[category] = {
            'accuracy': report['accuracy'],
            'macro_f1': report['macro avg']['f1-score'],
            'weighted_f1': report['weighted avg']['f1-score'],
            'classification_report': report,
            'confusion_matrix': cm.tolist()
        }
        
        print(f"✅ {category} evaluation completed!")
    
    # Save comprehensive evaluation results
    with open(results_dir / 'comprehensive_evaluation.json', 'w') as f:
        json.dump(evaluation_results, f, indent=2, default=str)
    
    # Create comparison chart
    create_model_comparison_chart(evaluation_results, results_dir)
    
    print(f"\n🎉 All model evaluations completed!")
    print(f"Results saved to: {results_dir}")

def create_model_comparison_chart(results, save_dir):
    """Create comparison chart for all models"""
    
    categories = list(results.keys())
    metrics = ['accuracy', 'macro_f1', 'weighted_f1']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, metric in enumerate(metrics):
        values = [results[cat][metric] for cat in categories]
        
        bars = axes[i].bar(categories, values, color=['skyblue', 'lightcoral', 'lightgreen'])
        axes[i].set_title(f'{metric.replace("_", " ").title()}')
        axes[i].set_ylabel('Score')
        axes[i].set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    evaluate_all_models()
