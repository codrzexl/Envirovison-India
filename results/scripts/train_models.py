#!/usr/bin/env python3
"""
Training script for EnviroVision India models
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config_loader import config
from utils.data_utils import DataPreprocessor
from models.transfer_learning import TransferLearningModel, VisionTransformerModel
from training.trainer import ModelTrainer
import argparse
from pathlib import Path

def train_category_model(category, architecture='efficientnet', use_vit=False):
    """Train model for a specific category"""
    print(f"\n{'='*50}")
    print(f"Training {category.upper()} Classification Model")
    print(f"Architecture: {architecture}")
    print(f"{'='*50}")
    
    # Initialize components
    preprocessor = DataPreprocessor(config)
    trainer = ModelTrainer(config, category)
    
    # Load and preprocess data
    data_dir = config.get('paths.data_dir')
    print(f"Loading data from: {data_dir}")
    
    try:
        df = preprocessor.load_dataset(data_dir, category)
        print(f"Loaded {len(df)} images for {category}")
        
        # Encode labels
        df = preprocessor.encode_labels(df, category)
        
        # Split data
        train_df, val_df, test_df = preprocessor.split_data(df)
        print(f"Data split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        # Create datasets
        train_dataset = preprocessor.create_tf_dataset(train_df, category, augment=True)
        val_dataset = preprocessor.create_tf_dataset(val_df, category, augment=False)
        test_dataset = preprocessor.create_tf_dataset(test_df, category, augment=False)
        
        # Create model
        if use_vit:
            model_builder = VisionTransformerModel(config, category)
            model = model_builder.create_model()
        else:
            model_builder = TransferLearningModel(config, category)
            model, base_model = model_builder.create_model()
        
        # Compile model
        model = model_builder.compile_model(model)
        
        print(f"\nModel Summary:")
        model.summary()
        
        # Train model
        history = trainer.train_model(model, train_dataset, val_dataset, train_df)
        
        # Fine-tune if using transfer learning
        if not use_vit:
            print("\nStarting fine-tuning...")
            history_fine = trainer.fine_tune_model(
                model, base_model, train_dataset, val_dataset, train_df
            )
        
        # Evaluate model
        class_names = config.get(f'classes.{category}')
        report, cm = trainer.evaluate_model(model, test_dataset, test_df, class_names)
        
        # Plot training history
        results_dir = Path(config.get('paths.results_dir'))
        trainer.plot_training_history(
            history, 
            save_path=results_dir / f'{category}_training_history.png'
        )
        
        # Save results
        trainer.save_training_results(history, report, category)
        
        # Save label encoders
        preprocessor.save_label_encoders(
            Path(config.get('paths.models_dir')) / 'label_encoders.json'
        )
        
        print(f"\n✅ {category.upper()} model training completed!")
        
    except Exception as e:
        print(f"❌ Error training {category} model: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Train EnviroVision India models')
    parser.add_argument('--category', type=str, choices=['rainfall', 'heatwave', 'air_quality', 'all'],
                       default='all', help='Category to train')
    parser.add_argument('--architecture', type=str, choices=['resnet18', 'efficientnet', 'efficientnet_b3'],
                       default='efficientnet', help='Model architecture')
    parser.add_argument('--use_vit', action='store_true', help='Use Vision Transformer instead of CNN')
    
    args = parser.parse_args()
    
    # Create necessary directories
    config.create_directories()
    
    # Train models
    categories = ['rainfall', 'heatwave', 'air_quality'] if args.category == 'all' else [args.category]
    
    for category in categories:
        try:
            train_category_model(category, args.architecture, args.use_vit)
        except Exception as e:
            print(f"Failed to train {category} model: {e}")
            continue
    
    print("\n🎉 All training completed!")

if __name__ == "__main__":
    main()
