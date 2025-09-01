#!/usr/bin/env python3
"""
Create dummy model files to eliminate warnings
"""

import tensorflow as tf
import numpy as np
import json
from pathlib import Path

def create_dummy_model(category):
    """Create a dummy model for a category"""
    # Create a simple model
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(224, 224, 3)),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')  # 3 classes
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_dummy_label_encoders():
    """Create dummy label encoders"""
    encoders_data = {
        'rainfall': {
            'classes': ['Light', 'Moderate', 'Heavy']
        },
        'heatwave': {
            'classes': ['Normal', 'Mild', 'Extreme']
        },
        'air_quality': {
            'classes': ['Good', 'Moderate', 'Unhealthy']
        }
    }
    
    return encoders_data

def main():
    """Create dummy models and label encoders"""
    print("🔧 Creating dummy models to eliminate warnings...")
    
    # Create models directory if it doesn't exist
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Create dummy models for each category
    categories = ['rainfall', 'heatwave', 'air_quality']
    
    for category in categories:
        print(f"📦 Creating {category} model...")
        
        # Create model
        model = create_dummy_model(category)
        
        # Save model
        model_path = models_dir / f"{category}_best_model.h5"
        model.save(str(model_path))
        
        print(f"✅ Saved {model_path}")
    
    # Create dummy label encoders
    print("📝 Creating label encoders...")
    encoders_data = create_dummy_label_encoders()
    
    encoder_path = models_dir / 'label_encoders.json'
    with open(encoder_path, 'w') as f:
        json.dump(encoders_data, f, indent=2)
    
    print(f"✅ Saved {encoder_path}")
    
    print("\n🎉 Dummy models created successfully!")
    print("⚠️  Note: These are dummy models for testing only.")
    print("📊 For real predictions, train with actual data.")
    print("\n📋 Next steps:")
    print("1. Restart the full application: streamlit run app/streamlit_app.py")
    print("2. The model warnings should be gone")
    print("3. You'll see basic predictions (not accurate)")

if __name__ == "__main__":
    main() 