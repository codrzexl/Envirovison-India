#!/usr/bin/env python3
"""
Script to create sample data structure for EnviroVision India
"""

import os
from pathlib import Path
import requests
from PIL import Image
import numpy as np

def create_data_structure():
    """Create the required data directory structure"""
    
    categories = {
        'rainfall': ['Light', 'Moderate', 'Heavy'],
        'heatwave': ['Normal', 'Mild', 'Extreme'],
        'air_quality': ['Good', 'Moderate', 'Unhealthy']
    }
    
    base_data_dir = Path('data')
    
    for category, classes in categories.items():
        for class_name in classes:
            class_dir = base_data_dir / category / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {class_dir}")
    
    print("\n✅ Data directory structure created!")
    print("\n📁 Directory structure:")
    print("data/")
    for category, classes in categories.items():
        print(f"├── {category}/")
        for class_name in classes:
            print(f"│   ├── {class_name}/")
            print(f"│   │   └── (place your images here)")
    
    print("\n📝 Instructions:")
    print("1. Place your images in the appropriate category and class folders")
    print("2. Supported formats: .jpg, .jpeg, .png")
    print("3. Recommended: At least 100 images per class for good performance")
    print("4. Images will be automatically resized to 224x224 during training")

def create_sample_images():
    """Create sample synthetic images for testing (optional)"""
    print("\n🎨 Creating sample synthetic images for testing...")
    
    categories = {
        'rainfall': {
            'Light': (200, 220, 255),      # Light blue
            'Moderate': (100, 150, 255),   # Medium blue
            'Heavy': (50, 100, 200)        # Dark blue
        },
        'heatwave': {
            'Normal': (100, 200, 100),     # Green
            'Mild': (255, 200, 100),       # Orange
            'Extreme': (255, 100, 100)     # Red
        },
        'air_quality': {
            'Good': (150, 255, 150),       # Light green
            'Moderate': (255, 255, 150),   # Yellow
            'Unhealthy': (255, 150, 150)   # Light red
        }
    }
    
    base_data_dir = Path('data')
    
    for category, classes in categories.items():
        for class_name, color in classes.items():
            class_dir = base_data_dir / category / class_name
            
            # Create 5 sample images per class
            for i in range(5):
                # Create a simple colored image with some noise
                img_array = np.full((224, 224, 3), color, dtype=np.uint8)
                
                # Add some random noise
                noise = np.random.randint(-30, 30, (224, 224, 3))
                img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
                
                # Create PIL image and save
                img = Image.fromarray(img_array)
                img_path = class_dir / f"sample_{i+1}.jpg"
                img.save(img_path)
            
            print(f"Created 5 sample images in {class_dir}")
    
    print("\n✅ Sample images created!")
    print("⚠️  Note: These are synthetic images for testing only.")
    print("   Replace with real environmental images for actual training.")

if __name__ == "__main__":
    create_data_structure()
    
    # Ask user if they want to create sample images
    create_samples = input("\n❓ Create sample synthetic images for testing? (y/n): ").lower().strip()
    if create_samples == 'y':
        create_sample_images()
    
    print("\n🚀 Setup complete! You can now:")
    print("1. Add your real images to the data folders")
    print("2. Run: python scripts/train_models.py --category all")
    print("3. Launch the app: streamlit run app/streamlit_app.py")
