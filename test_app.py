#!/usr/bin/env python3
"""
Quick test script for EnviroVision India
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test if all required modules can be imported"""
    print("🔍 Testing imports...")
    
    try:
        import streamlit as st
        print("✅ Streamlit imported successfully")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
    
    try:
        import tensorflow as tf
        print("✅ TensorFlow imported successfully")
    except ImportError as e:
        print(f"❌ TensorFlow import failed: {e}")
    
    try:
        import numpy as np
        print("✅ NumPy imported successfully")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
    
    try:
        import cv2
        print("✅ OpenCV imported successfully")
    except ImportError as e:
        print(f"❌ OpenCV import failed: {e}")
    
    try:
        import plotly.graph_objects as go
        print("✅ Plotly imported successfully")
    except ImportError as e:
        print(f"❌ Plotly import failed: {e}")

def test_config():
    """Test configuration loading"""
    print("\n🔍 Testing configuration...")
    
    try:
        from utils.config_loader import config
        print("✅ Configuration loaded successfully")
        
        # Test some config values
        image_size = config.get('data.image_size')
        print(f"✅ Image size: {image_size}")
        
        categories = ['rainfall', 'heatwave', 'air_quality']
        for category in categories:
            classes = config.get(f'classes.{category}')
            print(f"✅ {category} classes: {classes}")
            
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")

def test_data_structure():
    """Test data directory structure"""
    print("\n🔍 Testing data structure...")
    
    data_dir = Path("data")
    if not data_dir.exists():
        print("❌ Data directory not found")
        return
    
    categories = ['rainfall', 'heatwave', 'air_quality']
    
    for category in categories:
        category_dir = data_dir / category
        if category_dir.exists():
            print(f"✅ {category} directory found")
            
            # Check subdirectories
            subdirs = [d for d in category_dir.iterdir() if d.is_dir()]
            print(f"   - Found {len(subdirs)} subdirectories: {[d.name for d in subdirs]}")
            
            # Count images
            total_images = 0
            for subdir in subdirs:
                images = list(subdir.glob("*.jpg")) + list(subdir.glob("*.png"))
                total_images += len(images)
                print(f"   - {subdir.name}: {len(images)} images")
            
            print(f"   - Total images in {category}: {total_images}")
        else:
            print(f"❌ {category} directory not found")

def test_models():
    """Test model files"""
    print("\n🔍 Testing model files...")
    
    models_dir = Path("models")
    if not models_dir.exists():
        print("❌ Models directory not found")
        return
    
    # Check for model files
    model_files = list(models_dir.glob("*.h5"))
    if model_files:
        print(f"✅ Found {len(model_files)} trained model files:")
        for model_file in model_files:
            print(f"   - {model_file.name}")
    else:
        print("⚠️ No trained model files found (.h5 files)")
        print("   - Use demo application for testing")
        print("   - Train models with: python scripts/train_models.py --category all")

def test_applications():
    """Test application files"""
    print("\n🔍 Testing application files...")
    
    app_dir = Path("app")
    if not app_dir.exists():
        print("❌ App directory not found")
        return
    
    # Check for app files
    app_files = list(app_dir.glob("*.py"))
    if app_files:
        print(f"✅ Found {len(app_files)} application files:")
        for app_file in app_files:
            print(f"   - {app_file.name}")
    else:
        print("❌ No application files found")

def main():
    """Run all tests"""
    print("🌍 EnviroVision India - System Test")
    print("=" * 50)
    
    test_imports()
    test_config()
    test_data_structure()
    test_models()
    test_applications()
    
    print("\n" + "=" * 50)
    print("🎉 Test completed!")
    print("\n📋 Next Steps:")
    print("1. Open http://localhost:8502 for demo application")
    print("2. Open http://localhost:8501 for full application")
    print("3. Upload images to test the classification system")
    print("4. Add more training data if you want real predictions")

if __name__ == "__main__":
    main() 