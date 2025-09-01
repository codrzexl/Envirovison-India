import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
from PIL import Image
# import albumentations as A  # Commented out due to installation issues
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import json
from pathlib import Path

class DataPreprocessor:
    def __init__(self, config):
        self.config = config
        self.image_size = config.get('data.image_size')
        self.batch_size = config.get('data.batch_size')
        self.label_encoders = {}
        
    def create_augmentation_pipeline(self):
        """Create data augmentation pipeline using TensorFlow instead of Albumentations"""
        return tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomContrast(0.1),
            tf.keras.layers.RandomBrightness(0.1),
        ])
    
    def preprocess_image(self, image_path, augment=False):
        """Preprocess a single image"""
        # Load image
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize
        image = cv2.resize(image, tuple(self.image_size))
        
        # Convert to float and normalize
        image = image.astype(np.float32) / 255.0
        
        # Apply ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std
        
        return image
    
    def load_dataset(self, data_dir, category):
        """Load dataset for a specific category (rainfall, heatwave, air_quality)"""
        data_path = Path(data_dir) / category
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {data_path}")
        
        images = []
        labels = []
        
        # Get class names for this category
        class_names = self.config.get(f'classes.{category}')
        
        for class_name in class_names:
            class_dir = data_path / class_name
            if class_dir.exists():
                for img_path in class_dir.glob('*.jpg'):
                    images.append(str(img_path))
                    labels.append(class_name)
                for img_path in class_dir.glob('*.png'):
                    images.append(str(img_path))
                    labels.append(class_name)
        
        # Create DataFrame
        df = pd.DataFrame({
            'image_path': images,
            'label': labels,
            'category': category
        })
        
        return df
    
    def encode_labels(self, df, category):
        """Encode labels for a category"""
        if category not in self.label_encoders:
            self.label_encoders[category] = LabelEncoder()
            df['encoded_label'] = self.label_encoders[category].fit_transform(df['label'])
        else:
            df['encoded_label'] = self.label_encoders[category].transform(df['label'])
        
        return df
    
    def create_tf_dataset(self, df, category, augment=False):
        """Create TensorFlow dataset"""
        def preprocess_fn(image_path, label):
            # Load and preprocess image
            image = tf.py_function(
                func=lambda x: self.preprocess_image(x.numpy().decode(), augment),
                inp=[image_path],
                Tout=tf.float32
            )
            image.set_shape([*self.image_size, 3])
            
            # One-hot encode label
            num_classes = self.config.get(f'data.num_classes.{category}')
            label = tf.one_hot(label, num_classes)
            
            return image, label
        
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices((
            df['image_path'].values,
            df['encoded_label'].values
        ))
        
        dataset = dataset.map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def split_data(self, df):
        """Split data into train, validation, and test sets"""
        val_split = self.config.get('data.validation_split')
        test_split = self.config.get('data.test_split')
        
        # First split: train+val and test
        train_val, test = train_test_split(
            df, test_size=test_split, stratify=df['label'], random_state=42
        )
        
        # Second split: train and val
        train, val = train_test_split(
            train_val, test_size=val_split/(1-test_split), 
            stratify=train_val['label'], random_state=42
        )
        
        return train, val, test
    
    def save_label_encoders(self, save_path):
        """Save label encoders"""
        encoders_dict = {}
        for category, encoder in self.label_encoders.items():
            encoders_dict[category] = {
                'classes': encoder.classes_.tolist()
            }
        
        with open(save_path, 'w') as f:
            json.dump(encoders_dict, f, indent=2)
    
    def load_label_encoders(self, load_path):
        """Load label encoders"""
        with open(load_path, 'r') as f:
            encoders_dict = json.load(f)
        
        for category, encoder_data in encoders_dict.items():
            encoder = LabelEncoder()
            encoder.classes_ = np.array(encoder_data['classes'])
            self.label_encoders[category] = encoder
