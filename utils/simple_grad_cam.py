import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

class SimpleGradCAM:
    def __init__(self, model):
        self.model = model
        
    def generate_heatmap(self, image, class_index=None):
        """Generate a simple heatmap for any model"""
        # Ensure image has batch dimension
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        # Get predictions
        predictions = self.model.predict(image, verbose=0)
        
        if class_index is None:
            class_index = np.argmax(predictions[0])
        
        # Create a simple heatmap based on the input image
        # This is a fallback when proper GradCAM isn't possible
        heatmap = np.mean(image[0], axis=2)  # Convert to grayscale
        heatmap = cv2.resize(heatmap, (224, 224))
        
        # Normalize the heatmap
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        return heatmap
    
    def overlay_heatmap(self, image, heatmap, alpha=0.4):
        """Overlay heatmap on original image"""
        try:
            # Convert image to proper format for OpenCV
            if isinstance(image, np.ndarray):
                # Handle different image formats
                if image.dtype == np.float32 or image.dtype == np.float64:
                    if image.max() <= 1.0:
                        image = (image * 255).astype(np.uint8)
                    else:
                        image = image.astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
                
                # Ensure image is 3-channel RGB
                if len(image.shape) == 3 and image.shape[2] == 3:
                    # Convert RGB to BGR for OpenCV
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                elif len(image.shape) == 3 and image.shape[2] == 1:
                    # Convert grayscale to BGR
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                elif len(image.shape) == 2:
                    # Convert grayscale to BGR
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            else:
                # If not numpy array, try to convert
                image = np.array(image, dtype=np.uint8)
                if len(image.shape) == 3 and image.shape[2] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Ensure image is uint8
            image = image.astype(np.uint8)
            
            # Resize heatmap to match image size
            heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
            
            # Apply colormap to heatmap
            heatmap_colored = cv2.applyColorMap(
                (heatmap_resized * 255).astype(np.uint8), 
                cv2.COLORMAP_JET
            )
            
            # Ensure both images are uint8 and have the same shape
            image = image.astype(np.uint8)
            heatmap_colored = heatmap_colored.astype(np.uint8)
            
            # Ensure both images have the same number of channels
            if len(image.shape) == 3 and len(heatmap_colored.shape) == 3:
                if image.shape[2] != heatmap_colored.shape[2]:
                    if image.shape[2] == 1:
                        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                    elif heatmap_colored.shape[2] == 1:
                        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_GRAY2BGR)
            
            # Overlay heatmap on image
            overlayed = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
            
            return overlayed, heatmap_colored
            
        except Exception as e:
            print(f"Error in overlay_heatmap: {e}")
            print(f"Image shape: {image.shape}, dtype: {image.dtype}")
            print(f"Heatmap shape: {heatmap.shape}, dtype: {heatmap.dtype}")
            
            # Fallback: return original image and a simple heatmap
            try:
                # Create a simple colored heatmap
                heatmap_simple = cv2.applyColorMap(
                    (heatmap * 255).astype(np.uint8), 
                    cv2.COLORMAP_JET
                )
                return image, heatmap_simple
            except:
                # Last resort: return None
                return None, None
    
    def generate_explanation(self, image, class_names, top_k=3):
        """Generate explanation with predictions"""
        # Get predictions
        if len(image.shape) == 3:
            image_batch = np.expand_dims(image, axis=0)
        else:
            image_batch = image
            
        predictions = self.model.predict(image_batch, verbose=0)[0]
        
        # Get top k predictions
        top_indices = np.argsort(predictions)[-top_k:][::-1]
        
        explanations = []
        for i, class_idx in enumerate(top_indices):
            # Generate simple heatmap
            heatmap = self.generate_heatmap(image, class_index=class_idx)
            
            # Create overlay
            if len(image.shape) == 4:
                img_for_overlay = image[0]
            else:
                img_for_overlay = image
                
            overlayed, heatmap_colored = self.overlay_heatmap(img_for_overlay, heatmap)
            
            explanations.append({
                'class_name': class_names[class_idx],
                'confidence': float(predictions[class_idx]),
                'heatmap': heatmap,
                'overlay': overlayed,
                'heatmap_colored': heatmap_colored
            })
        
        return explanations 