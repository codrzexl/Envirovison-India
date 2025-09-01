import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

class GradCAM:
    def __init__(self, model, layer_name=None):
        self.model = model
        self.layer_name = layer_name or self._find_target_layer()
        
    def _find_target_layer(self):
        """Find the last convolutional layer in the model"""
        for layer in reversed(self.model.layers):
            # Check if layer has output_shape attribute and it's 4D (conv layer)
            if hasattr(layer, 'output_shape') and layer.output_shape is not None:
                if len(layer.output_shape) == 4:  # Conv layer has 4D output
                    return layer.name
            # Alternative check for layer type
            elif 'conv' in layer.name.lower() or 'Conv2D' in str(type(layer)):
                if hasattr(layer, 'output_shape') and layer.output_shape is not None:
                    if len(layer.output_shape) == 4:
                        return layer.name
        
        # If no conv layer found, try to find any layer with 4D output
        for layer in reversed(self.model.layers):
            if hasattr(layer, 'output_shape') and layer.output_shape is not None:
                if len(layer.output_shape) == 4:
                    return layer.name
        
        # Fallback: return the last layer that might work
        for layer in reversed(self.model.layers):
            if hasattr(layer, 'output_shape') and layer.output_shape is not None:
                return layer.name
        
        raise ValueError("Could not find a suitable layer for GradCAM")
    
    def generate_heatmap(self, image, class_index=None, alpha=0.4):
        """Generate Grad-CAM heatmap for an image"""
        # Ensure image has batch dimension
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        # Create a model that maps the input image to the activations
        # of the target layer as well as the output predictions
        grad_model = tf.keras.models.Model(
            [self.model.inputs], 
            [self.model.get_layer(self.layer_name).output, self.model.output]
        )
        
        # Compute the gradient of the top predicted class for our input image
        # with respect to the activations of the target layer
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(image)
            if class_index is None:
                class_index = tf.argmax(predictions[0])
            class_channel = predictions[:, class_index]
        
        # Compute gradients
        grads = tape.gradient(class_channel, conv_outputs)
        
        # Pool the gradients over all the axes leaving out the channel dimension
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight the channels by the corresponding gradients
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Normalize the heatmap
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        
        return heatmap.numpy()
    
    def overlay_heatmap(self, image, heatmap, alpha=0.4, colormap='jet'):
        """Overlay heatmap on original image"""
        # Denormalize image if it's normalized
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        
        # Resize heatmap to match image size
        heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        
        # Apply colormap to heatmap
        if colormap == 'jet':
            heatmap_colored = cv2.applyColorMap(
                (heatmap_resized * 255).astype(np.uint8), 
                cv2.COLORMAP_JET
            )
        else:
            # Custom colormap
            cmap = plt.get_cmap(colormap)
            heatmap_colored = (cmap(heatmap_resized) * 255).astype(np.uint8)[:, :, :3]
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_RGB2BGR)
        
        # Overlay heatmap on image
        overlayed = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
        
        return overlayed, heatmap_colored
    
    def generate_explanation(self, image, class_names, top_k=3):
        """Generate comprehensive explanation with top predictions and heatmaps"""
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
            # Generate heatmap for this class
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
                'rank': i + 1,
                'heatmap': heatmap,
                'overlay': overlayed,
                'heatmap_colored': heatmap_colored
            })
        
        return explanations
    
    def plot_explanations(self, image, explanations, figsize=(15, 5)):
        """Plot original image with top predictions and their heatmaps"""
        n_explanations = len(explanations)
        fig, axes = plt.subplots(1, n_explanations + 1, figsize=figsize)
        
        # Plot original image
        if len(image.shape) == 4:
            img_to_show = image[0]
        else:
            img_to_show = image
            
        if img_to_show.max() <= 1.0:
            img_to_show = (img_to_show * 255).astype(np.uint8)
            
        axes[0].imshow(img_to_show)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Plot explanations
        for i, explanation in enumerate(explanations):
            axes[i + 1].imshow(cv2.cvtColor(explanation['overlay'], cv2.COLOR_BGR2RGB))
            axes[i + 1].set_title(
                f"{explanation['class_name']}\n"
                f"Confidence: {explanation['confidence']:.3f}"
            )
            axes[i + 1].axis('off')
        
        plt.tight_layout()
        return fig
