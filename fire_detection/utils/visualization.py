import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from data.dataloader import FireDataGenerator
from config import CONFIG

def generate_grad_cam(model, img_array, layer_name):
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]
    
    grads = tape.gradient(loss, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(conv_outputs, pooled_grads), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    return heatmap

def overlay_heatmap_on_image(img, heatmap):
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    return superimposed_img

def visualize_predictions(model, test_data, num_samples=20):
    test_generator = FireDataGenerator(
        test_data[0][:num_samples], 
        test_data[1][:num_samples], 
        batch_size=num_samples, 
        image_size=CONFIG['IMAGE_SIZE'],
        augment=False
    )
    images, true_masks = test_generator[0]
    pred_masks = model.predict(images)
    
    plt.figure(figsize=(15, 5 * num_samples))
    for i in range(min(num_samples, len(images))):
        plt.subplot(num_samples, 3, i * 3 + 1)
        plt.imshow(images[i])
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(num_samples, 3, i * 3 + 2)
        plt.imshow(true_masks[i, :, :, 0], cmap='gray')
        plt.title('Ground Truth Mask')
        plt.axis('off')
        
        plt.subplot(num_samples, 3, i * 3 + 3)
        plt.imshow(pred_masks[i, :, :, 0], cmap='gray')
        plt.title('Predicted Mask')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('prediction_results.png', dpi=300)
    plt.show()

def visualize_predictions_with_gradcam(model, test_data, layer_name, num_samples=5):
    test_generator = FireDataGenerator(
        test_data[0][:num_samples], 
        test_data[1][:num_samples], 
        batch_size=num_samples, 
        image_size=CONFIG['IMAGE_SIZE'],
        augment=False
    )
    images, true_masks = test_generator[0]
    pred_masks = model.predict(images)
    
    for i in range(min(num_samples, len(images))):
        plt.figure(figsize=(20, 5))
        
        plt.subplot(1, 5, 1)
        plt.imshow(images[i])
        plt.title("Original Image")
        plt.axis('off')
        
        plt.subplot(1, 5, 2)
        plt.imshow(images[i])
        plt.imshow(true_masks[i, :, :, 0], cmap='gray', alpha=0.5)
        plt.title("Ground Truth")
        plt.axis('off')
        
        plt.subplot(1, 5, 3)
        plt.imshow(images[i])
        plt.imshow(pred_masks[i, :, :, 0], cmap='Greens', alpha=0.5)
        plt.title("Prediction")
        plt.axis('off')
        
        img_array = np.expand_dims(images[i], axis=0)
        heatmap = generate_grad_cam(model, img_array, layer_name)
        superimposed_img = overlay_heatmap_on_image((images[i] * 255).astype(np.uint8), heatmap)
        
        plt.subplot(1, 5, 4)
        plt.imshow(superimposed_img)
        plt.title("Grad-CAM")
        plt.axis('off')
        
        plt.subplot(1, 5, 5)
        plt.imshow(images[i])
        plt.imshow(heatmap, cmap='jet', alpha=0.5)
        plt.title("Heatmap Overlay")
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(f"gradcam_example_{i}.png", dpi=300)
        plt.show()