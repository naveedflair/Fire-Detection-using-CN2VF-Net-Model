import os
import json
import numpy as np
import tensorflow as tf
from config import CONFIG
from data.preprocessing import load_dataset
from models.hybrid_model import build_hybrid_transformer_cnn_model
from training.train import train_model, evaluate_model
from utils.visualization import visualize_predictions, visualize_predictions_with_gradcam
from utils.metrics import calculate_statistical_significance

def main():
    # Check for GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"Found {len(gpus)} GPU(s): {gpus}")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("No GPU found, using CPU")
    
    # Create output directory
    os.makedirs('results', exist_ok=True)
    
    # Print configuration
    print("Configuration:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    
    # Load dataset
    print("\nLoading dataset...")
    try:
        train_data, val_data, test_data = load_dataset(CONFIG['DATASET_PATH'])
        print(f"Dataset loaded: {len(train_data[0])} train, {len(val_data[0])} val, {len(test_data[0])} test samples")
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return
    
    # Train and evaluate models
    if CONFIG['MODEL_TYPE'] in ['hybrid', 'both']:
        print("\nBuilding and training Hybrid Transformer-CNN Segmentation Model...")
        try:
            hybrid_model = build_hybrid_transformer_cnn_model(
                input_shape=(*CONFIG['IMAGE_SIZE'], 3),
                num_patches=CONFIG['NUM_PATCHES'],
                patch_size=CONFIG['PATCH_SIZE'],
                transformer_layers=CONFIG['TRANSFORMER_LAYERS'],
                transformer_heads=CONFIG['TRANSFORMER_HEADS'],
                transformer_dim=CONFIG['TRANSFORMER_DIM'],
                transformer_mlp_dim=CONFIG['TRANSFORMER_MLP_DIM']
            )
            
            hybrid_model.summary()
            
            history, hybrid_model, metrics = train_model(
                hybrid_model, 
                train_data, 
                val_data, 
                CONFIG
            )
            
            with open('results/hybrid_model_metrics.json', 'w') as f:
                json.dump(metrics, f, indent=2)
            
            print("\nEvaluating Hybrid Transformer-CNN Model...")
            hybrid_results, y_true, y_pred_hybrid = evaluate_model(hybrid_model, test_data, CONFIG)
            
            np.save('results/hybrid_predictions.npy', y_pred_hybrid)
            np.save('results/true_labels.npy', y_true)
            
            print("\nVisualizing predictions...")
            visualize_predictions(hybrid_model, test_data)
            visualize_predictions_with_gradcam(hybrid_model, test_data, 'top_activation')
            
        except Exception as e:
            print(f"Error with Hybrid Model: {str(e)}")
    
    # Generate final report
    print("\n=== FINAL REPORT ===")
    print(f"\nDataset Statistics:")
    print(f"- Training samples: {len(train_data[0])}")
    print(f"- Validation samples: {len(val_data[0])}")
    print(f"- Test samples: {len(test_data[0])}")
    print(f"- Input shape: {CONFIG['IMAGE_SIZE']}")
    print(f"- Patch size: {CONFIG['PATCH_SIZE']}")
    
    print("\nTraining Configuration:")
    print(f"- Batch size: {CONFIG['BATCH_SIZE']}")
    print(f"- Epochs: {CONFIG['EPOCHS']}")
    print(f"- Learning rate: {CONFIG['LEARNING_RATE']}")
    
    if CONFIG['MODEL_TYPE'] in ['hybrid', 'both']:
        try:
            with open('results/hybrid_model_metrics.json') as f:
                hybrid_metrics = json.load(f)
            print("\nHybrid Model Performance:")
            print(f"- Parameters: {hybrid_metrics['model_specs']['total_params']:,}")
            print(f"- Model size: {hybrid_metrics['model_specs']['model_size_mb']:.2f} MB")
            print(f"- Training time: {hybrid_metrics['training_time_hours']:.2f} hours")
            print(f"- Inference speed: {hybrid_metrics['inference_speed']['fps']:.2f} FPS")
            print(f"- Best validation mAP50: {max(hybrid_metrics['final_metrics']['val_calculate_map50']):.4f}")
        except:
            pass
    
    print("\nAll metrics and visualizations saved to disk.")

if __name__ == "__main__":
    main()