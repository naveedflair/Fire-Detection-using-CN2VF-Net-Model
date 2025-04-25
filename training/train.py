import time
import json
import matplotlib.pyplot as plt
from tensorflow.keras import optimizers
from data.dataloader import FireDataGenerator
from utils.metrics import (calculate_map50, mAP5095, 
                         precision_metric, recall_metric, 
                         f1_score_metric)
from utils.hardware import get_hardware_info, get_model_metrics, measure_inference_speed
from .losses import combined_loss
from .callbacks import get_callbacks

def train_model(model, train_data, val_data, config):
    start_time = time.time()
    
    train_generator = FireDataGenerator(
        train_data[0], train_data[1], 
        batch_size=config['BATCH_SIZE'], 
        image_size=config['IMAGE_SIZE'], 
        augment=True
    )
    val_generator = FireDataGenerator(
        val_data[0], val_data[1], 
        batch_size=config['BATCH_SIZE'], 
        image_size=config['IMAGE_SIZE'], 
        augment=False
    )
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=config['LEARNING_RATE']),
        loss=lambda y_true, y_pred: combined_loss(y_true, y_pred, alpha=0.7),
        metrics=[calculate_map50, mAP5095(), precision_metric, recall_metric, f1_score_metric]
    )
    
    callbacks = get_callbacks()
    
    history = model.fit(
        train_generator,
        epochs=config['EPOCHS'],
        validation_data=val_generator,
        callbacks=callbacks
    )
    
    training_time = time.time() - start_time
    
    metrics = {
        'training_time_seconds': training_time,
        'training_time_hours': training_time / 3600,
        'final_metrics': {k: [float(v) for v in vals] for k, vals in history.history.items()},
        'hardware': get_hardware_info(),
        'model_specs': get_model_metrics(model),
        'inference_speed': measure_inference_speed(model)
    }
    
    plot_training_metrics(history)
    
    print("\n=== Training Summary ===")
    print(f"Total training time: {metrics['training_time_hours']:.2f} hours")
    print(f"Model parameters: {metrics['model_specs']['total_params']:,}")
    print(f"Model size: {metrics['model_specs']['model_size_mb']:.2f} MB")
    print(f"Inference speed: {metrics['inference_speed']['fps']:.2f} FPS")
    print(f"Hardware used: {metrics['hardware']}")
    
    return history, model, metrics

def plot_training_metrics(history):
    plt.figure(figsize=(25, 18))
    metrics_to_plot = ['loss', 'calculate_map50', 'mAP5095', 'precision_metric', 'f1_score_metric']
    titles = ['Loss', 'mAP50', 'mAP50-95', 'Precision', 'F1-Score']
    
    for i, (metric, title) in enumerate(zip(metrics_to_plot, titles), 1):
        plt.subplot(3, 2, i)
        plt.plot(history.history[metric], label=f'Training {title}')
        plt.plot(history.history[f'val_{metric}'], label=f'Validation {title}')
        plt.title(f'Training and Validation {title}')
        plt.xlabel('Epochs')
        plt.ylabel(title)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_metrics.png', dpi=300)
    plt.show()

def evaluate_model(model, test_data, config):
    test_generator = FireDataGenerator(
        test_data[0], 
        test_data[1], 
        batch_size=config['BATCH_SIZE'], 
        image_size=config['IMAGE_SIZE'],
        augment=False
    )
    
    full_test_generator = FireDataGenerator(
        test_data[0], 
        test_data[1], 
        batch_size=len(test_data[0]), 
        image_size=config['IMAGE_SIZE']
    )
    X_test, y_test = full_test_generator[0]
    
    results = model.evaluate(test_generator)
    eval_metrics = dict(zip(model.metrics_names, results))
    
    y_pred = model.predict(X_test).flatten()
    y_true = y_test.flatten()
    
    eval_metrics.update({
        'test_samples': len(X_test),
        'input_shape': X_test[0].shape,
        'prediction_shape': y_pred.shape
    })
    
    print("\nTest Results:")
    for metric_name, value in eval_metrics.items():
        if isinstance(value, (int, float)):
            print(f"{metric_name}: {value:.4f}")
        else:
            print(f"{metric_name}: {value}")
    
    return eval_metrics, y_true, y_pred