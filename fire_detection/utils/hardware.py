import os
import psutil
import tensorflow as tf

def get_model_metrics(model):
    model.save('temp_model.h5')
    model_size_mb = os.path.getsize('temp_model.h5') / (1024 * 1024)
    os.remove('temp_model.h5')
    
    trainable_params = np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_params = np.sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
    
    return {
        'total_params': trainable_params + non_trainable_params,
        'trainable_params': trainable_params,
        'non_trainable_params': non_trainable_params,
        'model_size_mb': model_size_mb
    }

def measure_inference_speed(model, input_shape=(1, 416, 416, 3), n_runs=100):
    dummy_input = np.random.rand(*input_shape).astype(np.float32)
    
    for _ in range(10):
        model.predict(dummy_input)
    
    start_time = time.time()
    for _ in range(n_runs):
        model.predict(dummy_input)
    end_time = time.time()
    
    avg_time_ms = (end_time - start_time) * 1000 / n_runs
    fps = 1000 / avg_time_ms
    
    return {
        'inference_time_ms': avg_time_ms,
        'fps': fps
    }

def get_hardware_info():
    gpu_info = tf.config.list_physical_devices('GPU')
    cpu_info = {
        'cores': psutil.cpu_count(logical=False),
        'threads': psutil.cpu_count(logical=True)
    }
    ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    
    return {
        'GPU': [gpu.name for gpu in gpu_info] if gpu_info else 'None',
        'CPU': cpu_info,
        'RAM_GB': ram_gb
    }