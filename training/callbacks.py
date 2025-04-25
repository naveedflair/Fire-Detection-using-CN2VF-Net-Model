from tensorflow.keras.callbacks import (ModelCheckpoint, 
                                      EarlyStopping, 
                                      ReduceLROnPlateau)

def get_callbacks():
    checkpoint_callback = ModelCheckpoint(
        'best_model.keras', 
        monitor='val_mAP5095',
        mode='max', 
        save_best_only=True, 
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_mAP5095',
        factor=0.5, 
        patience=5, 
        min_lr=1e-6, 
        mode='max', 
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_mAP5095',
        patience=10,
        mode='max',
        restore_best_weights=True,
        verbose=1
    )
    
    return [checkpoint_callback, reduce_lr, early_stopping]