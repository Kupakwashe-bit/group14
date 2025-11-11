import os
import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Dropout

def print_model_summary(model_path):
    print(f"\n{'='*50}")
    print(f"Inspecting model: {model_path}")
    
    # Check if file exists
    if not os.path.exists(model_path):
        print(f"Error: File not found: {model_path}")
        return
    
    # Try to load the model directly
    print("\nAttempting to load model directly...")
    try:
        model = load_model(model_path, compile=False)
        print("\nModel loaded successfully!")
        model.summary()
        return
    except Exception as e:
        print(f"\nDirect model loading failed: {e}")
    
    # If direct loading fails, try to inspect the HDF5 file
    print("\nInspecting HDF5 file structure...")
    try:
        with h5py.File(model_path, 'r') as f:
            print("\nTop-level keys:", list(f.keys()))
            
            # Check for model configuration
            if 'model_weights' in f:
                print("\nModel weights structure:")
                def print_weights(name, obj):
                    if isinstance(obj, h5py.Group):
                        print(f"  Group: {name}")
                        if obj.attrs:
                            print("    Attributes:", dict(obj.attrs))
                    elif isinstance(obj, h5py.Dataset):
                        print(f"  Dataset: {name}, shape: {obj.shape}, dtype: {obj.dtype}")
                
                f.visititems(print_weights)
            
            # Check for model configuration
            if 'model_config' in f.attrs:
                print("\nModel configuration:")
                print(f.attrs['model_config'])
            
            # Check for training configuration
            if 'training_config' in f.attrs:
                print("\nTraining configuration:")
                print(f.attrs['training_config'])
                
    except Exception as e:
        print(f"Error inspecting HDF5 file: {e}")
    
    # Try to create a model with the correct architecture and load weights
    print("\nAttempting to create model with matching architecture...")
    try:
        # Try different architectures
        architectures = [
            # Architecture 1
            [
                Input(shape=(32, 32, 3)),
                Conv2D(32, (3, 3), activation='relu', padding='same', name='conv2d'),
                MaxPooling2D((2, 2), name='max_pooling2d'),
                Dropout(0.25, name='dropout'),
                Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2d_1'),
                MaxPooling2D((2, 2), name='max_pooling2d_1'),
                Flatten(name='flatten'),
                Dense(64, activation='relu', name='dense'),
                Dense(10, activation='softmax', name='dense_1')
            ],
            # Architecture 2
            [
                Input(shape=(32, 32, 3)),
                Conv2D(32, (3, 3), activation='relu', padding='same', name='conv2d'),
                Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2d_1'),
                MaxPooling2D((2, 2), name='max_pooling2d'),
                Dropout(0.25, name='dropout'),
                Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2d_2'),
                MaxPooling2D((2, 2), name='max_pooling2d_1'),
                Flatten(name='flatten'),
                Dense(64, activation='relu', name='dense'),
                Dense(10, activation='softmax', name='dense_1')
            ]
        ]
        
        for i, layers in enumerate(architectures, 1):
            print(f"\nTrying architecture {i}...")
            try:
                # Create model
                model = tf.keras.Sequential(layers)
                model.compile(optimizer='adam',
                            loss='sparse_categorical_crossentropy',
                            metrics=['accuracy'])
                
                # Try to load weights
                model.load_weights(model_path, by_name=True, skip_mismatch=True)
                print("Successfully loaded weights with architecture:")
                model.summary()
                return
                
            except Exception as e:
                print(f"Failed with architecture {i}: {e}")
        
    except Exception as e:
        print(f"Error creating model: {e}")
    
    print("\nCould not determine the correct model architecture.")

if __name__ == "__main__":
    model_path = 'models/cifar10_model.h5'
    print_model_summary(model_path)
