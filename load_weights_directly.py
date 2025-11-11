import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer

def inspect_weights(model_path):
    print(f"Inspecting weights in: {model_path}")
    
    # Open the HDF5 file
    with h5py.File(model_path, 'r') as f:
        # Get the model weights
        model_weights = f['model_weights']
        
        # Print all layer names and their weights
        def print_weights(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"\nLayer: {name}")
                print(f"  Shape: {obj.shape}")
                print(f"  Dtype: {obj.dtype}")
                print(f"  Size: {obj.size} elements")
        
        print("\nModel weights structure:")
        model_weights.visititems(print_weights)

def create_and_load_model():
    print("\nCreating model with correct architecture...")
    
    # Create the model architecture based on our inspection
    model = Sequential([
        InputLayer(input_shape=(32, 32, 3), name='input_layer'),
        Conv2D(32, (3, 3), activation='relu', padding='valid', name='conv2d'),
        MaxPooling2D((2, 2), name='max_pooling2d'),
        Conv2D(64, (3, 3), activation='relu', padding='valid', name='conv2d_1'),
        MaxPooling2D((2, 2), name='max_pooling2d_1'),
        Conv2D(64, (3, 3), activation='relu', padding='valid', name='conv2d_2'),
        Flatten(name='flatten'),
        Dense(64, activation='relu', name='dense'),
        Dropout(0.5, name='dropout'),
        Dense(10, activation='softmax', name='dense_1')
    ])
    
    # Compile the model
    model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    
    # Print model summary
    print("\nModel summary:")
    model.summary()
    
    # Try to load weights
    print("\nAttempting to load weights...")
    try:
        # Load weights by name, skipping mismatched layers
        model.load_weights('models/cifar10_model.h5', by_name=True, skip_mismatch=True)
        print("Weights loaded successfully!")
        
        # Test with random data
        print("\nTesting with random data...")
        test_input = np.random.random((1, 32, 32, 3)).astype('float32')
        prediction = model.predict(test_input)
        print(f"Prediction shape: {prediction.shape}")
        print(f"Sample prediction: {prediction[0][:5]}...")
        
        return model
        
    except Exception as e:
        print(f"Error loading weights: {e}")
        raise

if __name__ == "__main__":
    model_path = 'models/cifar10_model.h5'
    
    # First inspect the weights
    inspect_weights(model_path)
    
    # Then try to create and load the model
    create_and_load_model()
