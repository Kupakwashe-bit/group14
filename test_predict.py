import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer
from tensorflow.keras.optimizers import Adam

def create_model():
    """Create a CNN model with the correct architecture."""
    model = Sequential([
        InputLayer(input_shape=(32, 32, 3)),
        Conv2D(32, (3, 3), activation='relu', padding='same', name='conv2d'),
        MaxPooling2D((2, 2), name='max_pooling2d'),
        Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2d_1'),
        MaxPooling2D((2, 2), name='max_pooling2d_1'),
        Flatten(name='flatten'),
        Dense(64, activation='relu', name='dense'),
        Dropout(0.5, name='dropout'),
        Dense(10, activation='softmax', name='dense_1')
    ])
    
    # Compile the model
    model.compile(optimizer=Adam(),
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    
    return model

def load_and_test_model():
    model_path = 'models/cifar10_model.h5'
    
    # Create the model with the correct architecture
    print("Creating model with the correct architecture...")
    model = create_model()
    model.summary()
    
    try:
        # Try to load the weights
        print("\nLoading weights from:", model_path)
        model.load_weights(model_path)
        print("Weights loaded successfully!")
        
        # Test the model with a random image
        print("\nTesting the model with a random image...")
        test_image = np.random.random((1, 32, 32, 3))  # Random test image
        predictions = model.predict(test_image)
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        
        class_names = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
        
        print(f"\nPrediction results:")
        print(f"Predicted class: {class_names[predicted_class]} (confidence: {confidence:.2%})")
        
        # Print top 3 predictions
        print("\nTop 3 predictions:")
        top_indices = np.argsort(predictions[0])[-3:][::-1]
        for idx in top_indices:
            print(f"- {class_names[idx]}: {predictions[0][idx]:.2%}")
            
    except Exception as e:
        print(f"\nError loading weights: {str(e)}")
        print("\nPossible solutions:")
        print("1. Check if the model architecture matches the saved weights")
        print("2. Try using the same version of Keras/TensorFlow that was used to save the model")
        print("3. If possible, retrain and save the model with the current version")

if __name__ == "__main__":
    load_and_test_model()
