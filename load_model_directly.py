import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

def main():
    model_path = 'models/cifar10_model.h5'
    
    print(f"TensorFlow version: {tf.__version__}")
    try:
        import keras
        print(f"Keras version: {keras.__version__}")
    except (ImportError, AttributeError):
        print("Standalone Keras not found, using TensorFlow's Keras")
    
    # Try to load the model directly
    try:
        print(f"\nAttempting to load model from: {model_path}")
        model = load_model(model_path, compile=False)
        print("Model loaded successfully!")
        
        # Print model summary
        model.summary()
        
        # Test prediction with random data
        print("\nTesting prediction with random data...")
        test_input = np.random.random((1, 32, 32, 3))
        prediction = model.predict(test_input)
        
        class_names = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
        
        predicted_class = np.argmax(prediction[0])
        confidence = np.max(prediction[0])
        
        print(f"\nPrediction results:")
        print(f"Predicted class: {class_names[predicted_class]} (confidence: {confidence:.2%})")
        
        # Print top 3 predictions
        print("\nTop 3 predictions:")
        top_indices = np.argsort(prediction[0])[-3:][::-1]
        for idx in top_indices:
            print(f"- {class_names[idx]}: {prediction[0][idx]:.2%}")
            
    except Exception as e:
        print(f"\nError loading model: {str(e)}")
        print("\nTroubleshooting steps:")
        print("1. Check if the model file exists and is not corrupted")
        print("2. Try installing the exact version of TensorFlow/Keras used to save the model")
        print("3. If you have access to the training code, try saving the model in a different format")
        print("4. Consider retraining the model with the current version of TensorFlow")
        print("\nFull error details:")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
