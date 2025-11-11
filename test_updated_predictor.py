import os
import numpy as np
from PIL import Image
import requests
from io import BytesIO
from predictor import Predictor

def test_with_sample_image():
    """Test the predictor with a sample image from the internet."""
    # Initialize the predictor
    print("Initializing predictor...")
    predictor = Predictor(model_checkpoint='models/cifar10_model.h5')
    
    # URL of a sample image (a dog)
    image_url = "https://images.unsplash.com/photo-1587300003388-59208cc962cb?w=500&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8N3x8ZG9nfGVufDB8fDB8fHww"
    
    try:
        # Download the image
        print(f"\nDownloading test image from {image_url}...")
        response = requests.get(image_url)
        response.raise_for_status()
        
        # Make prediction
        print("\nMaking prediction...")
        result = predictor.predict(response.content)
        
        # Print results
        print("\nPrediction Results:")
        print(f"Predicted class: {result['class']}")
        print(f"Confidence: {result['confidence']:.2%}")
        
        # Print top 3 predictions
        print("\nTop 3 predictions:")
        for i, pred in enumerate(sorted(result['all_predictions'], 
                                      key=lambda x: x['score'], 
                                      reverse=True)[:3], 1):
            print(f"{i}. {pred['class']}: {pred['score']:.2%}")
            
    except Exception as e:
        print(f"\nError during prediction: {str(e)}")
        raise

def test_with_local_image(image_path):
    """Test the predictor with a local image file."""
    # Initialize the predictor
    print("Initializing predictor...")
    predictor = Predictor(model_checkpoint='models/cifar10_model.h5')
    
    try:
        # Read the image file
        print(f"\nReading image from {image_path}...")
        with open(image_path, 'rb') as f:
            image_data = f.read()
        
        # Make prediction
        print("\nMaking prediction...")
        result = predictor.predict(image_data)
        
        # Print results
        print("\nPrediction Results:")
        print(f"Predicted class: {result['class']}")
        print(f"Confidence: {result['confidence']:.2%}")
        
        # Print top 3 predictions
        print("\nTop 3 predictions:")
        for i, pred in enumerate(sorted(result['all_predictions'], 
                                      key=lambda x: x['score'], 
                                      reverse=True)[:3], 1):
            print(f"{i}. {pred['class']}: {pred['score']:.2%}")
            
    except Exception as e:
        print(f"\nError during prediction: {str(e)}")
        raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test the CIFAR-10 predictor')
    parser.add_argument('--local-image', type=str, help='Path to a local image file to test')
    args = parser.parse_args()
    
    if args.local_image:
        test_with_local_image(args.local_image)
    else:
        test_with_sample_image()
