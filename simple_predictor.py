import os
import logging
import numpy as np
from typing import Dict, Any
from PIL import Image, ImageFile, UnidentifiedImageError
import io
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer
from tensorflow.keras.optimizers import Adam

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

class SimplePredictor:
    """A simplified predictor for CIFAR-10 image classification."""
    
    def __init__(self, model_path: str = 'models/cifar10_model.h5'):
        """Initialize the predictor with the specified model."""
        self.model_path = model_path
        self.class_names = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
        
        # Set up device
        self.device = 'cuda' if tf.config.list_physical_devices('GPU') else 'cpu'
        if self.device == 'cuda':
            try:
                gpus = tf.config.experimental.list_physical_devices('GPU')
                if gpus:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                logger.warning(f"Could not set up GPU: {e}")
                self.device = 'cpu'
        
        # Load the model
        self.model = self._load_model()
        logger.info(f"Predictor initialized on device: {self.device}")
    
    def _create_model(self) -> tf.keras.Model:
        """Create the CIFAR-10 model architecture."""
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
        model.compile(optimizer=Adam(),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
        
        return model
    
    def _load_model(self) -> tf.keras.Model:
        """Load the model from the specified path."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
        
        # First try to load the model directly
        try:
            logger.info(f"Attempting to load model directly from {self.model_path}")
            model = load_model(self.model_path, compile=False)
            logger.info("Model loaded successfully with load_model()")
            return model
        except Exception as e:
            logger.warning(f"Direct model loading failed: {e}")
        
        # If direct loading fails, create the model and load weights
        try:
            logger.info("Creating model and loading weights...")
            model = self._create_model()
            
            # Load weights
            model.load_weights(self.model_path)
            logger.info("Successfully loaded model weights")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise ValueError(f"Could not load model: {e}")
    
    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """Preprocess the image for the model."""
        try:
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            # Resize to 32x32 for CIFAR-10
            image = image.resize((32, 32))
            
            # Convert to numpy array and normalize
            image_array = np.array(image, dtype=np.float32) / 255.0
            
            # Add batch dimension
            return np.expand_dims(image_array, axis=0)
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            raise ValueError(f"Could not preprocess image: {e}")
    
    def predict(self, image_data: bytes) -> Dict[str, Any]:
        """
        Make a prediction on the input image.
        
        Args:
            image_data: Binary image data
            
        Returns:
            Dictionary containing prediction results
        """
        try:
            # Load and preprocess image
            image = Image.open(io.BytesIO(image_data))
            processed_image = self.preprocess_image(image)
            
            # Make prediction
            predictions = self.model.predict(processed_image, verbose=0)
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_idx])
            
            # Prepare results
            result = {
                'class': self.class_names[predicted_class_idx],
                'class_idx': int(predicted_class_idx),
                'confidence': confidence,
                'all_predictions': [
                    {'class': name, 'score': float(score)} 
                    for name, score in zip(self.class_names, predictions[0])
                ]
            }
            
            return result
            
        except UnidentifiedImageError:
            raise ValueError("Could not identify image file")
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise

# Example usage
if __name__ == "__main__":
    import requests
    
    # Initialize predictor
    predictor = SimplePredictor()
    
    # Test with a sample image
    image_url = "https://images.unsplash.com/photo-1587300003388-59208cc962cb?w=500&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8N3x8ZG9nfGVufDB8fDB8fHww"
    
    try:
        # Download the image
        print(f"Downloading test image from {image_url}...")
        response = requests.get(image_url)
        response.raise_for_status()
        
        # Make prediction
        print("Making prediction...")
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
        print(f"\nError: {e}")
