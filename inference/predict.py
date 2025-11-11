import os
import logging
import numpy as np
from typing import Dict, Any, List, Tuple
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

class BadImageException(Exception):
    """Exception raised for invalid or corrupted images"""
    pass

class Predictor:
    """
    Handles model loading, preprocessing, and inference for CIFAR-10 image classification.
    """
    def __init__(
        self,
        model_checkpoint: str = 'models/cifar10_model.h5',
        model_type: str = 'cifar10'
    ):
        """
        Initialize the predictor with the specified model.
        
        Args:
            model_checkpoint: Path to the model file
            model_type: Type of model (currently only 'cifar10' is supported)
        """
        self.model_type = model_type.lower()
        self.model_path = model_checkpoint
        
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
        
        # CIFAR-10 class names
        self.class_names = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
        
        # Load model
        self.model = self._load_model(model_checkpoint)
        
        logger.info(f"Predictor initialized with {model_type} model on device: {self.device}")
    
    def _create_model_architecture(self) -> tf.keras.Model:
        """Create the CIFAR-10 model architecture that matches the saved weights."""
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
    
    def _load_model(self, checkpoint_path: str) -> tf.keras.Model:
        """Load the model from the specified path."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Model file not found at {checkpoint_path}")
        
        # First try to load the model directly
        try:
            logger.info(f"Attempting to load model directly from {checkpoint_path}")
            model = load_model(checkpoint_path, compile=False)
            logger.info("Model loaded successfully with load_model()")
            return model
        except Exception as e:
            logger.warning(f"Direct model loading failed: {e}")
        
        # If direct loading fails, create the model and load weights
        try:
            logger.info("Creating model and loading weights...")
            model = self._create_model_architecture()
            
            # Load weights directly (this worked in SimplePredictor)
            model.load_weights(checkpoint_path)
            logger.info("Successfully loaded model weights")
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise ValueError(f"Could not load model: {e}")
    
    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """
        Preprocess the image for the model.
        
        Args:
            image: PIL Image to preprocess
            
        Returns:
            Preprocessed image as numpy array
        """
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
            raise BadImageException(f"Could not preprocess image: {e}")
    
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
            raise BadImageException("Could not identify image file")
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise
        
    def _load_labels(self, labels_path: str) -> List[str]:
        """Load class labels from file."""
        try:
            if not os.path.exists(labels_path):
                logger.warning(f"Labels file not found at {labels_path}. Using default ImageNet labels.")
                return self._get_imagenet_labels()
                
            with open(labels_path, 'r', encoding='utf-8') as f:
                return [line.strip() for line in f.readlines() if line.strip()]
                
        except Exception as e:
            logger.error(f"Error loading labels: {str(e)}")
            return self._get_imagenet_labels()
    
    def _get_imagenet_labels(self) -> List[str]:
        """Get default ImageNet class labels."""
        import json
        import urllib.request
        
        try:
            url = 'https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json'
            with urllib.request.urlopen(url) as f:
                class_idx = json.load(f)
                return [class_idx[str(i)][1] for i in range(len(class_idx))]
        except Exception as e:
            logger.error(f"Failed to load ImageNet labels: {str(e)}")
            return [f"class_{i}" for i in range(1000)]
    
    def _load_model(self, checkpoint_path: str) -> Any:
        """Load the appropriate model based on model type."""
        try:
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Model file not found at {checkpoint_path}")
                
            if self.model_type == 'cifar10':
                logger.info(f"Loading CIFAR-10 model from {checkpoint_path}")
                
                # Try different loading methods for compatibility
                try:
                    # Try loading with custom_objects if needed
                    model = load_model(checkpoint_path, compile=False)
                except Exception as e:
                    logger.warning(f"Standard load failed, trying with custom objects: {str(e)}")
                    try:
                        from tensorflow.keras.layers import InputLayer
                        from tensorflow.keras.models import Sequential
                        from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
                        
                        # Define a compatible model architecture
                        model = Sequential([
                            InputLayer(input_shape=(32, 32, 3)),
                            Conv2D(32, (3, 3), activation='relu', padding='same'),
                            MaxPooling2D((2, 2)),
                            Conv2D(64, (3, 3), activation='relu', padding='same'),
                            MaxPooling2D((2, 2)),
                            Flatten(),
                            Dense(64, activation='relu'),
                            Dropout(0.5),
                            Dense(10, activation='softmax')
                        ])
                        
                        # Load only the weights
                        model.load_weights(checkpoint_path)
                        logger.info("Successfully loaded model weights with custom architecture")
                    except Exception as e2:
                        logger.error(f"Failed to load model with custom architecture: {str(e2)}")
                        raise
                
                model.make_predict_function()  # For better performance
                return model
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
                
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            # Provide more detailed error information
            import traceback
            logger.error(f"Error details: {traceback.format_exc()}")
            raise
    
    def _load_image_bytes(self, data: bytes) -> Image.Image:
        """Load image from bytes and validate."""
        try:
            img = Image.open(io.BytesIO(data))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            return img
        except UnidentifiedImageError:
            raise BadImageException("Could not identify image file")
        except Exception as e:
            raise BadImageException(f"Error processing image: {str(e)}")
    
    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """
        Preprocess image for CIFAR-10 model.
        
        Args:
            image: PIL Image
            
        Returns:
            Preprocessed image as numpy array
        """
        # Resize to 32x32 for CIFAR-10
        image = image.resize((32, 32))
        
        # Convert to numpy array and normalize
        image_array = np.array(image, dtype=np.float32) / 255.0
        
        # If image is grayscale, convert to RGB by repeating the channel
        if len(image_array.shape) == 2:
            image_array = np.stack((image_array,) * 3, axis=-1)
            
        # If image has alpha channel, remove it
        if image_array.shape[2] == 4:
            image_array = image_array[:, :, :3]
            
        # Add batch dimension
        return np.expand_dims(image_array, axis=0)
    
    def predict(self, image_data: bytes) -> Dict[str, Any]:
        """
        Run inference on an image.
        
        Args:
            image_data: Binary image data
            
        Returns:
            Dictionary containing:
            - label: Predicted class label
            - confidence: Prediction confidence (0-1)
            - all_predictions: List of all class probabilities
        """
        try:
            # Load image
            image = Image.open(io.BytesIO(image_data))
            
            # Preprocess image
            processed_image = self.preprocess_image(image)
            
            # Run inference
            predictions = self.model.predict(processed_image, verbose=0)
            
            # Get top prediction
            predicted_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_idx])
            predicted_label = self.cifar10_labels[predicted_idx]
            
            # Get all predictions
            all_predictions = [
                {"label": label, "score": float(score)}
                for label, score in zip(self.cifar10_labels, predictions[0])
            ]
            
            return {
                "label": predicted_label,
                "confidence": confidence,
                "all_predictions": all_predictions
            }
            
        except UnidentifiedImageError:
            raise BadImageException("Could not identify image file")
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise
