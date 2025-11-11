import torch
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T
from typing import List, Tuple, Dict, Any
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ObjectDetector:
    def __init__(self, threshold: float = 0.5):
        """
        Initialize the object detector with a pre-trained Faster R-CNN model.
        
        Args:
            threshold: Confidence threshold for detections (0-1)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.threshold = threshold
        
        # Load a pre-trained model for object detection
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # COCO class names (YOLOv5 uses COCO dataset by default)
        self.classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
            'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        
        # Colors for different classes
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
        
        logger.info(f"Object detector initialized on device: {self.device}")
    
    def detect(self, image_bytes: bytes) -> Tuple[Image.Image, List[Dict[str, Any]]]:
        """
        Detect objects in an image.
        
        Args:
            image_bytes: Image data in bytes
            
        Returns:
            Tuple of (annotated_image, detections)
            where detections is a list of dictionaries with keys:
            - 'label': class name
            - 'score': confidence score (0-1)
            - 'box': [x1, y1, x2, y2] coordinates
        """
        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            
            # Convert to OpenCV format (BGR)
            img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Run inference
            results = self.model(img_cv)
            
            # Process results
            detections = []
            
            # Get detections
            pred = results.xyxy[0].cpu().numpy()  # Get predictions
            
            for *xyxy, conf, cls in pred:
                if conf < self.threshold:
                    continue
                    
                x1, y1, x2, y2 = map(int, xyxy)
                class_id = int(cls)
                class_name = self.classes[class_id]
                
                # Add to detections
                detections.append({
                    'label': class_name,
                    'score': float(conf),
                    'box': [x1, y1, x2, y2]
                })
            
            # Draw bounding boxes on the image
            annotated_img = self.draw_boxes(image, detections)
            
            return annotated_img, detections
            
        except Exception as e:
            logger.error(f"Error during detection: {str(e)}")
            raise
    
    def draw_boxes(self, image: Image.Image, detections: List[Dict[str, Any]]) -> Image.Image:
        """
        Draw bounding boxes and labels on the image.
        
        Args:
            image: PIL Image
            detections: List of detection dictionaries
            
        Returns:
            Annotated PIL Image
        """
        draw = ImageDraw.Draw(image)
        
        for det in detections:
            label = det['label']
            score = det['score']
            box = det['box']
            
            # Get color for this class
            class_id = self.classes.index(label)
            color = tuple(map(int, self.colors[class_id % len(self.colors)]))
            
            # Draw rectangle
            draw.rectangle(box, outline=color, width=2)
            
            # Create label with score
            label_text = f"{label} {score:.2f}"
            
            # Calculate text size
            try:
                font = ImageFont.truetype("arial.ttf", 12)
            except IOError:
                font = ImageFont.load_default()
            
            # Draw text background
            text_bbox = draw.textbbox((box[0], box[1]), label_text, font=font)
            draw.rectangle([box[0], box[1] - 15, text_bbox[2] + 5, box[1]], fill=color)
            
            # Draw text
            draw.text((box[0] + 2, box[1] - 15), label_text, fill=(255, 255, 255), font=font)
        
        return image

# Global instance of the detector
detector = ObjectDetector(threshold=0.5)

def detect_objects(image_bytes: bytes) -> Tuple[Image.Image, List[Dict[str, Any]]]:
    """
    Detect objects in an image and return the annotated image and detections.
    
    Args:
        image_bytes: Image data in bytes
        
    Returns:
        Tuple of (annotated_image, detections)
    """
    return detector.detect(image_bytes)
