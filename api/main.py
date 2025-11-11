import os
import io
import logging
import uvicorn
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Response, status, Depends
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse, StreamingResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from typing import Optional, Dict, Any
import json

# Import after setting up environment
from inference.predict import Predictor, BadImageException
from inference.detect import detect_objects

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

# Get the base directory
BASE_DIR = Path(__file__).parent.parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

# Ensure the static and templates directories exist
STATIC_DIR.mkdir(exist_ok=True, parents=True)
TEMPLATES_DIR.mkdir(exist_ok=True, parents=True)

# Initialize FastAPI app
app = FastAPI(
    title="Computer Vision API",
    description="A REST API for image classification and object detection",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Set up templates
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Mount static files
try:
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
except Exception as e:
    logger.error(f"Failed to mount static files: {e}")
    raise

# Initialize predictors
predictor = None
cifar_predictor = None

# Initialize predictors in startup event
@app.on_event("startup")
async def startup_event():
    global predictor, cifar_predictor
    try:
        # Initialize main predictor (original model)
        predictor = Predictor()
        
        # Initialize CIFAR-10 predictor
        cifar_predictor = Predictor(
            model_checkpoint=str(BASE_DIR / 'models' / 'cifar10_model.h5'),
            model_type='cifar10'
        )
        
        logger.info("✅ Predictors initialized successfully")
        
        # Verify static files are accessible
        if not (STATIC_DIR / "index.html").exists():
            logger.warning(f"index.html not found in {STATIC_DIR}")
        else:
            logger.info(f"✅ Static files found in {STATIC_DIR}")
            
    except Exception as e:
        logger.error(f"❌ Failed to initialize predictors: {e}")
        predictor = None
        cifar_predictor = None

# Serve the main page
# Exception handlers
@app.exception_handler(404)
async def not_found_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(
        status_code=404,
        content={"detail": "The requested resource was not found"},
    )

@app.exception_handler(500)
async def server_error_exception_handler(request: Request, exc: Exception):
    logger.error(f"Server error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )

# Root endpoint
@app.get("/", response_class=HTMLResponse)
async def read_root():
    try:
        index_path = STATIC_DIR / "index.html"
        if not index_path.exists():
            logger.error(f"index.html not found in {STATIC_DIR}")
            raise HTTPException(
                status_code=500,
                detail=f"Index file not found in {STATIC_DIR}"
            )
        return FileResponse(index_path)
    except Exception as e:
        logger.error(f"Error serving index.html: {e}")
        return HTMLResponse(
            content="""
            <h1>Welcome to Computer Vision API</h1>
            <p>The web interface is not available. Please check the server logs.</p>
            <p>You can still use the API endpoints:</p>
            <ul>
                <li><a href="/api/docs">API Documentation</a></li>
                <li>POST /predict - For image classification</li>
                <li>POST /detect - For object detection</li>
            </ul>
            """,
            status_code=200
        )

# Serve static files
@app.get("/{file_path:path}")
async def serve_static(file_path: str):
    try:
        # Security: Prevent directory traversal
        if ".." in file_path or file_path.startswith("/"):
            raise HTTPException(status_code=400, detail="Invalid file path")
            
        static_file = STATIC_DIR / file_path
        
        # If the path is a directory, look for index.html
        if static_file.is_dir():
            index_file = static_file / "index.html"
            if index_file.exists():
                return FileResponse(index_file)
            raise HTTPException(status_code=404, detail="Directory index not found")
            
        # If it's a file, serve it
        if static_file.exists() and static_file.is_file():
            return FileResponse(static_file)
            
        # Check if it's an HTML file without extension
        html_file = static_file.with_suffix('.html')
        if html_file.exists():
            return FileResponse(html_file)
            
        logger.warning(f"File not found: {file_path}")
        raise HTTPException(status_code=404, detail="File not found")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving static file {file_path}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Add a simple health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": predictor is not None}

@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    """
    Handle image upload and return classification results using the default model.
    
    Args:
        file (UploadFile): The image file to classify
        
    Returns:
        JSONResponse: Contains predictions and metadata
    """
    if not predictor:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Predictor not initialized"
        )
        
    try:
        # Read image data
        contents = await file.read()
        
        # Get prediction
        label, confidence = predictor.predict(contents)
        
        return {
            "model": predictor.model_name,
            "prediction": label,
            "confidence": confidence,
            "status": "success"
        }
        
    except BadImageException as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing image: {str(e)}"
        )

@app.post("/api/predict/cifar10")
async def predict_cifar10(file: UploadFile = File(...)):
    """
    Handle image upload and return CIFAR-10 classification results.
    
    Args:
        file (UploadFile): The image file to classify (will be resized to 32x32)
        
    Returns:
        JSONResponse: Contains CIFAR-10 predictions and confidence scores
    """
    if not cifar_predictor:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="CIFAR-10 predictor not initialized"
        )
    
    # Check file type
    content_type = file.content_type
    if content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(
            status_code=400,
            detail=f"File type {content_type} not supported. Please upload a JPG or PNG image."
        )
    
    try:
        # Read image data
        contents = await file.read()
        
        # Get prediction
        result = cifar_predictor.predict(contents)
        
        return {
            "model": "cifar10",
            "prediction": result["label"],
            "confidence": result["confidence"],
            "all_predictions": result["all_predictions"],
            "status": "success"
        }
        
    except BadImageException as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"CIFAR-10 prediction error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing image: {str(e)}"
        )

@app.post("/detect")
async def detect_objects_endpoint(file: UploadFile = File(...)):
    """
    Detect objects in an image and return the image with bounding boxes.
    
    Args:
        file (UploadFile): The image file to process
        
    Returns:
        StreamingResponse: Image with bounding boxes drawn
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    logger.info(f"Processing detection for image: {file.filename}")
    contents = await file.read()
    
    try:
        # Detect objects and get annotated image
        annotated_img, detections = detect_objects(contents)
        
        # Convert PIL Image to bytes
        img_byte_arr = io.BytesIO()
        annotated_img.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()
        
        # Return the image with bounding boxes
        return Response(content=img_byte_arr, media_type="image/jpeg")
        
    except Exception as e:
        logger.error(f"Detection error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during object detection: {str(e)}")

@app.post("/detect_json")
async def detect_objects_json(file: UploadFile = File(...)):
    """
    Detect objects in an image and return the detections as JSON.
    
    Args:
        file (UploadFile): The image file to process
        
    Returns:
        JSONResponse: List of detected objects with bounding boxes and confidence
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    logger.info(f"Processing detection for image: {file.filename}")
    contents = await file.read()
    
    try:
        # Detect objects
        _, detections = detect_objects(contents)
        
        # Convert numpy arrays to lists for JSON serialization
        for det in detections:
            det['box'] = [int(x) for x in det['box']]
        
        return JSONResponse({
            "detections": detections,
            "count": len(detections)
        })
        
    except Exception as e:
        logger.error(f"Detection error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during object detection: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": predictor is not None,
        "device": str(predictor.device) if predictor else None
    }
