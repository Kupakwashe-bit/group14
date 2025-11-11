# 🖼️ Image Classification API with ResNet50 and FastAPI

A production-ready image classification service using PyTorch's ResNet50 model, served via FastAPI. The API can be used as-is with the default ImageNet weights or fine-tuned on custom datasets.

## ✨ Features

- 🚀 FastAPI for high-performance API serving
- 🧠 Pre-trained ResNet50 model with optional fine-tuning
- 📦 Containerized with Docker for easy deployment
- 📊 Built-in model training and evaluation
- 📝 Comprehensive logging and error handling
- 🏗️ Modular and extensible architecture

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- pip
- Docker (optional, for containerized deployment)

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the API

```bash
uvicorn api.main:app --reload
```

The API will be available at `http://localhost:8000`

### Making Predictions

#### Using cURL:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/your/image.jpg"
```

#### Using Python:

```python
import requests

url = "http://localhost:8000/predict"
files = {"file": open("path/to/your/image.jpg", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

### Example Response

```json
{
  "predictions": [
    {"label": "golden_retriever", "confidence": 0.8765},
    {"label": "labrador_retriever", "confidence": 0.0987},
    ...
  ],
  "meta": {
    "model": "resnet50-finetuned",
    "device": "cuda",
    "topk": 5
  }
}
```

## 🐳 Docker Deployment

### Build the Docker Image

```bash
docker build -t image-classifier .
```

### Run the Container

```bash
docker run -p 8000:8000 image-classifier
```

### With Custom Model and Labels

```bash
docker run -p 8000:8000 \
  -v /path/to/your/models:/app/models \
  -e MODEL_CHECKPOINT=/app/models/your_model.pt \
  -e LABELS_PATH=/app/models/your_labels.txt \
  image-classifier
```

## 🏗️ Training a Custom Model

### Dataset Structure

Organize your dataset as follows:

```
data/
└── train/
    ├── class1/
    │   ├── img1.jpg
    │   └── ...
    ├── class2/
    │   ├── img1.jpg
    │   └── ...
    └── ...
```

### Start Training

```bash
python models/train.py \
  --data-dir data \
  --epochs 20 \
  --batch-size 32 \
  --learning-rate 0.001 \
  --pretrained \
  --fine-tune
```

### Training Arguments

- `--data-dir`: Path to the training data directory
- `--epochs`: Number of training epochs (default: 10)
- `--batch-size`: Batch size for training (default: 32)
- `--learning-rate`: Learning rate (default: 0.001)
- `--pretrained`: Use pretrained weights (default: False)
- `--fine-tune`: Fine-tune all layers (default: False, only train classifier)
- `--output-dir`: Directory to save model checkpoints (default: 'models')

## 🛠️ API Endpoints

### `POST /predict`

Classify an uploaded image.

**Request:**
- `file`: Image file to classify (JPEG, PNG, etc.)

**Response:**
- `predictions`: List of predicted classes with confidence scores
- `meta`: Metadata including model and device information

### `GET /health`

Check the health status of the API.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda"
}
```

## 📚 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_CHECKPOINT` | `models/resnet50_ft.pt` | Path to the model checkpoint file |
| `LABELS_PATH` | `models/labels.txt` | Path to the class labels file |
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) |

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For any questions or feedback, please open an issue on GitHub.
