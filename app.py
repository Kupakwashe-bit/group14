import os
from flask import Flask, render_template, request, jsonify
from PIL import Image
import io
import base64
from predictor import Predictor

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize the predictor
predictor = Predictor(model_checkpoint='models/cifar10_model.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Read the image file
        image_data = file.read()
        
        # Make prediction
        result = predictor.predict(image_data)
        
        # Convert image to base64 for display
        img = Image.open(io.BytesIO(image_data)).convert('RGB')
        img.thumbnail((300, 300))  # Resize for display
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return jsonify({
            'success': True,
            'image': f"data:image/jpeg;base64,{img_str}",
            'prediction': result['class'],
            'confidence': f"{result['confidence']:.2%}",
            'all_predictions': result['all_predictions']
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
