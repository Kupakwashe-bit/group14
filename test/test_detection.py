import os
import requests

# Path to a test image (update this to point to an actual image on your system)
image_path = os.path.join(os.path.expanduser("~"), "Pictures", "test_image.jpg")

try:
    with open(image_path, 'rb') as img:
        response = requests.post(
            "http://127.0.0.1:8000/detect",
            files={"file": img}
        )
    print(f"Status Code: {response.status_code}")
    print("Response:", response.text)
except Exception as e:
    print(f"Error: {e}")