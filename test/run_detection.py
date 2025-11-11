import requests
import os

# Path to the test image in the project root
image_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_dog.jpg")

print(f"Looking for image at: {image_path}")

# Check if image exists
if not os.path.exists(image_path):
    print(f"Error: Image not found at {image_path}")
    print("Please make sure the image exists at the specified path.")
else:
    print(f"Found image at: {image_path}")
    
    try:
        with open(image_path, 'rb') as img:
            print("Sending request to server...")
            response = requests.post(
                "http://127.0.0.1:8000/detect",
                files={"file": img}
            )
        
        print(f"Status Code: {response.status_code}")
        
        # If the response is an image, save it
        if 'image' in response.headers.get('Content-Type', ''):
            output_path = os.path.join(os.path.dirname(__file__), "detection_result.jpg")
            with open(output_path, 'wb') as f:
                f.write(response.content)
            print(f"Saved detection result to: {output_path}")
        else:
            print("Response content:", response.text)
            
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
