import requests

def test_predict():
    url = "http://127.0.0.1:8000/predict"
    files = {'file': open('test_dog.jpg', 'rb')}
    headers = {'accept': 'application/json'}
    
    try:
        response = requests.post(url, files=files, headers=headers)
        print(f"Status Code: {response.status_code}")
        print("Response:", response.json())
    except Exception as e:
        print(f"Error: {e}")
    finally:
        files['file'].close()

if __name__ == "__main__":
    test_predict()
