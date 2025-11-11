import requests
import sys
import os
from pathlib import Path

def test_server(base_url="http://localhost:8000"):
    """Test the server endpoints"""
    print("Testing server endpoints...")
    
    # Test root endpoint
    try:
        response = requests.get(f"{base_url}/")
        print(f"\nRoot endpoint (GET /): {response.status_code}")
        if response.status_code == 200:
            print("✅ Root endpoint is working")
        else:
            print(f"❌ Unexpected status code: {response.text}")
    except Exception as e:
        print(f"❌ Error accessing root endpoint: {e}")
    
    # Test static files
    test_files = ["index.html", "detect.html", "css/styles.css"]
    for file in test_files:
        try:
            response = requests.get(f"{base_url}/{file}")
            print(f"\nTesting static file: {file} - {response.status_code}")
            if response.status_code == 200:
                print(f"✅ {file} is accessible")
            else:
                print(f"❌ {file} not found or error: {response.text}")
        except Exception as e:
            print(f"❌ Error accessing {file}: {e}")
    
    # Test API endpoints
    api_endpoints = [
        ("/api/docs", "API Documentation"),
        ("/health", "Health Check"),
    ]
    
    for endpoint, name in api_endpoints:
        try:
            response = requests.get(f"{base_url}{endpoint}")
            print(f"\nTesting {name} ({endpoint}): {response.status_code}")
            if response.status_code == 200:
                print(f"✅ {name} is working")
            else:
                print(f"⚠️  {name} returned {response.status_code}: {response.text}")
        except Exception as e:
            print(f"❌ Error accessing {endpoint}: {e}")
    
    # Check for required static files
    static_dir = Path(__file__).parent / "static"
    required_files = ["index.html", "detect.html", "js/main.js", "css/styles.css"]
    
    print("\nChecking for required static files:")
    for file in required_files:
        file_path = static_dir / file
        if file_path.exists():
            print(f"✅ Found: {file}")
        else:
            print(f"❌ Missing: {file}")

if __name__ == "__main__":
    # Allow passing a different base URL as a command line argument
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    test_server(base_url)
