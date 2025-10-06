
import requests

# Create a dummy file to upload
with open("dummy.txt", "w") as f:
    f.write("This is a test file.")

# The URL of the upload endpoint
url = "http://127.0.0.1:8000/upload/2"

# The file to upload
files = {"file": open("dummy.txt", "rb")}

try:
    # Send the POST request
    response = requests.post(url, files=files)

    # Print the response
    print(f"Status Code: {response.status_code}")
    print(f"Response JSON: {response.json()}")

except requests.exceptions.ConnectionError as e:
    print(f"Connection error: {e}")
    print("Please make sure the FastAPI server is running on http://127.0.0.1:8000")

finally:
    # Clean up the dummy file
    import os
    os.remove("dummy.txt")
