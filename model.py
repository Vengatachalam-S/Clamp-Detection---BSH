from flask import Flask, render_template, request, jsonify
import torch
from torchvision import transforms
from PIL import Image
import base64
import io
from torchvision import models
import torch.nn as nn
import requests
import os

# Initialize Flask App
app = Flask(__name__)

# Google Drive File ID and Destination File
FILE_ID = "1W54xonnkYjeKqqnltCe_iz2tFX7OjCGb"
MODEL_FILE = "model.pth"

# Function to Download Model from Google Drive
def download_model_from_drive(file_id, destination):
    URL = f"https://drive.google.com/uc?id={file_id}&export=download"
    session = requests.Session()
    response = session.get(URL, stream=True)
    token = get_confirm_token(response)
    if token:
        params = {"confirm": token}
        response = session.get(URL, params=params, stream=True)
    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value
    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)

# Download Model if Not Available Locally
if not os.path.exists(MODEL_FILE):
    print("Downloading the model file from Google Drive...")
    download_model_from_drive(FILE_ID, MODEL_FILE)
    print("Model file downloaded successfully.")

# Load the Trained Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)  # Assuming 2 classes: 'okay' and 'not okay'
model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
model = model.to(device)
model.eval()

# Define Data Transformations
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Helper Function to Predict
class_names = ["not okay", "okay"]
def predict_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = data_transform(image).unsqueeze(0).to(device)
    outputs = model(image)
    _, preds = torch.max(outputs, 1)
    return class_names[preds.item()]

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    image_data = data.get("image")
    if not image_data:
        return jsonify({"error": "No image provided."}), 400

    # Decode Base64 Image
    image_data = base64.b64decode(image_data.split(",")[1])
    
    # Predict
    prediction = predict_image(image_data)
    return jsonify({"prediction": prediction})

if __name__ == "__main__":
    app.run(debug=True)
