from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import requests
from bs4 import BeautifulSoup
import os

app = Flask(__name__)
CORS(app)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def predict_description(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=["t-shirt", "jeans", "shoes", "hat", "jacket", "bag"],
                       images=image, return_tensors="pt", padding=True).to(device)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    idx = torch.argmax(probs).item()
    return ["t-shirt", "jeans", "shoes", "hat", "jacket", "bag"][idx]

def search_product_link(query):
    headers = {"User-Agent": "Mozilla/5.0"}
    url = f"https://www.flipkart.com/search?q={query.replace(' ', '+')}"
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    a_tag = soup.find("a", href=True)
    if a_tag:
        return "https://www.flipkart.com" + a_tag['href']
    return "Sorry, product not found"

@app.route('/search', methods=['POST'])
def search():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files['image']
    filepath = os.path.join("uploads", image_file.filename)
    os.makedirs("uploads", exist_ok=True)
    image_file.save(filepath)

    description = predict_description(filepath)
    link = search_product_link(description)

    return jsonify({"result": f"{description} - {link}"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
