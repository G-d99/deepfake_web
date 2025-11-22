# app.py
import os
from flask import Flask, request, render_template, jsonify
import joblib
from skimage import io, color
from skimage.transform import resize
import numpy as np
import tempfile

app = Flask(__name__)
model = joblib.load("deepfake_svm_model.joblib")

def extract_histogram_from_file(filepath):
    img = io.imread(filepath)
    if img is None:
        return None
    if img.ndim == 3:
        gray = color.rgb2gray(img)
    else:
        gray = img
    gray_resized = resize(gray, (64,64))
    hist, _ = np.histogram(gray_resized.flatten(), bins=64, range=(0,1))
    hist = hist / np.sum(hist)
    return hist

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    f = request.files.get("file")
    if f is None:
        return jsonify({"error":"파일 없음"}), 400
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        f.save(tmp.name)
        feats = extract_histogram_from_file(tmp.name)
    if feats is None:
        return jsonify({"error":"이미지 처리 실패"}), 400
    pred = model.predict([feats])[0]
    prob = model.predict_proba([feats])[0].tolist()
    return jsonify({"result": "real" if pred==0 else "fake", "probability": prob})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
