# train_model.py
import os
import numpy as np
from skimage import io, color
from skimage.transform import resize
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
from tqdm import tqdm

def extract_histogram(image_path):
    img = io.imread(image_path)
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

X, y = [], []
base = "dataset"
for label, folder in enumerate(["real","fake"]):
    folder_path = os.path.join(base, folder)
    if not os.path.exists(folder_path):
        continue
    for fname in tqdm(os.listdir(folder_path)):
        if not (fname.lower().endswith(".jpg") or fname.lower().endswith(".png")):
            continue
        f = extract_histogram(os.path.join(folder_path, fname))
        if f is not None:
            X.append(f)
            y.append(label)

X = np.array(X)
y = np.array(y)
print("데이터 수:", len(X))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
svm = SVC(kernel='rbf', C=2, gamma='scale', probability=True)
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
print("정확도:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=["real","fake"]))
joblib.dump(svm, "deepfake_svm_model.joblib")
print("모델 저장됨: deepfake_svm_model.joblib")
