import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import joblib

DATA_DIR = "asl_alphabet_train/asl_alphabet_train"
IMG_SIZE = 64  # Resize images to 64x64
LIMIT_PER_CLASS = 800  #  limit to speed up training

MAX_IMAGES_PER_CLASS = 500  # limit to avoid memory issues
# each letter has around 3000 images

X = []
y = []

print("Loading and preprocessing images...")

for label in os.listdir(DATA_DIR):
    folder_path = os.path.join(DATA_DIR, label)

    if not os.path.isdir(folder_path):
        continue

    count = 0
    for img_name in os.listdir(folder_path):

        if count >= MAX_IMAGES_PER_CLASS:
            break

        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        X.append(img_gray.flatten())
        y.append(label)

        count += 1
        if count >= LIMIT_PER_CLASS:
            break

print(f"Loaded {len(X)} images.")

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# ---------------- SPLIT & TRAIN ----------------
print("Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training SVM classifier...")
clf = SVC(kernel="linear", probability=True)
clf.fit(X_train, y_train)

# ---------------- EVALUATE ----------------
acc = clf.score(X_test, y_test)
print(f"Model accuracy: {acc:.2f}")

# ---------------- SAVE ----------------
os.makedirs("model", exist_ok=True)
joblib.dump(clf, "model/asl_svm_model.pkl")
print("Model saved to model/asl_svm_model.pkl")
