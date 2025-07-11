# add_new_person.py version 1

import os
import numpy as np
import pickle

import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, Normalizer

# === CONFIG ===
DATA_DIR = 'datasets1'
EMBED_PATH = 'embeddings/embeddings.npy'
LABEL_PATH = 'embeddings/labels.npy'
MODEL_PATH = 'models/face_classifier.pkl'

# === Load Existing Data ===
embeddings = np.load(EMBED_PATH)
labels = np.load(LABEL_PATH).tolist()

# === Load Classifier Components ===
with open(MODEL_PATH, 'rb') as f:
    classifier, encoder, normalizer = pickle.load(f)

# === Setup MTCNN & ResNet ===
device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(image_size=160, margin=20, keep_all=False, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# === Detect and Embed New Person ===
new_people = [d for d in os.listdir(DATA_DIR) if d not in encoder.classes_ and os.path.isdir(os.path.join(DATA_DIR, d))]

if not new_people:
    print("⚠️ No new person detected (or already trained).")
    exit()

new_embeddings = []
new_labels = []

for person in new_people:
    print(f"🔍 Processing new person: {person}")
    person_dir = os.path.join(DATA_DIR, person)

    for img_name in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img_name)
        img = Image.open(img_path).convert('RGB')

        face = mtcnn(img)
        if face is not None:
            with torch.no_grad():
                embedding = resnet(face.unsqueeze(0).to(device))
                new_embeddings.append(embedding.squeeze(0).cpu().numpy())
                new_labels.append(person)

# === Append to Existing Data ===
all_embeddings = np.concatenate([embeddings, np.array(new_embeddings)], axis=0)
all_labels = labels + new_labels

# === Normalize & Encode ===
normalized_embeddings = normalizer.transform(all_embeddings)
encoded_labels = encoder.fit_transform(all_labels)

# === Retrain Classifier ===
classifier = SVC(kernel='linear', probability=True)
classifier.fit(normalized_embeddings, encoded_labels)

# === Save Updated Artifacts ===
np.save(EMBED_PATH, all_embeddings)
np.save(LABEL_PATH, np.array(all_labels))

with open(MODEL_PATH, 'wb') as f:
    pickle.dump((classifier, encoder, normalizer), f)

print("✅ New person(s) added and classifier updated successfully.")
