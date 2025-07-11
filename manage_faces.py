# manage_faces.py

import os
import argparse
import numpy as np
import pickle
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, Normalizer
import torch

# === CONFIG ===
DATA_DIR = 'datasets2'
EMBED_PATH = 'embeddings/embeddings.npy'
LABEL_PATH = 'embeddings/labels.npy'
MODEL_PATH = 'models/face_classifier.pkl'

# === Load Components ===
def load_data_and_model():
    embeddings = np.load(EMBED_PATH)
    labels = np.load(LABEL_PATH).tolist()
    with open(MODEL_PATH, 'rb') as f:
        classifier, encoder, normalizer = pickle.load(f)
    return embeddings, labels, classifier, encoder, normalizer

# === Save Model ===
def save_model(embeddings, labels, classifier, encoder, normalizer):
    np.save(EMBED_PATH, embeddings)
    np.save(LABEL_PATH, np.array(labels))
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump((classifier, encoder, normalizer), f)

# === Face Embedding ===
def extract_embeddings(person_dir, person_name):
    mtcnn = MTCNN(image_size=160, margin=20, keep_all=False)
    resnet = InceptionResnetV1(pretrained='vggface2').eval()

    person_embeddings = []
    for img_name in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img_name)
        try:
            img = Image.open(img_path).convert('RGB')
            face = mtcnn(img)
            if face is not None:
                with torch.no_grad():
                    embedding = resnet(face.unsqueeze(0))
                    person_embeddings.append(embedding.squeeze(0).numpy())
        except Exception as e:
            print(f"⚠️ Failed to process {img_name}: {e}")
    return person_embeddings

# === Add Person(s) ===
def add_new_people():
    embeddings, labels, classifier, encoder, normalizer = load_data_and_model()
    existing_names = set(encoder.classes_)

    new_people = [d for d in os.listdir(DATA_DIR) if d not in existing_names and os.path.isdir(os.path.join(DATA_DIR, d))]

    if not new_people:
        print("⚠️ No new person found.")
        return

    new_embeddings, new_labels = [], []

    for person in new_people:
        print(f"➕ Adding {person}")
        person_dir = os.path.join(DATA_DIR, person)
        person_embeds = extract_embeddings(person_dir, person)
        new_embeddings.extend(person_embeds)
        new_labels.extend([person] * len(person_embeds))

    all_embeddings = np.concatenate([embeddings, np.array(new_embeddings)], axis=0)
    all_labels = labels + new_labels

    normalized_embeddings = normalizer.transform(all_embeddings)
    encoded_labels = encoder.fit_transform(all_labels)

    classifier = SVC(kernel='linear', probability=True)
    classifier.fit(normalized_embeddings, encoded_labels)

    save_model(all_embeddings, all_labels, classifier, encoder, normalizer)
    print("✅ New people added and classifier updated.")

# === Remove Person ===
def remove_person(target_name):
    embeddings, labels, classifier, encoder, normalizer = load_data_and_model()

    if target_name not in encoder.classes_:
        print(f"❌ Person '{target_name}' not found in trained data.")
        return

    # Filter out target
    indices = [i for i, label in enumerate(labels) if label != target_name]
    updated_embeddings = embeddings[indices]
    updated_labels = [labels[i] for i in indices]

    normalized_embeddings = normalizer.transform(updated_embeddings)
    encoded_labels = encoder.fit_transform(updated_labels)

    classifier = SVC(kernel='linear', probability=True)
    classifier.fit(normalized_embeddings, encoded_labels)

    save_model(updated_embeddings, updated_labels, classifier, encoder, normalizer)
    print(f"🗑️ Removed '{target_name}' and updated classifier.")

# === List Users ===
def list_users():
    _, labels, _, _, _ = load_data_and_model()
    names = sorted(set(labels))
    print("📋 Trained Users:")
    for name in names:
        print(f" - {name}")

# === Main Entry ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manage Facial Recognition Users")
    parser.add_argument('--add', action='store_true', help='Add new person(s) from datasets2/<name>/')
    parser.add_argument('--remove', type=str, help='Remove person by name')
    parser.add_argument('--list', action='store_true', help='List all trained users')
    args = parser.parse_args()

    if args.add:
        add_new_people()
    elif args.remove:
        remove_person(args.remove)
    elif args.list:
        list_users()
    else:
        parser.print_help()
