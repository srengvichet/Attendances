# face_detection_and_embedding.py
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import cv2
import os
import numpy as np
from PIL import Image

mtcnn = MTCNN(image_size=160, margin=20, keep_all=False)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

def extract_embeddings(data_dir='datasets'):
    embeddings = []
    labels = []

    for person_name in os.listdir(data_dir):
        person_path = os.path.join(data_dir, person_name)
        if not os.path.isdir(person_path):
            continue

        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            img = Image.open(img_path).convert('RGB')

            face = mtcnn(img)
            if face is not None:
                with torch.no_grad():
                    embedding = resnet(face.unsqueeze(0))
                    embeddings.append(embedding.squeeze(0).numpy())
                    labels.append(person_name)

    embeddings = np.array(embeddings)
    labels = np.array(labels)

    # Save
    os.makedirs('embeddings', exist_ok=True)
    np.save('embeddings/embeddings.npy', embeddings)
    np.save('embeddings/labels.npy', labels)

if __name__ == "__main__":
    extract_embeddings()
