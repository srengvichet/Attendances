# train_svm_classifier.py
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, Normalizer
import pickle
import numpy as np
import os

data = np.load('embeddings/embeddings.npy')
labels = np.load('embeddings/labels.npy')

l2_normalizer = Normalizer('l2')
data = l2_normalizer.transform(data)

encoder = LabelEncoder()
labels = encoder.fit_transform(labels)

classifier = SVC(kernel='linear', probability=True)
classifier.fit(data, labels)

# Save model
os.makedirs('models', exist_ok=True)
with open('models/face_classifier.pkl', 'wb') as f:
    pickle.dump((classifier, encoder, l2_normalizer), f)
