# recognize_and_mark_attendance.py version 2

from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2
import torch
import numpy as np
import pickle
from PIL import Image, ImageEnhance
import torchvision.transforms.functional as TF
from datetime import datetime
import csv

from firebase.firebase_helper import save_attendance_to_firebase

# Load models
mtcnn = MTCNN(image_size=160, margin=20)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# Load classifier
with open('models/face_classifier.pkl', 'rb') as f:
    classifier, encoder, normalizer = pickle.load(f)

attendance = {}
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img = TF.adjust_brightness(img, 1.1)
    img = TF.adjust_contrast(img, 1.2)
    img = ImageEnhance.Sharpness(img).enhance(1.5)

    face = mtcnn(img)
    if face is not None:
        with torch.no_grad():
            embedding = resnet(face.unsqueeze(0))
            norm_embedding = normalizer.transform(embedding.numpy())
            probs = classifier.predict_proba(norm_embedding)[0]
            pred = np.argmax(probs)
            name = encoder.inverse_transform([pred])[0]
            confidence = probs[pred]

            if confidence > 0.95 and name not in attendance:
                now = datetime.now().strftime("%H:%M:%S")
                attendance[name] = now

                # Save to Firebase
                save_attendance_to_firebase(name, now)

                # Save to CSV
                with open("attendance.csv", "a", newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([name, now])

                cv2.putText(frame, f"{name} ({confidence:.2f})", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Face Attendance", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("✅ Attendance Summary:")
for name, time in attendance.items():
    print(f"{name}: {time}")



# # recognize_and_mark_attendance.py version 1
# from PIL import Image
# from facenet_pytorch import MTCNN, InceptionResnetV1
# import cv2
# import torch
# import numpy as np
# import pickle
# from datetime import datetime
#
# # Load models
# mtcnn = MTCNN(image_size=160, margin=20)
# resnet = InceptionResnetV1(pretrained='vggface2').eval()
#
# with open('models/face_classifier.pkl', 'rb') as f:
#     classifier, encoder, normalizer = pickle.load(f)
#
# attendance = {}
#
# cap = cv2.VideoCapture(0)
#
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     face = mtcnn(Image.fromarray(img))
#
#     if face is not None:
#         with torch.no_grad():
#             embedding = resnet(face.unsqueeze(0))
#             norm_embedding = normalizer.transform(embedding.numpy())
#             probs = classifier.predict_proba(norm_embedding)[0]
#             pred = np.argmax(probs)
#             name = encoder.inverse_transform([pred])[0]
#             confidence = probs[pred]
#
#             if confidence > 0.85:
#                 now = datetime.now().strftime("%H:%M:%S")
#                 attendance[name] = now
#                 cv2.putText(frame, f"{name} ({confidence:.2f})", (10, 30),
#                             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#
#     cv2.imshow("Face Attendance", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()
#
# print("📝 Attendance log:")
# for k, v in attendance.items():
#     print(f"{k} - {v}")
