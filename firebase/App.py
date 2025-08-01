# version 3 set the location for face recognition
import json

import streamlit as st
import cv2
import numpy as np
import pickle
from datetime import datetime, time, timedelta
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from firebase_helper import save_attendance_to_firebase
from PIL import Image, ImageEnhance
import torchvision.transforms.functional as TF
import firebase_admin
from firebase_admin import credentials, storage, db
import os
import requests
from io import BytesIO

# --- Firebase Init ---
cred_path = "firebase/firebase_config.json"
if not firebase_admin._apps:
    cred = credentials.Certificate(json.loads(st.secrets["FIREBASE_CONFIG_JSON"]))
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://sampleapp-1fb9a-default-rtdb.asia-southeast1.firebasedatabase.app/',
        'storageBucket': 'sampleapp-1fb9a.appspot.com'
    })

# --- Load Classifier ---
with open('models/face_classifier.pkl', 'rb') as f:
    classifier, encoder, normalizer = pickle.load(f)

# --- Load MTCNN & FaceNet ---
mtcnn = MTCNN(image_size=160, margin=20)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# --- UI ---
st.set_page_config(page_title="📸 Face Attendance Scanner", layout="centered")
st.title("📸 Real-Time Face Attendance Scanner")

start = st.button("🎬 Start Attendance")
FRAME_WINDOW = st.image([])

cap = None
attendance = {}
unregistered_warned = set()

if start:
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # --- Define ROI ---
        h, w, _ = frame.shape
        roi_x1, roi_y1 = int(w * 0.2), int(h * 0.2)
        roi_x2, roi_y2 = int(w * 0.8), int(h * 0.8)

        # Draw ROI box on original frame
        cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 255), 2)

        roi_frame = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        roi_rgb = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(roi_rgb)

        # Enhance
        pil_img = TF.adjust_brightness(pil_img, 1.1)
        pil_img = TF.adjust_contrast(pil_img, 1.2)
        pil_img = ImageEnhance.Sharpness(pil_img).enhance(1.5)

        boxes, _ = mtcnn.detect(pil_img)

        if boxes is not None:
            face_tensor = mtcnn(pil_img)

            if face_tensor is not None:
                with torch.no_grad():
                    embedding = resnet(face_tensor.unsqueeze(0))
                    norm_embedding = normalizer.transform(embedding.numpy())
                    probs = classifier.predict_proba(norm_embedding)[0]
                    pred = np.argmax(probs)
                    name = encoder.inverse_transform([pred])[0]
                    confidence = probs[pred]

                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box)
                        x1 += roi_x1
                        x2 += roi_x1
                        y1 += roi_y1
                        y2 += roi_y1
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"{name} ({confidence*100:.1f}%)", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    if confidence > 0.80 and name not in attendance:
                        user_ref = db.reference(f"students/{name}")
                        user_data = user_ref.get()
                        if not user_data:
                            if name not in unregistered_warned:
                                st.error(f"⚠️ Student `{name}` is not registered in the system.")
                                unregistered_warned.add(name)

                        now_obj = datetime.now()
                        now = now_obj.strftime("%I:%M:%S %p")
                        date_str = now_obj.strftime("%Y-%m-%d")
                        attendance[name] = now

                        morning_start = time(6, 0)
                        morning_on_time_end = time(7, 0)
                        morning_late_end = time(11, 59)
                        evening_start = time(12, 0)
                        evening_end = time(18, 0)
                        current_time = now_obj.time()

                        period = None
                        status = None
                        minutes_offset = 0

                        if morning_start <= current_time <= morning_late_end:
                            period = "morning"
                            scan_time = timedelta(hours=now_obj.hour, minutes=now_obj.minute)
                            cutoff = timedelta(hours=7, minutes=0)
                            status = "on_time" if current_time <= morning_on_time_end else "late"
                            if status == "late":
                                minutes_offset = int((scan_time - cutoff).total_seconds() / 60)

                        elif current_time < evening_start:
                            wait_min = int((datetime.combine(datetime.today(), evening_start) - now_obj).total_seconds() / 60)
                            st.warning(f"⏳ Too early to check-out. Come back in {wait_min} minutes.")
                            continue

                        elif evening_start <= current_time <= evening_end:
                            period = "evening"
                            status = "on_time"

                        elif current_time > evening_end:
                            period = "evening"
                            status = "late"
                            minutes_offset = int((datetime.combine(datetime.today(), current_time) - datetime.combine(datetime.today(), evening_end)).total_seconds() / 60)

                        else:
                            st.warning("⛔ Attendance not in allowed range.")
                            continue

                        ref = db.reference(f"attendance/{date_str}/{period}/{name}")
                        if ref.get():
                            st.warning(f"✅ {name} already marked for {period}.")
                            continue

                        snapshot_path = f"{name}.jpg"
                        bucket = storage.bucket()
                        blob = bucket.blob(f"snapshots/{snapshot_path}")
                        if not blob.exists():
                            cv2.imwrite(snapshot_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                            blob.upload_from_filename(snapshot_path)
                            blob.make_public()
                        photo_url = blob.public_url

                        save_attendance_to_firebase(
                            user_id=name,
                            timestamp=now,
                            photo_url=photo_url,
                            period=period,
                            status=status,
                            minutes_offset=minutes_offset
                        )

                        col1, col2 = st.columns([1, 3])
                        try:
                            if os.path.exists(snapshot_path):
                                img = Image.open(snapshot_path)
                            else:
                                response = requests.get(photo_url)
                                img = Image.open(BytesIO(response.content))
                            col1.image(img, width=160, caption=name)
                        except Exception as e:
                            col1.warning("⚠️ Image load failed")
                            st.error(str(e))

                        with col2:
                            st.markdown(f"### ✅ {name}")
                            st.markdown(f"🕒 Time: `{now}`")
                            st.markdown(f"📍 Period: `{period}`")
                            st.markdown(f"📌 Status: `{status}`")
                            if status == "late":
                                st.markdown(f"⚠️ You are **{minutes_offset} minutes late** for check-in.")
                            elif status == "on_time":
                                st.success("✅ You're on time!")

                        if os.path.exists(snapshot_path):
                            os.remove(snapshot_path)

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

if cap:
    cap.release()
    cv2.destroyAllWindows()

st.markdown("---")
st.markdown("🔁 After scanning, view records in the [📊 Dashboard](/Dashboard)")





# version 2
# import streamlit as st
# import cv2
# import numpy as np
# import pickle
# from datetime import datetime, time, timedelta
# import torch
# from facenet_pytorch import MTCNN, InceptionResnetV1
# from firebase_helper import save_attendance_to_firebase
# from PIL import Image, ImageEnhance
# import torchvision.transforms.functional as TF
# import firebase_admin
# from firebase_admin import credentials, storage, db
# import os
# import requests
# from io import BytesIO
#
# # --- Firebase Init ---
# cred_path = "C:/Users/ADMIN/PycharmProjects/Attendance/firebase/firebase_config.json"
# if not firebase_admin._apps:
#     cred = credentials.Certificate(cred_path)
#     firebase_admin.initialize_app(cred, {
#         'databaseURL': 'https://sampleapp-1fb9a-default-rtdb.asia-southeast1.firebasedatabase.app/',
#         'storageBucket': 'sampleapp-1fb9a.appspot.com'
#     })
#
# # --- Load Classifier ---
# with open('C:/Users/ADMIN/PycharmProjects/Attendance/models/face_classifier.pkl', 'rb') as f:
#     classifier, encoder, normalizer = pickle.load(f)
#
# # --- Load MTCNN & FaceNet ---
# mtcnn = MTCNN(image_size=160, margin=20)
# resnet = InceptionResnetV1(pretrained='vggface2').eval()
#
# # --- UI ---
# st.set_page_config(page_title="📸 Face Attendance Scanner", layout="centered")
# st.title("📸 Real-Time Face Attendance Scanner")
#
# start = st.button("🎬 Start Attendance")
# FRAME_WINDOW = st.image([])
#
# cap = None
# attendance = {}
# unregistered_warned = set()
#
# if start:
#     cap = cv2.VideoCapture(0)
#
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#
#         img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         pil_img = Image.fromarray(img_rgb)
#
#         # Enhance brightness & contrast
#         pil_img = TF.adjust_brightness(pil_img, 1.1)
#         pil_img = TF.adjust_contrast(pil_img, 1.2)
#         pil_img = ImageEnhance.Sharpness(pil_img).enhance(1.5)
#
#         # --- Detect face bounding boxes ---
#         boxes, _ = mtcnn.detect(pil_img)
#
#         if boxes is not None:
#             # Recognize faces
#             face_tensor = mtcnn(pil_img)
#
#             if face_tensor is not None:
#                 with torch.no_grad():
#                     embedding = resnet(face_tensor.unsqueeze(0))
#                     norm_embedding = normalizer.transform(embedding.numpy())
#                     probs = classifier.predict_proba(norm_embedding)[0]
#                     pred = np.argmax(probs)
#                     name = encoder.inverse_transform([pred])[0]
#                     confidence = probs[pred]
#
#                     # Draw all detected boxes (1 face for now)
#                     for box in boxes:
#                         x1, y1, x2, y2 = map(int, box)
#                         cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                         cv2.putText(img_rgb, f"{name} ({confidence*100:.1f}%)", (x1, y1 - 10),
#                                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
#
#                     if confidence > 0.80 and name not in attendance:
#                         # --- Check if student exists ---
#                         user_ref = db.reference(f"students/{name}")
#                         user_data = user_ref.get()
#                         if not user_data:
#                             if name not in unregistered_warned:
#                                 st.error(f"⚠️ Student `{name}` is not registered in the system.")
#                                 unregistered_warned.add(name)
#
#                         now_obj = datetime.now()
#                         now = now_obj.strftime("%I:%M:%S %p")
#                         date_str = now_obj.strftime("%Y-%m-%d")
#                         attendance[name] = now
#
#                         # --- Time logic ---
#                         morning_start = time(6, 0)
#                         morning_on_time_end = time(7, 0)
#                         morning_late_end = time(11, 59)
#                         evening_start = time(12, 0)
#                         evening_end = time(18, 0)
#                         current_time = now_obj.time()
#
#                         period = None
#                         status = None
#                         minutes_offset = 0
#
#                         if morning_start <= current_time <= morning_late_end:
#                             period = "morning"
#                             scan_time = timedelta(hours=now_obj.hour, minutes=now_obj.minute)
#                             cutoff = timedelta(hours=7, minutes=0)
#
#                             if current_time <= morning_on_time_end:
#                                 status = "on_time"
#                             else:
#                                 status = "late"
#                                 minutes_offset = int((scan_time - cutoff).total_seconds() / 60)
#
#                         elif current_time < evening_start:
#                             wait_min = int((datetime.combine(datetime.today(), evening_start) - now_obj).total_seconds() / 60)
#                             st.warning(f"⏳ Too early to check-out. Come back in {wait_min} minutes.")
#                             continue
#
#                         elif evening_start <= current_time <= evening_end:
#                             period = "evening"
#                             status = "on_time"
#
#                         elif current_time > evening_end:
#                             period = "evening"
#                             status = "late"
#                             minutes_offset = int((datetime.combine(datetime.today(), current_time) - datetime.combine(
#                                 datetime.today(), evening_end)).total_seconds() / 60)
#
#                         else:
#                             st.warning("⛔ Attendance not in allowed range.")
#                             continue
#
#                         # --- Check if already recorded ---
#                         ref = db.reference(f"attendance/{date_str}/{period}/{name}")
#                         if ref.get():
#                             st.warning(f"✅ {name} already marked for {period}.")
#                             continue
#
#                         # --- Save snapshot only once ---
#                         snapshot_path = f"{name}.jpg"
#                         bucket = storage.bucket()
#                         blob = bucket.blob(f"snapshots/{snapshot_path}")
#                         if not blob.exists():
#                             cv2.imwrite(snapshot_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
#                             blob.upload_from_filename(snapshot_path)
#                             blob.make_public()
#                         photo_url = blob.public_url
#
#                         # --- Save to Firebase ---
#                         save_attendance_to_firebase(
#                             user_id=name,
#                             timestamp=now,
#                             photo_url=photo_url,
#                             period=period,
#                             status=status,
#                             minutes_offset=minutes_offset
#                         )
#
#                         # --- Display Snapshot ---
#                         col1, col2 = st.columns([1, 3])
#                         try:
#                             if os.path.exists(snapshot_path):
#                                 img = Image.open(snapshot_path)
#                             else:
#                                 response = requests.get(photo_url)
#                                 img = Image.open(BytesIO(response.content))
#                             col1.image(img, width=160, caption=name)
#                         except Exception as e:
#                             col1.warning("❌ Image load failed")
#                             st.error(str(e))
#
#                         with col2:
#                             st.markdown(f"### ✅ {name}")
#                             st.markdown(f"🕒 Time: `{now}`")
#                             st.markdown(f"📍 Period: `{period}`")
#                             st.markdown(f"📌 Status: `{status}`")
#
#                             if status == "late":
#                                 st.markdown(f"⚠️ You are **{minutes_offset} minutes late** for check-in.")
#                             elif status == "early":
#                                 st.markdown(f"⚠️ You are **{minutes_offset} minutes early** for check-out.")
#                             elif status == "on_time":
#                                 st.success("✅ You're on time!")
#
#                         if os.path.exists(snapshot_path):
#                             os.remove(snapshot_path)
#
#         FRAME_WINDOW.image(img_rgb)
#
# if cap:
#     cap.release()
#     cv2.destroyAllWindows()
#
# st.markdown("---")
# st.markdown("🔁 After scanning, view records in the [📊 Dashboard](Dashboard.py)")



# version 1
# import streamlit as st
# import cv2
# import numpy as np
# import pickle
# from datetime import datetime, time
# import torch
# from facenet_pytorch import MTCNN, InceptionResnetV1
# from firebase_helper import save_attendance_to_firebase
# from PIL import Image, ImageEnhance
# import torchvision.transforms.functional as TF
# import firebase_admin
# from firebase_admin import credentials, storage, db
# import os
# from datetime import timedelta
#
#
# # --- Firebase Init ---
# cred_path = "C:/Users/ADMIN/PycharmProjects/Attendance/firebase/firebase_config.json"
# if not firebase_admin._apps:
#     cred = firebase_admin.credentials.Certificate(cred_path)
#     firebase_admin.initialize_app(cred, {
#         'databaseURL': 'https://sampleapp-1fb9a-default-rtdb.asia-southeast1.firebasedatabase.app/',
#         'storageBucket': 'sampleapp-1fb9a.firebasestorage.app'
#     })
#
# # --- Load Classifier ---
# with open('C:/Users/ADMIN/PycharmProjects/Attendance/models/face_classifier.pkl', 'rb') as f:
#     classifier, encoder, normalizer = pickle.load(f)
#
# # --- Load MTCNN & FaceNet ---
# mtcnn = MTCNN(image_size=160, margin=20)
# resnet = InceptionResnetV1(pretrained='vggface2').eval()
#
# # --- UI ---
# st.set_page_config(page_title="📸 Face Attendance Scanner", layout="centered")
# st.title("📸 Real-Time Face Attendance Scanner")
#
# start = st.button("🎬 Start Attendance")
# FRAME_WINDOW = st.image([])
#
# cap = None
# attendance = {}
# unregistered_warned = set()  # ✅ Add this line to track warned names
#
# if start:
#     cap = cv2.VideoCapture(0)  # 0 for built-in, 1/2 for external webcam
#
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#
#         img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         pil_img = Image.fromarray(img_rgb)
#
#         # Enhance brightness & contrast
#         pil_img = TF.adjust_brightness(pil_img, 1.1)
#         pil_img = TF.adjust_contrast(pil_img, 1.2)
#         pil_img = ImageEnhance.Sharpness(pil_img).enhance(1.5)
#
#         face = mtcnn(pil_img)
#         if face is not None:
#             with torch.no_grad():
#                 embedding = resnet(face.unsqueeze(0))
#                 norm_embedding = normalizer.transform(embedding.numpy())
#                 probs = classifier.predict_proba(norm_embedding)[0]
#                 pred = np.argmax(probs)
#                 name = encoder.inverse_transform([pred])[0]
#                 confidence = probs[pred]
#
#                 if confidence > 0.80 and name not in attendance:
#
#                     # --- Check if student exists in Firebase 'students' ---
#                     user_ref = db.reference(f"students/{name}")
#                     user_data = user_ref.get()
#                     if not user_data:
#                         if name not in unregistered_warned:
#                             st.error(f"⚠️ Student `{name}` is not registered in the system.")
#                             unregistered_warned.add(name)
#                     #     continue  # ⚠️ Skip this face, but keep camera open
#
#                     now_obj = datetime.now()
#                     now = now_obj.strftime("%I:%M:%S %p")
#                     date_str = now_obj.strftime("%Y-%m-%d")
#                     attendance[name] = now
#
#                     # --- Define time ranges ---
#                     morning_start = time(6, 0)
#                     morning_on_time_end = time(7, 0)
#                     morning_late_end = time(11, 59)
#
#                     # evening_start = time(17, 0)
#                     evening_start = time(12, 0)  #for testing only
#                     evening_end = time(18, 0)
#
#                     current_time = now_obj.time()
#
#                     # --- Determine period & status ---
#                     period = None
#                     status = None
#                     minutes_offset = 0
#
#                     if morning_start <= current_time <= morning_late_end:
#                         period = "morning"
#                         scan_time = timedelta(hours=now_obj.hour, minutes=now_obj.minute)
#                         cutoff = timedelta(hours=7, minutes=0)
#
#                         if current_time <= morning_on_time_end:
#                             status = "on_time"
#                         else:
#                             status = "late"
#                             minutes_offset = int((scan_time - cutoff).total_seconds() / 60)
#
#                     elif current_time < evening_start:
#                         st.warning(
#                             f"⏳ Too early to check-out. Come back in {int((datetime.combine(datetime.today(), evening_start) - now_obj).total_seconds() / 60)} minutes.")
#                         continue
#
#                     elif evening_start <= current_time <= evening_end:
#                         period = "evening"
#                         status = "on_time"
#
#                     elif current_time > evening_end:
#                         period = "evening"
#                         status = "late"
#                         minutes_offset = int((datetime.combine(datetime.today(), current_time) - datetime.combine(
#                             datetime.today(), evening_end)).total_seconds() / 60)
#
#                     else:
#                         st.warning("⛔ Attendance not in allowed range.")
#                         continue
#
#                     # --- Check if already recorded ---
#                     ref = db.reference(f"attendance/{date_str}/{period}/{name}")
#                     if ref.get():
#                         st.warning(f"✅ {name} already marked for {period}.")
#                         continue
#
#                     # --- Save snapshot only once ---
#                     snapshot_path = f"{name}.jpg"
#                     bucket = storage.bucket()
#                     blob = bucket.blob(f"snapshots/{snapshot_path}")
#                     if not blob.exists():
#                         cv2.imwrite(snapshot_path, frame)
#                         blob.upload_from_filename(snapshot_path)
#                         blob.make_public()
#                     photo_url = blob.public_url
#
#                     # --- Save to Firebase ---
#                     save_attendance_to_firebase(
#                         user_id=name,
#                         timestamp=now,
#                         photo_url=photo_url,
#                         period=period,
#                         status=status,
#                         minutes_offset=minutes_offset
#                     )
#
#                     # --- Display message ---
#                     col1, col2 = st.columns([1, 3])
#                     try:
#                         if os.path.exists(snapshot_path):
#                             img = Image.open(snapshot_path)
#                         else:
#                             import requests
#                             from io import BytesIO
#
#                             response = requests.get(photo_url)
#                             img = Image.open(BytesIO(response.content))
#                         col1.image(img, width=160, caption=name)
#                     except Exception as e:
#                         col1.warning("❌ Image load failed")
#                         st.error(str(e))
#
#                     with col2:
#                         st.markdown(f"### ✅ {name}")
#                         st.markdown(f"🕒 Time: `{now}`")
#                         st.markdown(f"📍 Period: `{period}`")
#                         st.markdown(f"📌 Status: `{status}`")
#
#                         if status == "late":
#                             st.markdown(f"⚠️ You are **{minutes_offset} minutes late** for check-in.")
#                         elif status == "early":
#                             st.markdown(f"⚠️ You are **{minutes_offset} minutes early** for check-out.")
#                         elif status == "on_time":
#                             st.success("✅ You're on time!")
#
#                     if os.path.exists(snapshot_path):
#                         os.remove(snapshot_path)
#
#         FRAME_WINDOW.image(img_rgb)
#
# if cap:
#     cap.release()
#     cv2.destroyAllWindows()
#
# st.markdown("---")
# st.markdown("🔁 After scanning, view records in the [📊 Dashboard](Dashboard.py)")


# # App.py version 1
#
# import streamlit as st
# import cv2
# import numpy as np
# import pickle
# import tempfile
# from datetime import datetime
#
# import torch
# from facenet_pytorch import MTCNN, InceptionResnetV1
# from firebase_helper import save_attendance_to_firebase
# from PIL import Image, ImageEnhance
# import torchvision.transforms.functional as TF
# import firebase_admin
# from firebase_admin import storage
#
# import streamlit as st
#
#
# # --- Firebase Init ---
# cred_path = "C:/Users/ADMIN/PycharmProjects/Attendance/firebase/firebase_config.json"
# if not firebase_admin._apps:
#     cred = firebase_admin.credentials.Certificate(cred_path)
#     firebase_admin.initialize_app(cred, {
#         'databaseURL': 'https://sampleapp-1fb9a-default-rtdb.asia-southeast1.firebasedatabase.app/',  # replace with your project URL
#
#     })
#
# # --- Load Classifier ---
# with open('C:/Users/ADMIN/PycharmProjects/Attendance/models/face_classifier.pkl', 'rb') as f:
#     classifier, encoder, normalizer = pickle.load(f)
#
# # --- Load MTCNN & FaceNet ---
# mtcnn = MTCNN(image_size=160, margin=20)
# resnet = InceptionResnetV1(pretrained='vggface2').eval()
#
# # --- UI ---
# st.set_page_config(page_title="📸 Face Attendance Scanner", layout="centered")
# st.title("📸 Real-Time Face Attendance Scanner")
#
# start = st.button("🎬 Start Attendance")
# FRAME_WINDOW = st.image([])
#
# cap = None
# attendance = {}
#
# if start:
#     cap = cv2.VideoCapture(2)# 0 for built-in webcam, 1 is for external webcam
#
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#
#         img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         pil_img = Image.fromarray(img_rgb)
#
#         # Enhance brightness & contrast (real-time augmentation)
#         pil_img = TF.adjust_brightness(pil_img, 1.1)
#         pil_img = TF.adjust_contrast(pil_img, 1.2)
#         pil_img = ImageEnhance.Sharpness(pil_img).enhance(1.5)
#
#         face = mtcnn(pil_img)
#         if face is not None:
#             with torch.no_grad():
#                 embedding = resnet(face.unsqueeze(0))
#                 norm_embedding = normalizer.transform(embedding.numpy())
#                 probs = classifier.predict_proba(norm_embedding)[0]
#                 pred = np.argmax(probs)
#                 name = encoder.inverse_transform([pred])[0]
#                 confidence = probs[pred]
#
#                 if confidence > 0.80 and name not in attendance:
#                     now = datetime.now().strftime("%H:%M:%S")
#                     attendance[name] = now
#
#                     # Save snapshot
#                     snapshot_path = f"{name}_{now.replace(':','-')}.jpg"
#                     cv2.imwrite(snapshot_path, frame)
#
#                     # Upload to Firebase Storage
#                     bucket = storage.bucket(name="sampleapp-1fb9a.firebasestorage.app")
#                     blob = bucket.blob(f"snapshots/{name}.jpg")  # 🔄 Fixed: use fixed name instead of timestamp
#
#                     if not blob.exists():  # ✅ Save only if it doesn't exist
#                         cv2.imwrite(snapshot_path, frame)
#                         blob.upload_from_filename(snapshot_path)
#                         blob.make_public()
#                         st.info(f"📸 Uploaded snapshot for {name}")
#                     else:
#                         st.info(f"📁 Snapshot for {name} already exists")
#
#                     photo_url = blob.public_url
#
#                     # Save to Firebase DB
#                     save_attendance_to_firebase(name, now, photo_url=photo_url)
#                     # Show confirmation on screen
#                     col1, col2 = st.columns([1, 3])
#
#                     # Load the image (just taken)
#                     image = Image.open(snapshot_path)
#                     col1.image(image, width=160, caption=name)
#
#                     # Display checkbox or confirmation
#                     with col2:
#                         st.markdown(f"### ✅ {name}")
#                         st.checkbox(f"Attendance marked at {now}", value=True, disabled=True, key=name)
#
#                     st.success(f"✅ Attendance recorded for {name}")
#
#         FRAME_WINDOW.image(img_rgb)
#         # if st.button("🛑 Stop"):
#         #     cap.release()
#         #     break
#
# if cap:
#     cap.release()
#     cv2.destroyAllWindows()
#
# st.markdown("---")
# st.markdown("🔁 After scanning, view records in the [📊 Dashboard](Dashboard.py)")
