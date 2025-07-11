import streamlit as st
import firebase_admin
from firebase_admin import credentials, db, storage
import tempfile
import uuid
from PIL import Image
import time

# --- Initialize Firebase ---
if not firebase_admin._apps:
    cred = credentials.Certificate("firebase_config.json")
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://sampleapp-1fb9a-default-rtdb.asia-southeast1.firebasedatabase.app/',
        'storageBucket': 'sampleapp-1fb9a.appspot.com'
    })

bucket = storage.bucket()
students_ref = db.reference("students")

# --- Streamlit UI ---
st.set_page_config(page_title="User CRUD", layout="centered")
st.title("📚 User CRUD Operations")

operation = st.selectbox("Select Operation", ["Create", "Read", "Update", "Delete"])

user_id = st.text_input("🆔 User ID")
# --- Live User Existence Check ---
user_exists = False
if user_id:
    user_exists = students_ref.child(user_id).get() is not None
    if user_exists:
        st.info(f"🟢 User `{user_id}` exists in the database.")
    else:
        st.warning(f"🔴 User `{user_id}` does not exist in the database.")

if operation in ["Create", "Update"]:
    full_name = st.text_input("🧑 Full Name")
    class_name = st.text_input("🏭 Class")
    role = st.selectbox("🧑‍🏫 Role", ["student", "teacher", "staff", "admin"])
    uploaded_avatar = st.file_uploader("📸 Upload Avatar", type=["jpg", "jpeg", "png"])

# --- Upload to Firebase Storage ---
def upload_avatar(image_file, user_id):
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(image_file.read())
    temp_file.flush()

    blob = bucket.blob(f'avatars/{user_id}.jpg')
    blob.upload_from_filename(temp_file.name, content_type='image/jpeg')
    blob.make_public()
    return blob.public_url

# --- Handle Operations ---
if st.button(f"{operation} User"):
    if not user_id:
        st.warning("⚠️ Please enter a User ID.")
    elif operation == "Create":
        with st.spinner("Creating student..."):
            photo_url = None
            if uploaded_avatar:
                photo_url = upload_avatar(uploaded_avatar, user_id)

            students_ref.child(user_id).set({
                "full_name": full_name,
                "class": class_name,
                "role": role,
                "photo_url": photo_url
            })
            time.sleep(1)
            st.success(f"✅ User {user_id} created.")

    elif operation == "Read":
        with st.spinner("Fetching student data..."):
            data = students_ref.child(user_id).get()
            time.sleep(0.5)
            if data:
                st.json(data)
                if data.get("photo_url"):
                    st.image(data["photo_url"], caption="Student Avatar", width=200)
            else:
                st.error("❌ Student not found.")


    elif operation == "Update":

        with st.spinner("Updating student..."):

            if user_exists:

                photo_url = students_ref.child(user_id).child("photo_url").get()

                if uploaded_avatar:
                    photo_url = upload_avatar(uploaded_avatar, user_id)

                students_ref.child(user_id).update({

                    "full_name": full_name,

                    "class": class_name,

                    "role": role,

                    "photo_url": photo_url

                })

                time.sleep(1)

                st.success(f"✅ Student {user_id} updated.")

            else:

                st.error("❌ Student not found.")

    elif operation == "Delete":
        with st.spinner("Deleting student..."):
            if students_ref.child(user_id).get():
                students_ref.child(user_id).delete()
                blob = bucket.blob(f'avatars/{user_id}.jpg')
                if blob.exists():
                    blob.delete()
                time.sleep(1)
                st.success(f"🗑️ User {user_id} deleted.")
            else:
                st.error("❌ User not found.")




# version 3
# import streamlit as st
# import firebase_admin
# from firebase_admin import credentials, db, storage
# import tempfile
# import uuid
# from PIL import Image
#
# # --- Initialize Firebase ---
# if not firebase_admin._apps:
#     cred = credentials.Certificate("firebase_config.json")
#     firebase_admin.initialize_app(cred, {
#         'databaseURL': 'https://sampleapp-1fb9a-default-rtdb.asia-southeast1.firebasedatabase.app/',
#         'storageBucket': 'sampleapp-1fb9a.appspot.com'
#     })
#
# bucket = storage.bucket()
# students_ref = db.reference("students")
#
# # --- Streamlit UI ---
# st.set_page_config(page_title="User CRUD", layout="centered")
# st.title("📚 User CRUD Operations")
#
# operation = st.selectbox("Select Operation", ["Create", "Read", "Update", "Delete"])
#
# user_id = st.text_input("🆔 User ID")
#
# if operation in ["Create", "Update"]:
#     full_name = st.text_input("🧑 Full Name")
#     class_name = st.text_input("🏫 Class")
#     role = st.selectbox("🧑‍🏫 Role", ["student", "teacher", "staff","admin"])
#     uploaded_avatar = st.file_uploader("📸 Upload Avatar", type=["jpg", "jpeg", "png"])
#
# # --- Upload to Firebase Storage ---
# def upload_avatar(image_file, user_id):
#     # Save file to a temp location
#     temp_file = tempfile.NamedTemporaryFile(delete=False)
#     temp_file.write(image_file.read())
#     temp_file.flush()
#
#     blob = bucket.blob(f'avatars/{user_id}.jpg')
#     blob.upload_from_filename(temp_file.name, content_type='image/jpeg')
#
#     # Make the URL public
#     blob.make_public()
#     return blob.public_url
#
# # --- Handle Operations ---
# if st.button(f"{operation} User"):
#     if not user_id:
#         st.warning("⚠️ Please enter a User ID.")
#     elif operation == "Create":
#         photo_url = None
#         if uploaded_avatar:
#             photo_url = upload_avatar(uploaded_avatar, user_id)
#
#         students_ref.child(user_id).set({
#             "full_name": full_name,
#             "class": class_name,
#             "role": role,
#             "photo_url": photo_url
#         })
#         st.success(f"✅ User {user_id} created.")
#
#     elif operation == "Read":
#         data = students_ref.child(user_id).get()
#         if data:
#             st.json(data)
#             if data.get("photo_url"):
#                 st.image(data["photo_url"], caption="Student Avatar", width=200)
#         else:
#             st.error("❌ Student not found.")
#
#     elif operation == "Update":
#         if students_ref.child(user_id).get():
#             photo_url = students_ref.child(user_id).child("photo_url").get()
#             if uploaded_avatar:
#                 photo_url = upload_avatar(uploaded_avatar, user_id)
#
#             students_ref.child(user_id).update({
#                 "full_name": full_name,
#                 "class": class_name,
#                 "role": role,
#                 "photo_url": photo_url
#             })
#             st.success(f"✅ Student {user_id} updated.")
#         else:
#             st.error("❌ Student not found.")
#
#     elif operation == "Delete":
#         if students_ref.child(user_id).get():
#             students_ref.child(user_id).delete()
#             # Optional: Delete avatar from storage
#             blob = bucket.blob(f'avatars/{user_id}.jpg')
#             if blob.exists():
#                 blob.delete()
#             st.success(f"🗑️ User {user_id} deleted.")
#         else:
#             st.error("❌ User not found.")


# import os #version2
# import cv2
# import numpy as np
# import torch
# import pickle
# from datetime import datetime
# from PIL import Image
# from facenet_pytorch import MTCNN, InceptionResnetV1
# import firebase_admin
# from firebase_admin import credentials, db, storage
# from sklearn.preprocessing import LabelEncoder, Normalizer
# from sklearn.svm import SVC
# import streamlit as st
#
# # --- Firebase Init ---
# if not firebase_admin._apps:
#     cred = credentials.Certificate("firebase_config.json")
#     firebase_admin.initialize_app(cred, {
#         'databaseURL': 'https://sampleapp-1fb9a-default-rtdb.asia-southeast1.firebasedatabase.app/',
#         'storageBucket': 'sampleapp-1fb9a.firebasestorage.app'
#     })
#
# # --- Models ---
# mtcnn = MTCNN(image_size=160, margin=20)
# resnet = InceptionResnetV1(pretrained='vggface2').eval()
#
# dataset_dir = "datasets"
# model_path = "models/face_classifier.pkl"
#
# # --- Streamlit UI ---
# st.set_page_config(page_title="🎓 Student Registration", layout="wide")
# st.title("📚 Student CRUD + Face Recognition Setup")
#
# with st.form("register_form"):
#     st.subheader("📝 Register New Student")
#     username = st.text_input("Username (used for face recognition)")
#     full_name = st.text_input("Full Name")
#     class_name = st.text_input("Class")
#     role = st.selectbox("Role", ["student", "teacher", "admin"])
#     submit = st.form_submit_button("📸 Capture & Register")
#
# if submit:
#     if not username or not full_name or not class_name:
#         st.warning("⚠️ Please fill in all fields.")
#         st.stop()
#
#     save_path = os.path.join(dataset_dir, username)
#     os.makedirs(save_path, exist_ok=True)
#
#     cap = cv2.VideoCapture(0)
#     count = 0
#     st_frame = st.empty()
#     st.info("📷 Look at the camera. Capturing 5 images...")
#
#     avatar_url = None
#     while count < 5:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         img = Image.fromarray(rgb)
#         face = mtcnn(img)
#
#         if face is not None:
#             filename = os.path.join(save_path, f"img_{count}.jpg")
#             cv2.imwrite(filename, cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
#             count += 1
#             st_frame.image(img, width=300, caption=f"Captured {count}/5")
#
#             if count == 1:
#                 # Save first image to Firebase Storage as avatar
#                 snapshot_name = f"avatars/{username}.jpg"
#                 temp_file = f"{username}.jpg"
#                 cv2.imwrite(temp_file, cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
#                 bucket = storage.bucket()
#                 blob = bucket.blob(snapshot_name)
#                 blob.upload_from_filename(temp_file)
#                 blob.make_public()
#                 avatar_url = blob.public_url
#                 os.remove(temp_file)
#
#     cap.release()
#     cv2.destroyAllWindows()
#
#     st.success("✅ Face capture complete. Now updating model...")
#
#     # --- Load existing model if available ---
#     if os.path.exists(model_path):
#         with open(model_path, 'rb') as f:
#             classifier, encoder, l2_normalizer = pickle.load(f)
#         embeddings_old = classifier.support_vectors_
#         labels_old = encoder.inverse_transform(classifier.classes_)
#     else:
#         embeddings_old = []
#         labels_old = []
#
#     # --- Extract only new embeddings ---
#     new_embeddings = []
#     new_labels = []
#
#     for image_name in os.listdir(save_path):
#         img_path = os.path.join(save_path, image_name)
#         try:
#             img = Image.open(img_path)
#             face = mtcnn(img)
#             if face is not None:
#                 emb = resnet(face.unsqueeze(0)).detach().numpy()
#                 new_embeddings.append(emb.squeeze())
#                 new_labels.append(username)
#         except:
#             continue
#
#     # --- Combine with existing data ---
#     all_embeddings = np.vstack([embeddings_old, new_embeddings]) if len(embeddings_old) else new_embeddings
#     all_labels = list(labels_old) + new_labels
#
#     # --- Retrain the model ---
#     encoder = LabelEncoder()
#     labels_encoded = encoder.fit_transform(all_labels)
#     l2_normalizer = Normalizer('l2')
#     all_embeddings = l2_normalizer.transform(all_embeddings)
#
#     classifier = SVC(kernel='linear', probability=True)
#     classifier.fit(all_embeddings, labels_encoded)
#
#     # --- Save updated model ---
#     with open(model_path, 'wb') as f:
#         pickle.dump((classifier, encoder, l2_normalizer), f)
#
#     # --- Save to Firebase DB ---
#     try:
#         db.reference(f"users/{username}").set({
#             "full_name": full_name,
#             "class": class_name,
#             "role": role,
#             "photo_url": avatar_url,
#             "created_at": datetime.now().isoformat()
#         })
#         st.success("🎉 Student successfully registered in Firebase DB!")
#     except Exception as e:
#         st.error(f"❌ Failed to save to Firebase DB: {e}")
#
#
#
#     st.success("🎉 Student successfully registered and incrementally trained!")


# import os #version 1
# import cv2
# import numpy as np
# import torch
# import pickle
# from datetime import datetime
# from PIL import Image
# from facenet_pytorch import MTCNN, InceptionResnetV1
# import firebase_admin
# from firebase_admin import credentials, db, storage
# from sklearn.preprocessing import LabelEncoder, Normalizer
# from sklearn.svm import SVC
# import streamlit as st
#
# # --- Firebase Init ---
# if not firebase_admin._apps:
#     cred = credentials.Certificate("firebase_config.json")
#     firebase_admin.initialize_app(cred, {
#         'databaseURL': 'https://sampleapp-1fb9a-default-rtdb.asia-southeast1.firebasedatabase.app/',
#         'storageBucket': 'sampleapp-1fb9a.appspot.com'
#     })
#
# # --- Face Models ---
# mtcnn = MTCNN(image_size=160, margin=20)
# resnet = InceptionResnetV1(pretrained='vggface2').eval()
#
# # --- Constants ---
# dataset_dir = "datasets"
# model_path = "C:/Users/ADMIN/PycharmProjects/Attendance/models/face_classifier.pkl"
#
# # --- Streamlit UI ---
# st.set_page_config(page_title="🎓 Student Registration", layout="wide")
# st.title("📚 Student CRUD + Face Recognition Setup")
#
# with st.form("register_form"):
#     st.subheader("📝 Register New Student")
#     username = st.text_input("Username (used for face recognition)")
#     full_name = st.text_input("Full Name")
#     class_name = st.text_input("Class")
#     role = st.selectbox("Role", ["student", "teacher", "admin"])
#     submit = st.form_submit_button("📸 Capture & Register")
#
# if submit:
#     if not username or not full_name or not class_name:
#         st.warning("⚠️ Please fill in all fields.")
#         st.stop()
#
#     save_path = os.path.join(dataset_dir, username)
#     os.makedirs(save_path, exist_ok=True)
#
#     cap = cv2.VideoCapture(0)
#     count = 0
#     st_frame = st.empty()
#     st.info("📷 Look at the camera. Capturing 5 images...")
#
#     avatar_url = None
#     while count < 5:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         img = Image.fromarray(rgb)
#         face = mtcnn(img)
#
#         if face is not None:
#             filename = os.path.join(save_path, f"img_{count}.jpg")
#             cv2.imwrite(filename, cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
#             count += 1
#             st_frame.image(img, width=300, caption=f"Captured {count}/5")
#
#             if count == 1:
#                 # Save first image to Firebase Storage as avatar
#                 snapshot_name = f"avatars/{username}.jpg"
#                 cv2.imwrite(username + ".jpg", cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
#                 bucket = storage.bucket()
#                 blob = bucket.blob(snapshot_name)
#                 blob.upload_from_filename(username + ".jpg")
#                 blob.make_public()
#                 avatar_url = blob.public_url
#                 os.remove(username + ".jpg")
#
#     cap.release()
#     cv2.destroyAllWindows()
#
#     st.success("✅ Face image collection completed. Now training model...")
#
#     # --- Train Model (Incrementally) ---
#     embeddings, labels = [], []
#     for person in os.listdir(dataset_dir):
#         person_path = os.path.join(dataset_dir, person)
#         for img_file in os.listdir(person_path):
#             try:
#                 image = Image.open(os.path.join(person_path, img_file))
#                 face = mtcnn(image)
#                 if face is not None:
#                     emb = resnet(face.unsqueeze(0)).detach().numpy()
#                     embeddings.append(emb.squeeze())
#                     labels.append(person)
#             except:
#                 continue
#
#     l2_normalizer = Normalizer('l2')
#     embeddings = l2_normalizer.transform(embeddings)
#     encoder = LabelEncoder()
#     labels_encoded = encoder.fit_transform(labels)
#
#     clf = SVC(kernel='linear', probability=True)
#     clf.fit(embeddings, labels_encoded)
#
#     with open(model_path, 'wb') as f:
#         pickle.dump((clf, encoder, l2_normalizer), f)
#
#     # --- Save to Firebase DB ---
#     db.reference(f"users/{username}").set({
#         "full_name": full_name,
#         "class": class_name,
#         "role": role,
#         "photo_url": avatar_url,
#         "created_at": datetime.now().isoformat()
#     })
#
#     st.success("🎉 Student successfully registered and trained!")
#
# # import os
# # import cv2
# # import numpy as np
# # import torch
# # import pickle
# # from datetime import datetime
# # from PIL import Image
# # from facenet_pytorch import MTCNN, InceptionResnetV1
# # import firebase_admin
# # from firebase_admin import credentials, db, storage
# # from sklearn.preprocessing import LabelEncoder, Normalizer
# # from sklearn.svm import SVC
# # import streamlit as st
# #
# # # --- Firebase Init ---
# # if not firebase_admin._apps:
# #     cred = credentials.Certificate("firebase_config.json")
# #     firebase_admin.initialize_app(cred, {
# #         'databaseURL': 'https://sampleapp-1fb9a-default-rtdb.asia-southeast1.firebasedatabase.app/',
# #         'storageBucket': 'sampleapp-1fb9a.firebasestorage.app'
# #     })
# #
# # # --- Face Models ---
# # mtcnn = MTCNN(image_size=160, margin=20)
# # resnet = InceptionResnetV1(pretrained='vggface2').eval()
# #
# # dataset_dir = "datasets"
# # # model_path = "C:/Users/ADMIN/PycharmProjects/Attendance/models/face_classifier.pkl"
# #
# # # --- Streamlit Page ---
# # st.set_page_config(page_title="🎓 Student Registration", layout="wide")
# # st.title("📊 CRUD Operation Overview")
# # st.header("🎓 Student Registration & Training")
# #
# # with st.form("register_form"):
# #     st.subheader("Register New Student")
# #     username = st.text_input("Username (must match face_classifier)")
# #     full_name = st.text_input("Full Name")
# #     class_name = st.text_input("Class")
# #     role = st.selectbox("Role", ["student", "teacher", "dean", "vice_dean"])
# #     submit = st.form_submit_button("📸 Capture & Register")
# #
# # if submit:
# #     if not username or not full_name or not class_name:
# #         st.warning("⚠️ Please fill in all fields.")
# #     else:
# #         # save_path = os.path.join(dataset_dir, username)
# #         # os.makedirs(save_path, exist_ok=True)
# #
# #         # cap = cv2.VideoCapture(0)
# #         # st_frame = st.empty()
# #         # st_frame.info("Capturing face images. Please look at the camera...")
# #
# #         # --- Firebase Save ---
# #         ref = db.reference(f"users/{username}")
# #         ref.set({
# #             "full_name": full_name,
# #             "class": class_name,
# #             "role": role,
# #             "created_at": datetime.now().isoformat()
# #         })
# #
# #         st.success("✅ Student registered and model updated!")
