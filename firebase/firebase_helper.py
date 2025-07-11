# firebase_helper.py

import firebase_admin
from firebase_admin import credentials, db
from datetime import datetime

# Initialize Firebase (run once)
cred = credentials.Certificate("firebase_config.json")
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://sampleapp-1fb9a-default-rtdb.asia-southeast1.firebasedatabase.app/',  # 🔁 Replace this
        'storageBucket': 'sampleapp-1fb9a.firebasestorage.app'  # 🔁 Replace with your bucket
    })
def save_attendance_to_firebase(user_id, timestamp, photo_url=None, period="morning", status="on_time", minutes_offset=0):
    date_str = datetime.now().strftime("%Y-%m-%d")  # e.g., 2025-07-11
    ref = db.reference(f"attendance/{date_str}/{period}/{user_id}")
    ref.set({
        'user_id': user_id,
        'timestamp': timestamp,
        'photo_url': photo_url,
        'status': status,
        'minutes_offset': minutes_offset
    })


# def save_attendance_to_firebase(name, time_str, device="desktop"):
#     date_str = datetime.now().strftime("%Y-%m-%d")
#     ref = db.reference(f"attendance/{date_str}/{name}")
#     ref.set({
#         "time": time_str,
#         "device": device
#     })
