import json

import streamlit as st
import firebase_admin
from firebase_admin import credentials, db
from datetime import datetime
from io import BytesIO
from PIL import Image
import requests
from math import ceil
import streamlit.components.v1 as components  # at the top if not already added

# --- Init ---
st.set_page_config(page_title="📊 Dashboard", layout="wide")
st.title("📊 Daily Attendance Overview")

# --- Firebase ---
cred = credentials.Certificate(json.loads(st.secrets["FIREBASE_CONFIG_JSON"]))
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://sampleapp-1fb9a-default-rtdb.asia-southeast1.firebasedatabase.app/'
    })

# --- Date Selector ---
selected_date = st.date_input("📅 Select Date", datetime.today())
date_str = selected_date.strftime("%Y-%m-%d")

# --- Fetch Attendance Data ---
attendance_ref = db.reference(f"attendance/{date_str}")
attendance_data = attendance_ref.get()

if not attendance_data:
    st.warning("🚫 No attendance data found for this date.")
    st.stop()
#display student attendance with Student Card Layout with loading image from Firebase
# --- Display Grid of Cards ---
def display_attendance_grid(attendance_dict):
    students = list(attendance_dict.items())
    num_cols = 3
    rows = ceil(len(students) / num_cols)

    for row in range(rows):
        cols = st.columns(num_cols)
        for i in range(num_cols):
            idx = row * num_cols + i
            if idx >= len(students):
                break

            student_id, info = students[idx]
            timestamp = info.get("timestamp", "N/A")
            status = info.get("status", "N/A")
            minutes_offset = info.get("minutes_offset", 0)

            # Get user profile from /students
            student_ref = db.reference(f"students/{student_id}").get()
            avatar_url = None
            full_name = student_id  # fallback

            if student_ref:
                full_name = student_ref.get("full_name", student_id)
                avatar_url = student_ref.get("photo_url")

            # Fallback to attendance snapshot
            if not avatar_url:
                avatar_url = info.get("photo_url")

            # Construct avatar HTML
            avatar_html = f"""
                <div style="
                    width: 120px;
                    height: 140px;
                    margin: 0 auto 10px;
                    border-radius: 10px;
                    overflow: hidden;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    background-color: #f0f0f0;
                    box-shadow: 0 2px 6px rgba(0,0,0,0.1);
                ">
                    <img src="{avatar_url}" style="
                        width: 100%;
                        height: 100%;
                        object-fit: cover;
                    " />
                </div>
            """ if avatar_url else "<div style='font-size:40px;'>🧑</div>"

            # Construct status display
            if status == "late":
                status_html = f"""
                    <p style='color:orange;'>📌 <b>LATE</b></p>
                    <p style='color:red;'>⚠️ {minutes_offset} min late</p>
                """
            elif status == "on_time":
                status_html = """
                    <p style='color:green;'>📌 <b>ON TIME</b></p>
                    <p style='color:green;'>✅ On time</p>
                """
            else:
                status_html = f"<p style='color:gray;'>📌 <b>{status.upper()}</b></p>"

            # Build final card HTML
            card_html = f"""
                <div style='
                    background-color: #fff;
                    border-radius: 12px;
                    padding: 15px;
                    margin-bottom: 10px;
                    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
                    text-align: center;
                    border: 1px solid #e0e0e0;
                '>
                    {avatar_html}
                    <h4 style='margin: 10px 0 5px;'>{full_name}</h4>
                    <p style='margin: 5px 0; font-size: 14px;'>🕒 {timestamp}</p>
                    {status_html}
                </div>
            """

            with cols[i]:
                with st.container():
                    components.html(f"""
                    <div style='
                        box-sizing: border-box;
                        padding: 10px;
                        width: 100%;
                        max-width: 100%;
                    '>
                        <div style='
                            background-color: #fff;
                            border-radius: 12px;
                            padding: 12px;
                            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
                            text-align: center;
                            border: 1px solid #e0e0e0;
                            box-sizing: border-box;
                            width: 100%;
                            max-width: 100%;
                        '>
                        <h2 style='margin: 10px auto 8px;'>{full_name}</h2>
                            <div style='
                                width: 120px;
                                height: 160px;
                                margin: 0 auto 10px;
                                border-radius: 10px;
                                overflow: hidden;
                                background-color: #f0f0f0;
                            '>
                                <img src="{avatar_url}" style='
                                    width: 100%;
                                    height: 100%;
                                    object-fit: cover;
                                ' />
                            </div>
                            
                            <p style='margin: 4px 0; font-size: 13px;'>🕒 {timestamp}</p>
                            {"<p style='color:orange;'>📌 <b>LATE</b></p><p style='color:red;'>⚠️ " + str(minutes_offset) + " min late</p>" if status == "late" else ""}
                            {"<p style='color:green;'>📌 <b>ON TIME</b></p><p style='color:green;'>✅ On time</p>" if status == "on_time" else ""}
                        </div>
                    </div>
                    """, height=400)


# def display_attendance_grid(attendance_dict):
#     students = list(attendance_dict.items())
#     num_cols = 3
#     rows = ceil(len(students) / num_cols)
#
#     for row in range(rows):
#         cols = st.columns(num_cols)
#         for i in range(num_cols):
#             idx = row * num_cols + i
#             if idx >= len(students):
#                 break
#
#             student_id, info = students[idx]
#             photo_url = info.get("photo_url")
#             timestamp = info.get("timestamp", "N/A")
#             status = info.get("status", "N/A")
#             minutes_offset = info.get("minutes_offset", 0)
#
#             # Avatar HTML
#             avatar_html = f"<img src='{photo_url}' width='120' style='border-radius:10px;'>" if photo_url else "<div style='font-size:40px;'>🧑</div>"
#
#             # Status HTML
#             if status == "late":
#                 status_html = f"<p style='color:orange;'>📌 <b>{status.upper()}</b></p><p style='color:red;'>⚠️ {minutes_offset} min late</p>"
#             elif status == "on_time":
#                 status_html = f"<p style='color:green;'>📌 <b>{status.upper()}</b></p><p style='color:green;'>✅ On time</p>"
#             else:
#                 status_html = f"<p style='color:gray;'>📌 <b>{status.upper()}</b></p>"
#
#             # Final card HTML
#             card_html = f"""
#                 <div style='
#                     background-color: #fff;
#                     border-radius: 12px;
#                     padding: 20px;
#                     margin-bottom: 20px;
#                     box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
#                     text-align: center;
#                     border: 1px solid #e0e0e0;
#                 '>
#                     {avatar_html}
#                     <h4 style='margin: 10px 0 5px;'>{student_id}</h4>
#                     <p style='margin: 5px 0;'>🕒 {timestamp}</p>
#                     {status_html}
#                 </div>
#             """
#
#             with cols[i]:
#                 st.markdown(card_html, unsafe_allow_html=True)



# --- Helper: display student cards in rows ---
# def display_attendance_grid(attendance_dict):
#     students = list(attendance_dict.items())
#     num_cols = 3  # students per row
#     rows = ceil(len(students) / num_cols)
#
#     for row in range(rows):
#         cols = st.columns(num_cols)
#         for i in range(num_cols):
#             idx = row * num_cols + i
#             if idx >= len(students):
#                 break
#             student, info = students[idx]
#             photo_url = info.get("photo_url")
#             timestamp = info.get("timestamp", "N/A")
#             status = info.get("status", "N/A")
#             minutes_offset = info.get("minutes_offset", 0)
#
#             with cols[i]:
#                 # Avatar
#                 try:
#                     response = requests.get(photo_url)
#                     img = Image.open(BytesIO(response.content))
#                     st.image(img, width=140)
#                 except:
#                     st.warning("❌ No image")
#
#                 # Info
#                 st.markdown(f"**🧑 {student}**")
#                 st.markdown(f"🕒 `{timestamp}`  \n📌 **{status.upper()}**")
#                 if status == "late":
#                     st.warning(f"⚠️ `{minutes_offset}` min late")
#                 elif status == "on_time":
#                     st.success("✅ On time")

# --- Morning and Evening Data ---
morning_data = attendance_data.get("morning", {})
evening_data = attendance_data.get("evening", {})

# --- Columns Morning | Evening ---
col_morning, col_evening = st.columns(2)

# --- Morning Attendance ---
with col_morning:
    st.subheader("☀️ Morning Attendance")
    if morning_data:
        display_attendance_grid(morning_data)
    else:
        st.info("📭 No one attended in the morning.")

# --- Evening Attendance ---
with col_evening:
    st.subheader("🌙 Evening Attendance")
    if evening_data:
        display_attendance_grid(evening_data)
    else:
        st.info("📭 No one attended in the evening.")

# # Dashboard.py
#
# import streamlit as st
# import firebase_admin
# from firebase_admin import credentials, db
# from datetime import datetime
# import time
# import streamlit as st
#
# st.set_page_config(page_title="📊 Dashboard", layout="wide")
# st.title("📊 Attendance Dashboard")
#
#
# # --- Firebase Initialization ---
# cred = credentials.Certificate("firebase_config.json")
# if not firebase_admin._apps:
#     firebase_admin.initialize_app(cred, {
#         'databaseURL': 'https://sampleapp-1fb9a-default-rtdb.asia-southeast1.firebasedatabase.app/'  # replace with your project URL
#     })
#
# # --- Streamlit UI ---
# st.set_page_config(page_title="📊 Face Attendance Dashboard", layout="wide")
# st.title("📊 Real-Time Attendance Dashboard")
#
# # --- Date Selection ---
# selected_date = st.date_input("📅 Select Date", datetime.today())
# date_str = selected_date.strftime("%Y-%m-%d")
#
# # --- Auto-refresh every 10 seconds ---
# st_autorefresh = st.experimental_rerun if st.query_params.get("refresh") else lambda: None
# st_autorefresh()
# time.sleep(10)
#
# # --- Firebase Attendance Fetch ---
# ref = db.reference(f"attendance/{date_str}")
# data = ref.get()
#
# # --- Display ---
# if data:
#     for name, info in data.items():
#         with st.container():
#             st.markdown(f"### 🧑 {name}")
#             st.markdown(f"- 🕒 Time: `{info.get('time', 'N/A')}`")
#             st.markdown(f"- 📱 Device: `{info.get('device', 'unknown')}`")
#             st.markdown("---")
# else:
#     st.warning("🚫 No attendance data found for this date.")
