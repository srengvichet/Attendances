import plotly.express as px
import streamlit as st
import firebase_admin
from firebase_admin import credentials, db
from datetime import datetime
import pandas as pd

# --- Firebase Init ---
cred = credentials.Certificate("firebase_config.json")
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://sampleapp-1fb9a-default-rtdb.asia-southeast1.firebasedatabase.app/'
    })

st.set_page_config(page_title="📅 Attendance Reports", layout="wide")
st.title("📅 Daily & Monthly Attendance Reports")

report_type = st.radio("📁 Report Type", ["📅 Daily", "🗓️ Monthly"])

if report_type == "📅 Daily":
    selected_date = st.date_input("📅 Choose Date")
    date_str = selected_date.strftime("%Y-%m-%d")
    ref = db.reference(f"attendance/{date_str}")
    data = ref.get()

    if data:
        records = []
        for period in ["morning", "evening"]:
            period_data = data.get(period, {})
            for student, info in period_data.items():
                records.append({
                    "Name": student,
                    "Period": period,
                    "Time": info.get("timestamp"),
                    "Status": info.get("status"),
                    "Late Minutes": info.get("minutes_offset", 0) if info.get("status") == "late" else 0
                })

        df = pd.DataFrame(records).sort_values(["Period", "Name"])
        st.write(f"📊 Total records: {len(df)}")
        st.dataframe(df)

        if not df.empty:
            # --- Late minutes bar chart ---
            late_df = df[df["Status"] == "late"]
            if not late_df.empty:
                fig = px.bar(late_df, x="Name", y="Late Minutes", color="Period", title="⏱️ Late Minutes Per Student")
                st.plotly_chart(fig, use_container_width=True)

            # --- Status Pie Chart ---
            status_counts = df["Status"].value_counts().reset_index()
            status_counts.columns = ["Status", "Count"]
            pie_fig = px.pie(status_counts, values="Count", names="Status", title="📊 Attendance Status Distribution")
            st.plotly_chart(pie_fig, use_container_width=True)


        # Export CSV
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download CSV", csv, f"attendance_{date_str}.csv", "text/csv")

    else:
        st.warning("No attendance data found for this date.")

elif report_type == "🗓️ Monthly":
    selected_date = st.date_input("📆 Select a day in the month")
    month_prefix = selected_date.strftime("%Y-%m")
    all_ref = db.reference("attendance")
    all_data = all_ref.get()

    if all_data:
        records = []
        for date_key, periods in all_data.items():
            if date_key.startswith(month_prefix):
                for period in ["morning", "evening"]:
                    period_data = periods.get(period, {})
                    for student, info in period_data.items():
                        records.append({
                            "Date": date_key,
                            "Name": student,
                            "Period": period,
                            "Status": info.get("status"),
                            "Late Minutes": info.get("minutes_offset", 0) if info.get("status") == "late" else 0
                        })

        df = pd.DataFrame(records).sort_values(["Date", "Period", "Name"])
        st.write(f"📊 Attendance records in {month_prefix}")
        st.dataframe(df)

        if not df.empty:
            # --- Daily late minutes trend ---
            late_df = df[df["Status"] == "late"]
            if not late_df.empty:
                summary = late_df.groupby("Date")["Late Minutes"].sum().reset_index()
                fig_line = px.line(summary, x="Date", y="Late Minutes", title="📈 Total Late Minutes Per Day")
                st.plotly_chart(fig_line, use_container_width=True)

            # --- Daily count of late scans ---
            count_df = late_df.groupby(["Date"]).size().reset_index(name="Late Count")
            fig_bar = px.bar(count_df, x="Date", y="Late Count", title="📊 Daily Late Check-In Count")
            st.plotly_chart(fig_bar, use_container_width=True)

        # Export CSV
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Monthly CSV", csv, f"monthly_report_{month_prefix}.csv", "text/csv")

    else:
        st.warning("Attendance DB is empty.")
