import streamlit as st
import cv2
import face_recognition
import numpy as np
import os
import pandas as pd
from datetime import datetime, date
from PIL import Image
import plotly.express as px

# Page config
st.set_page_config(page_title="Face Recognition Attendance", layout="wide")
st.title("📸 Face Recognition Attendance System - Snapshot Mode")

# Sidebar settings
st.sidebar.header("Settings")
known_faces_dir = st.sidebar.text_input(
    "Known Faces Directory",
    r"C:\Users\harsh\attendance_system\faces"
)

attendance_csv_dir = st.sidebar.text_input(
    "Daily Attendance CSV Folder",
    r"C:\Users\harsh\attendance_system\logs"
)


late_threshold_minutes = st.sidebar.number_input(
    "Late Threshold (minutes after 9:00 AM)", min_value=0, max_value=180, value=15
)

# Persistent state
if "attendance_log" not in st.session_state:
    st.session_state.attendance_log = {}

# Load known faces
@st.cache_data
def load_known_faces(known_faces_dir):
    encodings, names = [], []
    for file_name in os.listdir(known_faces_dir):
        if file_name.endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(known_faces_dir, file_name)
            image = face_recognition.load_image_file(path)
            try:
                encoding = face_recognition.face_encodings(image)[0]
            except IndexError:
                continue
            encodings.append(encoding)
            names.append(os.path.splitext(file_name)[0])
    return encodings, names

known_face_encodings, known_face_names = load_known_faces(known_faces_dir)

# Ensure logs directory exists
os.makedirs(attendance_csv_dir, exist_ok=True)

# CSV file for today
today_str = date.today().strftime("%Y-%m-%d")
csv_path = os.path.join(attendance_csv_dir, f"attendance_{today_str}.csv")

# Load existing CSV safely
if os.path.exists(csv_path):
    df_existing = pd.read_csv(csv_path)
    df_existing = df_existing.fillna('')
    st.session_state.attendance_log = {}
    for _, row in df_existing.iterrows():
        st.session_state.attendance_log[row["Name"]] = {
            "Arrival": row.get("Arrival", None),
            "Departure": row.get("Departure", None)
        }

# Columns layout
col1, col2 = st.columns([2, 3])

with col1:
    st.write("## Take Snapshot for Attendance")
    snapshot = st.camera_input("Click to capture image")

    if snapshot:
        frame = np.array(Image.open(snapshot))
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Detect faces
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            name = "Unknown"
            color = (0, 255, 0)
            if len(face_distances) > 0:
                best_idx = np.argmin(face_distances)
                if matches[best_idx]:
                    name = known_face_names[best_idx]
                    timestamp = datetime.now()
                    status = "On Time" if timestamp.hour < 9 or (timestamp.hour == 9 and timestamp.minute <= late_threshold_minutes) else "Late"

                    # --- FIXED: Arrival & Departure logic ---
                    if name not in st.session_state.attendance_log:
                        # First time: mark Arrival
                        st.session_state.attendance_log[name] = {"Arrival": f"{timestamp.strftime('%H:%M:%S')} ({status})", "Departure": None}
                    else:
                        # Mark Arrival if empty
                        if not st.session_state.attendance_log[name].get("Arrival"):
                            st.session_state.attendance_log[name]["Arrival"] = f"{timestamp.strftime('%H:%M:%S')} ({status})"
                        # Otherwise, mark Departure if empty
                        elif not st.session_state.attendance_log[name].get("Departure"):
                            st.session_state.attendance_log[name]["Departure"] = f"{timestamp.strftime('%H:%M:%S')}"

                    # Set color red if late
                    if st.session_state.attendance_log[name]["Arrival"] and isinstance(st.session_state.attendance_log[name]["Arrival"], str):
                        if "Late" in st.session_state.attendance_log[name]["Arrival"]:
                            color = (255, 0, 0)

            # Draw box
            cv2.rectangle(rgb_frame, (left, top), (right, bottom), color, 2)
            cv2.putText(rgb_frame, name, (left, top-10), cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 1)

        st.image(cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB), use_container_width=True)

    # --- Contact Me Section ---
    st.markdown("---")
    st.write("### Contact Me")
    st.write("**Harshita Gupta**")
    st.write("✉️ harshitaigcs@gmail.com")

with col2:
    if st.session_state.attendance_log:
        st.write("## Attendance Table")
        df = pd.DataFrame.from_dict(st.session_state.attendance_log, orient="index")
        st.dataframe(df)

        # Save CSV
        df_reset = df.reset_index().rename(columns={"index": "Name"})
        df_reset.to_csv(csv_path, index=False)

        # Download CSV
        csv = df_reset.to_csv(index=False).encode('utf-8')
        st.download_button("Download Attendance CSV", csv, f"attendance_{today_str}.csv", "text/csv")

        # --- Stats Charts ---
        st.write("## Attendance Stats")

        status_counts = {"On Time": 0, "Late": 0}
        arrival_times = []
        departure_times = []

        for record in st.session_state.attendance_log.values():
            # Arrival
            arrival = record.get("Arrival")
            if isinstance(arrival, str) and arrival:
                if "On Time" in arrival:
                    status_counts["On Time"] += 1
                elif "Late" in arrival:
                    status_counts["Late"] += 1
                hh_mm = arrival.split()[0].split(":")
                arrival_times.append(int(hh_mm[0])*60 + int(hh_mm[1]))
            # Departure
            departure = record.get("Departure")
            if isinstance(departure, str) and departure:
                hh_mm = departure.split(":")
                departure_times.append(int(hh_mm[0])*60 + int(hh_mm[1]))

        # On Time vs Late Bar Chart
        fig1 = px.bar(
            x=list(status_counts.keys()),
            y=list(status_counts.values()),
            color=list(status_counts.keys()),
            color_discrete_map={"On Time":"green","Late":"red"},
            labels={"x":"Status","y":"Count"},
            title="Arrival Status Count"
        )
        st.plotly_chart(fig1, use_container_width=True)

        # Arrival Trend Chart
        if arrival_times:
            arrival_hours = [t//60 + (t%60)/60 for t in arrival_times]
            fig2 = px.line(
                y=arrival_hours,
                x=list(st.session_state.attendance_log.keys())[:len(arrival_hours)],
                markers=True,
                labels={"x":"Name","y":"Arrival Time (Hours)"},
                title="Arrival Times Trend"
            )
            st.plotly_chart(fig2, use_container_width=True)

        # Departure Trend Chart
        if departure_times:
            departure_hours = [t/60 for t in departure_times]  # minutes → hours
            fig3 = px.line(
                y=departure_hours,
                x=list(st.session_state.attendance_log.keys())[:len(departure_hours)],
                markers=True,
                labels={"x":"Name","y":"Departure Time (Hours)"},
                title="Departure Times Trend"
            )
            st.plotly_chart(fig3, use_container_width=True)


# On a MacBook, if you open the Streamlit app in Chrome or Safari directly on your Mac, it should use your MacBook’s built-in webcam.
# dont use brave browser as it has issues with webcam permissions on macOS.

# You can now view your Streamlit app in your browser.
#   Local URL: http://localhost:8501
#   Network URL: http://192.168.1.26:8501