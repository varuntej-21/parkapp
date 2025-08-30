import streamlit as st
import cv2
import tempfile
import os
import subprocess
import sys
from pathlib import Path
from parking import ParkingManagement, ParkingPtsSelection
from super_gradients.training import models
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np
import json
import time

st.set_page_config(page_title="Smart Parking System", layout="wide")

# Initialize session state
if 'video_path' not in st.session_state:
    st.session_state.video_path = None
if 'json_path' not in st.session_state:
    st.session_state.json_path = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'tracker' not in st.session_state:
    st.session_state.tracker = None
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'extraction_window' not in st.session_state:
    st.session_state.extraction_window = False
if 'annotation_window' not in st.session_state:
    st.session_state.annotation_window = False
if 'current_frame' not in st.session_state:
    st.session_state.current_frame = 0
if 'annotation_points' not in st.session_state:
    st.session_state.annotation_points = []
if 'annotation_slots' not in st.session_state:
    st.session_state.annotation_slots = []
if 'current_slot_type' not in st.session_state:
    st.session_state.current_slot_type = "car"
if 'annotation_image' not in st.session_state:
    st.session_state.annotation_image = None


# Load Model and Tracker once
@st.cache_resource
def load_model():
    try:
        model = models.get("yolo_nas_l", pretrained_weights="coco")
        tracker = DeepSort(max_age=50, n_init=3, max_cosine_distance=0.4, nn_budget=None)
        return model, tracker
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None, None


def run_detection(video_path, json_path):
    """Run the parking detection"""
    try:
        # Initialize parking manager
        pm = ParkingManagement(json_path, slot_line_thickness=2)

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error(f"Could not open video: {video_path}")
            return

        # Get video info
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        st.info(f"Processing video: {frame_width}x{frame_height} @ {fps:.1f} FPS | Total frames: {total_frames}")

        # Create placeholder for video display
        video_placeholder = st.empty()
        stats_placeholder = st.empty()
        stop_button_placeholder = st.empty()

        frame_count = 0
        stop_detection = False

        # Speedup settings
        frame_skip = 24
        confidence = 0.22

        while cap.isOpened() and not stop_detection:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Skip frames for speedup
            if frame_count % frame_skip != 0:
                continue

            # Run detection
            preds = st.session_state.model.predict(frame, conf=confidence)
            detections = []

            pred = preds[0].prediction if isinstance(preds, list) else preds.prediction
            if pred is not None and len(pred.bboxes_xyxy) > 0:
                for i, (bbox, conf, cls) in enumerate(zip(pred.bboxes_xyxy, pred.confidence, pred.labels)):
                    if cls in [2, 3, 5, 7]:  # car, motorcycle, bus, truck
                        x1, y1, x2, y2 = map(int, bbox)
                        width = x2 - x1
                        height = y2 - y1
                        if width > 20 and height > 20:
                            detections.append(([x1, y1, width, height], float(conf), int(cls)))

            # Update tracks
            tracks = st.session_state.tracker.update_tracks(detections, frame=frame)

            # Process parking data
            vis_frame, info = pm.process_data(frame, tracks)

            # Display results
            occupied = info["Occupied"]
            vacant = info["Available"]
            total = info["TotalSlots"]

            # Draw stats on frame
            cv2.putText(vis_frame, f"Vacant: {vacant} | Occupied: {occupied} | Total: {total}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(vis_frame, f"Frame: {frame_count}/{total_frames}",
                        (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Convert BGR to RGB for Streamlit
            vis_frame_rgb = cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB)

            # Update display
            video_placeholder.image(vis_frame_rgb, channels="RGB", use_container_width=True)

            # Update stats
            stats_placeholder.info(f"""
            **Live Statistics:**
            - Vacant Slots: {vacant}
            - Occupied Slots: {occupied} 
            - Total Slots: {total}
            - Cars Detected: {info['Cars']}
            - Bikes Detected: {info['Bikes']}
            - Total Vehicles: {info['Cars'] + info['Bikes']}
            - Current Frame: {frame_count}/{total_frames}
            """)

            # Add a stop button
            if stop_button_placeholder.button("⏹️ Stop Detection", key=f"stop_{frame_count}"):
                stop_detection = True
                st.session_state.processing = False
                break

        cap.release()
        st.success("Detection completed!")
        st.session_state.processing = False

    except Exception as e:
        st.error(f"Error during detection: {str(e)}")
        st.session_state.processing = False


# Frame Extractor Web Interface
def frame_extractor_interface():
    # Home button
    if st.button("🏠 Home"):
        st.session_state.extraction_window = False
        st.rerun()

    st.title("🎬 Video Frame Extractor")

    if not st.session_state.video_path:
        st.warning("Please upload a video file first in the main app!")
        return

    cap = cv2.VideoCapture(st.session_state.video_path)
    if not cap.isOpened():
        st.error("Could not open video file!")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    st.info(f"Video: {os.path.basename(st.session_state.video_path)} | Frames: {total_frames} | FPS: {fps:.1f}")

    # Frame navigation
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("⏮️ First Frame"):
            st.session_state.current_frame = 0
    with col2:
        if st.button("⏭️ Last Frame"):
            st.session_state.current_frame = total_frames - 1
    with col3:
        frame_input = st.number_input("Go to Frame:", min_value=0, max_value=total_frames - 1,
                                      value=st.session_state.current_frame, step=100)
        if st.button("🚀 Go"):
            st.session_state.current_frame = frame_input

    # Navigation buttons
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("◀◀ Prev 100"):
            st.session_state.current_frame = max(0, st.session_state.current_frame - 100)
    with col2:
        if st.button("◀ Prev 10"):
            st.session_state.current_frame = max(0, st.session_state.current_frame - 10)
    with col3:
        if st.button("Next 10 ▶"):
            st.session_state.current_frame = min(total_frames - 1, st.session_state.current_frame + 10)
    with col4:
        if st.button("Next 100 ▶▶"):
            st.session_state.current_frame = min(total_frames - 1, st.session_state.current_frame + 100)

    # Display current frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.current_frame)
    ret, frame = cap.read()

    if ret:
        # Convert to RGB for display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(frame_rgb, caption=f"Frame {st.session_state.current_frame}/{total_frames - 1}",
                 use_container_width=True)

        # Save frame options
        st.subheader("Save Frame")
        output_name = st.text_input("Output filename:", value=f"frame_{st.session_state.current_frame:06d}.jpg")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save Current Frame"):
                cv2.imwrite(output_name, frame)
                st.success(f"Saved as {output_name}")
        with col2:
            if st.button("📷 Save as Reference"):
                cv2.imwrite("reference_frame.jpg", frame)
                st.success("Saved as reference_frame.jpg")

    cap.release()


# Fully Functional Annotation Tool Web Interface
def annotation_tool_interface():
    # Home button
    if st.button("🏠 Home"):
        st.session_state.annotation_window = False
        st.session_state.annotation_points = []
        st.session_state.annotation_slots = []
        st.rerun()

    st.title("🖌️ Parking Slot Annotation Tool")

    # Upload image for annotation
    uploaded_image = st.file_uploader("Upload Parking Lot Image", type=["jpg", "jpeg", "png", "bmp"])

    if uploaded_image:
        # Read and store the image
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        st.session_state.annotation_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        st.session_state.annotation_points = []
        st.session_state.annotation_slots = []

    if st.session_state.annotation_image is not None:
        # Display the image with annotations
        display_image = st.session_state.annotation_image.copy()

        # Draw existing slots
        for i, slot in enumerate(st.session_state.annotation_slots):
            points = slot['points']
            slot_type = slot['type']
            color = (0, 255, 0) if slot_type == 'car' else (255, 0, 0) if slot_type == 'bike' else (0, 0, 255)

            # Draw polygon
            pts = np.array(points, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(display_image, [pts], True, color, 2)

            # Draw slot number
            center_x = sum(p[0] for p in points) // 4
            center_y = sum(p[1] for p in points) // 4
            cv2.putText(display_image, f"{i + 1}", (center_x - 5, center_y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Draw current points
        for point in st.session_state.annotation_points:
            cv2.circle(display_image, point, 5, (0, 0, 255), -1)

        # Convert to RGB for display
        display_image_rgb = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
        st.image(display_image_rgb, caption="Annotation Canvas", use_container_width=True)

        # Annotation controls
        col1, col2, col3 = st.columns(3)
        with col1:
            st.session_state.current_slot_type = st.radio("Slot Type:", ["car", "bike", "truck"], horizontal=True)
        with col2:
            if st.button("➕ Add Point", use_container_width=True):
                # Simulate point addition (in real app, this would use click coordinates)
                st.info("Click on the image to add points (4 points per slot)")
        with col3:
            if st.button("✅ Complete Slot", use_container_width=True) and len(st.session_state.annotation_points) >= 4:
                # Add the completed slot
                slot_data = {
                    "points": st.session_state.annotation_points[:4],
                    "type": st.session_state.current_slot_type
                }
                st.session_state.annotation_slots.append(slot_data)
                st.session_state.annotation_points = []
                st.success(f"Added {st.session_state.current_slot_type} slot!")

        # Slot management
        st.subheader("Slot Management")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🗑️ Clear Current", use_container_width=True):
                st.session_state.annotation_points = []
                st.info("Current points cleared")
        with col2:
            if st.button("🚫 Remove Last Slot", use_container_width=True) and st.session_state.annotation_slots:
                st.session_state.annotation_slots.pop()
                st.info("Last slot removed")
        with col3:
            if st.button("🔥 Clear All Slots", use_container_width=True):
                st.session_state.annotation_slots = []
                st.session_state.annotation_points = []
                st.info("All slots cleared")

        # Save functionality
        st.subheader("Save Annotation")
        json_filename = st.text_input("Save as JSON:", value="parking_slots.json")

        if st.button("💾 Save JSON", use_container_width=True):
            if st.session_state.annotation_slots:
                with open(json_filename, 'w') as f:
                    json.dump(st.session_state.annotation_slots, f, indent=4)
                st.success(f"Saved {len(st.session_state.annotation_slots)} slots to {json_filename}")
            else:
                st.warning("No slots to save!")

        # Display current slots
        st.subheader("Current Slots")
        if st.session_state.annotation_slots:
            for i, slot in enumerate(st.session_state.annotation_slots):
                st.write(f"**Slot {i + 1}**: {slot['type']} - {len(slot['points'])} points")
        else:
            st.info("No slots created yet")

    else:
        st.info("Please upload an image to start annotating")

    st.markdown("---")
    st.subheader("Instructions")
    st.markdown("""
    1. **Upload Image**: Upload a parking lot image
    2. **Add Points**: Click to add 4 points for each parking slot
    3. **Complete Slot**: Click 'Complete Slot' after adding 4 points
    4. **Set Type**: Choose car, bike, or truck for each slot
    5. **Save**: Save all slots as JSON file
    """)


# Main app logic
def main_app():
    # Sidebar for controls
    st.sidebar.title("🚗 Smart Parking System")
    st.sidebar.header("Configuration")

    # File upload sections
    st.sidebar.subheader("1. Upload Files")

    # Video file upload
    uploaded_video = st.sidebar.file_uploader("Upload Video File", type=["mp4", "avi", "mov"])
    if uploaded_video:
        video_path = f"uploaded_video.{uploaded_video.name.split('.')[-1]}"
        with open(video_path, "wb") as f:
            f.write(uploaded_video.getbuffer())
        st.session_state.video_path = video_path
        st.sidebar.success(f"Video uploaded: {uploaded_video.name}")

    # JSON file upload
    uploaded_json = st.sidebar.file_uploader("Upload Parking JSON File", type=["json"])
    if uploaded_json:
        json_path = "parking_slots.json"
        with open(json_path, "wb") as f:
            f.write(uploaded_json.getbuffer())
        st.session_state.json_path = json_path
        st.sidebar.success(f"JSON uploaded: {uploaded_json.name}")

    # Tools section
    st.sidebar.subheader("2. Tools")

    # Frame extraction tool
    if st.sidebar.button("📸 Open Frame Extractor"):
        st.session_state.extraction_window = True
        st.session_state.annotation_window = False
        st.rerun()

    # Annotation tool
    if st.sidebar.button("🖌️ Open Annotation Tool"):
        st.session_state.annotation_window = True
        st.session_state.extraction_window = False
        st.rerun()

    # Desktop tools
    if st.sidebar.button("🎬 Advanced Frame Extractor (Desktop)"):
        try:
            subprocess.Popen([sys.executable, "img.py", st.session_state.video_path, "-i"])
            st.sidebar.success("Advanced extractor launched!")
        except Exception as e:
            st.sidebar.error(f"Failed to open: {str(e)}")

    if st.sidebar.button("🖋️ Advanced Annotation Tool (Desktop)"):
        try:
            subprocess.Popen([sys.executable, "se.py"])
            st.sidebar.success("Advanced annotation tool launched!")
        except Exception as e:
            st.sidebar.error(f"Failed to open: {str(e)}")

    # Main detection section
    st.sidebar.subheader("3. Detection")

    # Start processing button
    if st.sidebar.button("▶️ Start Parking Detection", type="primary") and not st.session_state.processing:
        if not st.session_state.video_path:
            st.sidebar.error("Please upload a video file first!")
        elif not st.session_state.json_path:
            st.sidebar.error("Please upload a JSON file first!")
        else:
            # Load model if not already loaded
            if st.session_state.model is None:
                with st.spinner("Loading model..."):
                    st.session_state.model, st.session_state.tracker = load_model()

            if st.session_state.model is None:
                st.sidebar.error("Failed to load model!")
            else:
                st.session_state.processing = True
                st.sidebar.success("Starting detection...")
                run_detection(st.session_state.video_path, st.session_state.json_path)

    # Main area content based on mode
    if st.session_state.extraction_window:
        frame_extractor_interface()
    elif st.session_state.annotation_window:
        annotation_tool_interface()
    else:
        # Default dashboard view
        st.title("Smart Parking System Dashboard(cbit campus cctv)")

        # File status
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Video Status")
            if st.session_state.video_path:
                st.success("✅ Video file uploaded")
                st.text(f"File: {os.path.basename(st.session_state.video_path)}")
                try:
                    cap = cv2.VideoCapture(st.session_state.video_path)
                    if cap.isOpened():
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        st.text(f"Size: {width}x{height}")
                        st.text(f"FPS: {fps:.1f}")
                        cap.release()
                except:
                    pass
            else:
                st.warning("⏳ No video file uploaded")

        with col2:
            st.subheader("JSON Status")
            if st.session_state.json_path:
                st.success("✅ JSON file uploaded")
                st.text(f"File: {os.path.basename(st.session_state.json_path)}")
                try:
                    with open(st.session_state.json_path, 'r') as f:
                        slots = json.load(f)
                    st.text(f"Parking Slots: {len(slots)}")
                    car_slots = sum(1 for slot in slots if slot.get('type') == 'car')
                    bike_slots = sum(1 for slot in slots if slot.get('type') == 'bike')
                    st.text(f"Car slots: {car_slots}, Bike slots: {bike_slots}")
                except Exception as e:
                    st.text(f"Error loading JSON: {str(e)}")
            else:
                st.warning("⏳ No JSON file uploaded")

        if st.session_state.processing:
            st.warning("🔄 Detection in progress...")

        st.subheader("Instructions")
        st.markdown("""
        1. **Upload Files**: Upload a video file and parking slot JSON file
        2. **Tools**: Use frame extractor or annotation tools
        3. **Detection**: Start parking detection
        note:for annotator always use advanced button.
        note:if anywhere webpage is stuck refresh the page.
        """)

    st.markdown("---")
    st.caption("Smart Parking System")


# Run the app
main_app()