import cv2
from super_gradients.training import models
from deep_sort_realtime.deepsort_tracker import DeepSort
from parking import ParkingManagement
import numpy as np
import sys
import os

# Default paths - UPDATED TO YOUR SPECIFIED PATHS
DEFAULT_VIDEO_PATH = "cnight.mp4"
DEFAULT_JSON = "parking_slots_7000.json"

# Use the default paths directly without argument parsing
video_path = DEFAULT_VIDEO_PATH
json_path = DEFAULT_JSON

if not os.path.exists(video_path):
    print(f"Error: Video file '{video_path}' not found!")
    sys.exit(1)

if not os.path.exists(json_path):
    print(f"Error: JSON file '{json_path}' not found!")
    sys.exit(1)

MODEL_NAME = "yolo_nas_l"
USED_CLASSES = [2, 3]  # car, motorcycle

print(f"Using video: {video_path}")
print(f"Using JSON: {json_path}")

print("Loading YOLO-NAS model...")
model = models.get(MODEL_NAME, pretrained_weights="coco")
print("Loading DeepSort tracker...")
tracker = DeepSort(max_age=50, n_init=3, max_cosine_distance=0.4, nn_budget=None)

try:
    print("Initializing parking manager...")
    parking_manager = ParkingManagement(json_path, slot_line_thickness=2)
    print(f"Loaded {len(parking_manager.slot_list)} parking slots")
except Exception as e:
    print(f"Error initializing parking manager: {str(e)}")
    sys.exit(1)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Could not open video file: {video_path}")
    sys.exit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video Properties: {frame_width}x{frame_height} @ {fps:.2f} FPS")
print(f"Total frames: {total_frames}")

# Function to reset video to beginning
def reset_video():
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return 0

frame_count = reset_video()
loop_count = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video reached. Restarting from beginning...")
            frame_count = reset_video()
            loop_count += 1
            parking_manager.reset_occupancy_buffers()  # Reset buffers for new loop
            continue

        frame_count += 1

        # ✅ Speed up playback 24x (skip frames)
        if frame_count % 24 != 0:
            continue

        if frame_count % 240 == 0:
            print(f"Processing frame {frame_count}/{total_frames} (Loop {loop_count})")

        # ✅ YOLO-NAS Inference with LOWER confidence (20%)
        preds = model.predict(frame, conf=0.20)
        detections = []

        pred = preds.prediction
        for i in range(len(pred.bboxes_xyxy)):
            x1, y1, x2, y2 = [int(v) for v in pred.bboxes_xyxy[i]]
            conf = float(pred.confidence[i])
            cls = int(pred.labels[i])

            if cls in USED_CLASSES and x2 > x1 and y2 > y1:
                detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls))

        # DeepSORT tracking
        tracks = tracker.update_tracks(detections, frame=frame)

        vis_frame, info = parking_manager.process_data(frame, tracks)

        # ✅ Only show Vacant/Occupied counts
        summary = f"Vacant: {info['Available']} | Occupied: {info['Occupied']} | Loop: {loop_count}"

        overlay_height = 50
        cv2.rectangle(vis_frame, (10, 10), (len(summary) * 12 + 20, overlay_height), (0, 0, 0), -1)
        cv2.rectangle(vis_frame, (10, 10), (len(summary) * 12 + 20, overlay_height), (255, 255, 255), 2)
        cv2.putText(vis_frame, summary, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        timestamp = f"Frame: {frame_count}/{total_frames}"
        cv2.putText(vis_frame, timestamp, (20, frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.namedWindow("Parking Lot Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Parking Lot Detection", 1280, 720)
        cv2.imshow("Parking Lot Detection", vis_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            cv2.imwrite(f"frame_{frame_count}_loop_{loop_count}.jpg", frame)
            print(f"Saved frame_{frame_count}_loop_{loop_count}.jpg")

except KeyboardInterrupt:
    print("\nProcessing interrupted by user")
except Exception as e:
    print(f"Error during processing: {str(e)}")

cap.release()
cv2.destroyAllWindows()
print(f"Processing completed. Processed {loop_count} loops and {frame_count} frames")