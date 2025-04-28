import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load YOLOv8n model
model = YOLO('yolov8n.pt')
tracker = DeepSort(max_age=15)

# Open input video
cap = cv2.VideoCapture(r"C:\Users\ASUS\Downloads\input_video.mp4")

# Output video setup
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter(r"C:\Users\ASUS\Downloads\output_video.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

prev_ids = set()
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Skip frames
    if frame_count % 3 != 0:
        continue

    # Resize smaller for YOLO input
    resized_frame = cv2.resize(frame, (640, 360))
    results = model(resized_frame)[0]

    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = model.names[cls]

        # Scale bounding boxes back to original size
        x_scale = width / 640
        y_scale = height / 360
        x1 = int(x1 * x_scale)
        x2 = int(x2 * x_scale)
        y1 = int(y1 * y_scale)
        y2 = int(y2 * y_scale)

        detections.append(([x1, y1, x2 - x1, y2 - y1], conf, label))

    tracks = tracker.update_tracks(detections, frame=frame)
    current_ids = set()

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        bbox = track.to_ltrb()
        label = track.get_det_class()

        current_ids.add(track_id)

        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'{label} ID: {track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    prev_ids = current_ids.copy()
    out.write(frame)

    # Free memory
    del results

cap.release()
out.release()
cv2.destroyAllWindows()

print("✅ Finished processing video successfully!")
