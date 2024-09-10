import cv2
from ultralytics import YOLO
import argparse
import os
import numpy as np
from yolo_classes import yolo_classes

# Postavljanje argument parsera
parser = argparse.ArgumentParser(description="YOLOv8 distanca od kamere za video sa praćenjem")
parser.add_argument('--model', type=str, required=True, help="Putanja do YOLOv8 modela")
parser.add_argument('--video', type=str, required=True, help="Putanja do video zapisa")
parser.add_argument('--output', type=str, default='output_video.mp4', help="Putanja do izlaznog video zapisa")
parser.add_argument('--min_confidence', type=float, default=0.7, help="Minimalna pouzdanost (confidence) za detekcije")
args = parser.parse_args()

# Provjera da li video postoji
if not os.path.exists(args.video):
    print(f"Video ne postoji: '{args.video}'.")
    exit()

# Uvezi YOLO model
model = YOLO(args.model)

# Funkcija za dobijanje boje za klasu
def get_class_color(class_id):
    np.random.seed(class_id)
    color = tuple(int(x) for x in np.random.randint(0, 255, 3))
    return color

# Kalkulacija relativne pozicije
def relative_position(y, picture_height):
    return y * (304 / picture_height)

# Funkcija za izračun udaljenosti
def calculate_distance(relative_y):
    try:
        return 2.8921 * 1.0238 ** relative_y
    except OverflowError as e:
        print(f"Greška u izračunavanju udaljenosti: {e}")
        return float('inf')  # Vratiti beskonačnost ako dođe do greške

# Funkcija za izračun brzine
def calculate_velocity(prev_center, curr_center, frame_time):
    if frame_time == 0:
        return (0, 0)
    dx = curr_center[0] - prev_center[0]
    dy = curr_center[1] - prev_center[1]
    velocity = (dx / frame_time, dy / frame_time)
    print(f"Prev Center: {prev_center}, Curr Center: {curr_center}, Velocity: {velocity}")  # Debug print
    return velocity

# Funkcija za kreiranje trackera
def create_tracker(tracker_type, frame, bbox):
    if tracker_type == 'KCF':
        tracker = cv2.TrackerKCF_create()
    elif tracker_type == 'CSRT':
        tracker = cv2.TrackerCSRT_create()
    else:
        tracker = cv2.TrackerKCF_create()  # Default tracker
    tracker.init(frame, bbox)
    return tracker

# Funkcija za predikciju i crtanje udaljenosti i brzine na frame-ove
def process_video(input_video_path, output_video_path):
    # Učitaj video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Greška pri otvaranju video zapisa: '{input_video_path}'.")
        return

    # Dohvati parametre video zapisa
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Odaberite odgovarajući codec
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_time = 1 / fps

    # Kreiraj VideoWriter za snimanje obrađenog video zapisa
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Kreiraj dictionary za praćenje prethodnih pozicija objekata
    prev_positions = {}
    trackers = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Izvrši predikciju
        results = model.predict(frame, imgsz=640, save=False, conf=args.min_confidence)

        # Debug print za rezultate
        print(f"Broj rezultata: {len(results)}")

        detected_objects = []
        detected_ids = set()  # Čuvanje ID-ova trenutno detektovanih objekata

        for result in results:
            boxes = result.boxes.xywh.cpu().numpy()  # Uzmemo xywh koordinate
            classes = result.boxes.cls.cpu().numpy().astype(int)  # Klase objekata

            if len(boxes) == 0:
                print("Nema detekcija u trenutnom frame-u")

            for box, class_id in zip(boxes, classes):
                x_center, y_center, bbox_width, bbox_height = box[:4]  # Izvučemo koordinate i dimenzije

                # Izračunaj xmin, ymin, xmax, ymax
                x_min = int(x_center - bbox_width / 2)
                y_min = int(y_center - bbox_height / 2)
                x_max = int(x_center + bbox_width / 2)
                y_max = int(y_center + bbox_height / 2)

                # Debug print za koordinate
                print(f"Detekcija: x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}, ID={class_id}")

                # Izračunaj visinu objekta u pikselima
                bbox_height = y_max - y_min

                # Izračunaj srednju tačku donje ivice bounding box-a
                center_x = (x_min + x_max) // 2
                center_y = y_max

                # Izračunaj relativnu poziciju
                rel_y = relative_position(height - center_y, height)
                # Izračunaj udaljenost
                distance = calculate_distance(rel_y)

                # Debug print za udaljenost
                print(f"ID: {class_id}, Relativna y: {rel_y:.1f}px, Udaljenost: {distance:.2f} m")

                # Dodaj detekciju za dalju obradu
                detected_objects.append((x_min, y_min, x_max, y_max, center_x, center_y, class_id))
                detected_ids.add(class_id)

        # Update tracking logic with improved handling
        new_trackers = {}
        for track_id, tracker in list(trackers.items()):  # Iterate over current trackers
            success, bbox = tracker.update(frame)
            if success:
                x_min, y_min, w, h = [int(v) for v in bbox]
                x_max = x_min + w
                y_max = y_min + h
                center_x = (x_min + x_max) // 2
                center_y = y_max
                new_trackers[track_id] = tracker

                # Calculate and display velocity if previous position is available
                if track_id in prev_positions:
                    velocity = calculate_velocity(prev_positions[track_id], (center_x, center_y), frame_time)
                    label = f"ID: {track_id} - Vel: {velocity[0]:.2f}px/s, {velocity[1]:.2f}px/s"
                    cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Update previous position
                prev_positions[track_id] = (center_x, center_y)
                # Draw bounding box and distance label
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), get_class_color(track_id), 2)
                cv2.putText(frame, f"Udaljenost: {distance:.2f} m", (x_min, y_min - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
            else:
                # Remove tracker if update fails
                print(f"Tracker sa ID {track_id} nije uspio.")
                continue  # Do not include failed tracker in new_trackers

        # Initialize new trackers for detected objects that are not tracked yet
        for obj in detected_objects:
            x_min, y_min, x_max, y_max, center_x, center_y, class_id = obj
            if class_id not in trackers and class_id not in new_trackers:
                bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
                tracker = create_tracker('KCF', frame, bbox)
                new_trackers[class_id] = tracker
                prev_positions[class_id] = (center_x, center_y)

        # Update trackers dictionary
        trackers = new_trackers

        # Snimi obrađen frame
        out.write(frame)

    # Oslobodi resurse
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Testiranje funkcionalnosti
process_video(args.video, args.output)
