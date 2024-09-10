import cv2
from ultralytics import YOLO
import argparse
import os
import numpy as np
from yolo_classes import yolo_classes

# Postavljanje argument parsera
parser = argparse.ArgumentParser(description="YOLOv8 distanca od kamere za video")
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
    return y*(304/picture_height)

# Funkcija za izračun udaljenosti
def calculate_distance(relative_y):
    try:
        return 2.8921 * 1.0238 ** relative_y
    except OverflowError as e:
        print(f"Greška u izračunavanju udaljenosti: {e}")
        return float('inf')  # Vratiti beskonačnost ako dođe do greške

# Funkcija za predikciju i crtanje udaljenosti na frame-ove
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

    # Kreiraj VideoWriter za snimanje obrađenog video zapisa
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Izvrši predikciju
        results = model.predict(frame, imgsz=640, save=False, conf=args.min_confidence)

        # Prikaz rezultata
        for result in results:
            boxes = result.boxes.xywh.cpu().numpy()  # Uzmemo xywh koordinate
            classes = result.boxes.cls.cpu().numpy().astype(int)  # Klase objekata

            for box, class_id in zip(boxes, classes):
                try:
                    x_center, y_center, bbox_width, bbox_height = box[:4]  # Izvučemo koordinate i dimenzije

                    # Proveri da li su koordinate u validnom opsegu
                    if not (0 <= x_center <= width and 0 <= y_center <= height and 0 <= bbox_width <= width and 0 <= bbox_height <= height):
                        print(f"Nevalidne koordinate: {x_center}, {y_center}, {bbox_width}, {bbox_height}")
                        continue

                    # Izračunaj xmin, ymin, xmax, ymax
                    x_min = int(x_center - bbox_width / 2)
                    y_min = int(y_center - bbox_height / 2)
                    x_max = int(x_center + bbox_width / 2)
                    y_max = int(y_center + bbox_height / 2)

                    # Izračunaj visinu objekta u pikselima
                    bbox_height = y_max - y_min

                    # Izračunaj srednju tačku donje ivice bounding box-a
                    center_x = (x_min + x_max) // 2
                    center_y = y_max

                    rel_y = relative_position(height - center_y, height)
                    # Izračunaj udaljenost
                    distance = calculate_distance(rel_y)

                    # Prikaži bounding box u boji klase
                    color = get_class_color(class_id)
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)

                    # Prikaži ime klase i udaljenost iznad bounding box-a
                    class_name = model.names[class_id]  # Pretpostavka da model.names sadrži imena klasa
                    label = f"{class_name} - l: {yolo_classes[class_id]['length']}" #ispiši distancu
                    cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    # Prikaži srednju tačku donje ivice
                    cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)  # Crveni krug za srednju tačku
                    cv2.putText(frame, f"Mjerna tocka ( y = {rel_y:.1f}px )", (center_x + 10, center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    cv2.putText(frame, f"Udaljenost: {distance:.2f} m", (center_x + 10, center_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2) # ispiši udaljenost

                except OverflowError as e:
                    print(f"Greška u obradi box-a: {e}")

        # Snimi obraden frame
        out.write(frame)

    # Oslobodi resurse
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Testiranje funkcionalnosti
process_video(args.video, args.output)