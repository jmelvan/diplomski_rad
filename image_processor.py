import cv2
from ultralytics import YOLO
import argparse
import os
import numpy as np
from yolo_classes import yolo_classes
from google.colab.patches import cv2_imshow


# Postavljanje argument parsera
parser = argparse.ArgumentParser(description="YOLOv8 distanca od kamere")
parser.add_argument('--model', type=str, required=True, help="Putanja do YOLOv8 modela")
parser.add_argument('--img', type=str, required=True, help="Putanja do slike")
parser.add_argument('--output', type=str, default='output_image.jpg', help="Putanja do izlazne slike")
parser.add_argument('--min_confidence', type=float, default=0.7, help="Minimalna pouzdanost (confidence) za detekcije") 
args = parser.parse_args()

# Provjera da li slika postoji
if not os.path.exists(args.img):
    print(f"Slika ne postoji: '{args.img}'.")
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

# Funkcija za izračun udaljenosti - default picture_height is 304px, so calculate relative
def calculate_distance(relative_y):
    try:
        return 2.8921 * 1.0238 ** relative_y
    except OverflowError as e:
        print(f"Greška u izračunavanju udaljenosti: {e}")
        return float('inf')  # Vratiti beskonačnost ako dođe do greške

# Funkcija za predikciju i crtanje udaljenosti
def predict_and_draw_distance(image_path, output_path):
    # Učitaj sliku
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    
    # Izvrši predikciju
    results = model.predict(image_path, imgsz=640, save=False, conf=args.min_confidence)
    
    # Prikaz rezultata
    for result in results:
        # Pretvori rezultate u numpy array
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
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
                
                # Prikaži ime klase i udaljenost iznad bounding box-a
                class_name = model.names[class_id]  # Pretpostavka da model.names sadrži imena klasa
                label = f"{class_name} - l: {yolo_classes[class_id]['length']}" #ispiši distancu
                cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Prikaži srednju tačku donje ivice
                cv2.circle(image, (center_x, center_y), 5, (0, 0, 255), -1)  # Crveni krug za srednju tačku
                cv2.putText(image, f"Mjerna tocka ( y = {rel_y:.1f}px )", (center_x + 10, center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.putText(image, f"Udaljenost: {distance:.2f} m", (center_x + 10, center_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2) # ispiši udaljenost

            
            except OverflowError as e:
                print(f"Greška u obradi box-a: {e}")

    # Sačuvaj sliku sa dodanim informacijama
    cv2.imwrite(output_path, image)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Testiranje funkcionalnosti
predict_and_draw_distance(args.img, args.output)