from datetime import datetime
from ultralytics import YOLO
import cv2
import math

class POS_System:
    def __init__(self):
        self.scanned_items = {}
        self.detected_items = set()
        self.payment_made = False
        self.invoice_given = False

    def detect_item(self, item):
        self.detected_items.add(item)

    def compare_items(self):
        if self.scanned_items == self.detected_items:
            print("Items match with the receipt.")
        else:
            print("Items do not match with the receipt.")

    def make_payment(self, method):
        if method == "cash" or method == "card" or method == "phone":
            print(f"Payment made via {method}.")
            self.payment_made = True
        else:
            print("Invalid payment method.")

    def give_invoice(self):
        print("Invoice given.")
        self.invoice_given = True

# Initialize supermarket POS system
pos_system = POS_System()

detected_labels = set()

model = YOLO('yolov8n.pt')

classNames = {0: 'person',    # Example class, adjust it according to your custom dataset
 1: 'bicycle',
 2: 'car',
 3: 'motorcycle',
 4: 'airplane',
 5: 'bus',
 6: 'train',
 7: 'truck',
 8: 'boat',
 9: 'traffic light',
 10: 'fire hydrant',
 11: 'stop sign',
 12: 'parking meter',
 13: 'bench',
 14: 'bird',
 15: 'cat',
 16: 'dog',
 17: 'horse',
 18: 'sheep',
 19: 'cow',
 20: 'elephant',
 21: 'bear',
 22: 'zebra',
 23: 'giraffe',
 24: 'backpack',
 25: 'umbrella',
 26: 'handbag',
 27: 'tie',
 28: 'suitcase',
 29: 'frisbee',
 30: 'skis',
 31: 'snowboard',
 32: 'sports ball',
 33: 'kite',
 34: 'baseball bat',
 35: 'baseball glove',
 36: 'skateboard',
 37: 'surfboard',
 38: 'tennis racket',
 39: 'bottle',
 40: 'wine glass',
 41: 'cup',
 42: 'fork',
 43: 'knife',
 44: 'spoon',
 45: 'bowl',
 46: 'banana',
 47: 'apple',
 48: 'sandwich',
 49: 'orange',
 50: 'broccoli',
 51: 'carrot',
 52: 'hot dog',
 53: 'pizza',
 54: 'donut',
 55: 'cake',
 56: 'chair',
 57: 'couch',
 58: 'potted plant',
 59: 'bed',
 60: 'dining table',
 61: 'toilet',
 62: 'tv',
 63: 'laptop',
 64: 'mouse',
 65: 'remote',
 66: 'keyboard',
 67: 'cell phone',
 68: 'microwave',
 69: 'oven',
 70: 'toaster',
 71: 'sink',
 72: 'refrigerator',
 73: 'book',
 74: 'clock',
 75: 'vase',
 76: 'scissors',
 77: 'teddy bear',
 78: 'hair drier',
 79: 'toothbrush'}


cap = cv2.VideoCapture("CCTV (2).mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, conf=0.3, iou=0.7, show=False)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            label = classNames[cls]

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} ({conf}%)", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            detected_labels.add(label)

            # Update payment_made and invoice_given flags
            if label == "cash" or label == "card" or label == "phone":
                pos_system.make_payment(label)
                pos_system.give_invoice()

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

for item in detected_labels:
    pos_system.detect_item(item)

# Compare scanned items with detected items
print("Scanned items:", pos_system.scanned_items)
print("Detected items:", pos_system.detected_items)
pos_system.compare_items()
