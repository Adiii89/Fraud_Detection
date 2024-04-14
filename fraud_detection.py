from ultralytics import YOLO
import cv2
import math
class POS_System:
    def __init__(self):
        self.scanned_items = {} # Using manually typed dictionary to verify
        self.detected_items = set()

    """
    here you can integrate a product scanning device with system, as soon as you scan the product it should 
    append that product name to the scanned item dictionary using the scan_item function
    """

    # def scan_item(self, item):
    #     self.scanned_items.append(item)
    def detect_item(self, item):
        self.detected_items.add(item)

    def compare_items(self):
        if self.scanned_items == self.detected_items:
            print("Items match with the receipt.")
        else:
            print("Items do not match with the receipt.")

# Initialize supermarket POS system
pos_system = POS_System()

detected_labels = set()


model = YOLO('yolov8n.pt')  #here you can give your model trained on the custom dataset
                           #According to your custom dataset you can update your classname
classNames = {
    0: 'background',
    1: 'person',
    2: 'bicycle',
    3: 'car',
    4: 'motorcycle',
    5: 'airplane',
    6: 'bus',
    7: 'train',
    8: 'truck',
    9: 'boat',
    10: 'traffic light',
    11: 'fire hydrant',
    12: 'stop sign',
    13: 'parking meter',
    14: 'bench',
    15: 'bird',
    16: 'cat',
    17: 'dog',
    18: 'horse',
    19: 'sheep',
    20: 'cow',
    21: 'elephant',
    22: 'bear',
    23: 'zebra',
    24: 'giraffe',
    25: 'backpack',
    26: 'umbrella',
    27: 'handbag',
    28: 'tie',
    29: 'suitcase',
    30: 'frisbee',
    31: 'skis',
    32: 'snowboard',
    33: 'sports ball',
    34: 'kite',
    35: 'baseball bat',
    36: 'baseball glove',
    37: 'skateboard',
    38: 'surfboard',
    39: 'tennis racket',
    40: 'bottle',
    41: 'wine glass',
    42: 'cup',
    43: 'fork',
    44: 'knife',
    45: 'spoon',
    46: 'bowl',
    47: 'banana',
    48: 'apple',
    49: 'sandwich',
    50: 'orange',
    51: 'broccoli',
    52: 'carrot',
    53: 'hot dog',
    54: 'pizza',
    55: 'donut',
    56: 'cake',
    57: 'chair',
    58: 'couch',
    59: 'potted plant',
    60: 'bed',
    61: 'dining table',
    62: 'toilet',
    63: 'tv',
    64: 'laptop',
    65: 'mouse',
    66: 'remote',
    67: 'keyboard',
    68: 'cell phone',
    69: 'microwave',
    70: 'oven',
    71: 'toaster',
    72: 'sink',
    73: 'refrigerator',
    74: 'book',
    75: 'clock',
    76: 'vase',
    77: 'scissors',
    78: 'teddy bear',
    79: 'hair drier',
    80: 'toothbrush'
}

cap = cv2.VideoCapture("C:/Users/admin/OneDrive/Desktop/jhj.mp4")
#cap = cv2.VideoCapture(0)
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