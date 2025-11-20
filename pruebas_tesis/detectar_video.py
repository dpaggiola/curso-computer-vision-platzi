from ultralytics import YOLO
import cv2

# === MODELOS ===
coco_model = YOLO("modelos_finales/coco.pt")
custom_model = YOLO("modelos_finales/custom.pt")

def procesar_frame(frame):
    # 1. Inferencia COCO
    coco_res = coco_model.predict(source=frame, imgsz=640, conf=0.3)[0]
    # 2. Inferencia CUSTOM
    custom_res = custom_model.predict(source=frame, imgsz=640, conf=0.25)[0]

    # Dibujar en frame
    def dibujar(boxes, names, color):
        for b in boxes:
            cls = int(b.cls[0])
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            conf = float(b.conf)
            name = names[cls]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{name} - conf: {conf:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    dibujar(coco_res.boxes, coco_model.names, (255, 0, 0))
    dibujar(custom_res.boxes, custom_model.names, (0, 0, 255))

    return frame

# === PROCESAR VIDEO ===
video_in = "video.mp4"
video_out = "salida_detectada.mp4"

cap = cv2.VideoCapture(video_in)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(video_out, fourcc, 30.0,
                      (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = procesar_frame(frame)
    out.write(frame)

cap.release()
out.release()
print("Video procesado guardado como:", video_out)
