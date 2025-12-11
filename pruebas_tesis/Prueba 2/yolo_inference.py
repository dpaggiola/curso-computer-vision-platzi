from ultralytics import YOLO

model = YOLO('yolo11s')

results = model.predict('input_video/video.mp4', save=True)
print(results[0])
print('=================================================')

for box in results[0].boxes:
  print(box)