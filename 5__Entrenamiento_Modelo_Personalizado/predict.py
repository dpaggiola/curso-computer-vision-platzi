from ultralytics import YOLO
import sys
from PIL import Image

if len(sys.argv) < 2:
    print('Uso: python predict.py <ruta_imagen>')
    sys.exit(1)

img = sys.argv[1]
model = YOLO('yolo11n.pt')
model.reset_class_names(['person','sports ball','cone','goal','ladder'])
res = model(img)
# guardar plot en disk
out = res[0]
out.plot(save=True)
print('Resultado guardado.')
