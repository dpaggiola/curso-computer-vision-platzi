from ultralytics import YOLO

# Carga el modelo preentrenado y resetea las clases
model = YOLO('yolo11n.pt')
model.reset_class_names(['person','sports ball','cone','goal','ladder'])
print('Modelo cargado y clases reseteadas:', model.names)
print('Ejecutando validación con data.yaml...')
model.val()
