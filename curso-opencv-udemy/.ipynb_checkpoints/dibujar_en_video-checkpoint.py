import cv2

captura = cv2.VideoCapture(0)

ancho = int(captura.get(cv2.CAP_PROP_FRAME_WIDTH))
alto = int(captura.get(cv2.CAP_PROP_FRAME_HEIGHT))

# esquina izquierda
x = 0
y = 0

# dimensiones del rectangulo
w = ancho // 4
h = alto // 4

while True:

    resultado, video = captura.read()

    # dibujamos el rectangulo
    cv2.rectangle(video, (x,y), (x+w, y+h), color= (255,0,0), thickness=4)

    cv2.imshow('Nuestro video con rectangulo', video)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

captura.release()
cv2.destroyAllWindows()