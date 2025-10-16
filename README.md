# Curso de Computer Vision
Platzi

## Clase 1
Resumen

La visión por computadora está revolucionando la forma en que los negocios comprenden y aprovechan la información visual. Esta tecnología permite transformar imágenes y videos en datos valiosos que impulsan decisiones estratégicas, optimizan operaciones y crean experiencias más inteligentes. Desde centros comerciales que analizan el flujo de clientes hasta fábricas que detectan defectos en tiempo real, la visión artificial se ha convertido en una herramienta indispensable para empresas innovadoras.

¿Qué es Computer Vision y cómo transforma los negocios?
Computer Vision es la tecnología que permite a las máquinas interpretar y comprender el contenido visual. Va mucho más allá de simplemente "ver" imágenes - consiste en extraer información significativa de ellas y convertirla en datos accionables. Esta capacidad está transformando industrias enteras:

Retail: Centros comerciales analizan qué tiendas atraen más clientes.
Deportes: Tiendas deportivas pueden contar cuántas personas visitan diferentes secciones (fútbol vs. tenis).
Manufactura: Fábricas detectan soldaduras defectuosas en tiempo real.
La clave está en que estos sistemas no solo capturan imágenes, sino que las interpretan y generan información valiosa que puede utilizarse para tomar decisiones de negocio fundamentadas en datos reales.

Secure Vision AI: Computer Vision en acción
En el contexto de una startup de inteligencia artificial como Secure Vision AI, la visión por computadora se aplica para resolver problemas empresariales concretos mediante el análisis de video. Esta tecnología permite implementar soluciones como:

Detección de movimiento para identificar personas u objetos
Sistemas de tracking para seguir trayectorias
Análisis de comportamiento para entender patrones
Estas capacidades permiten desarrollar aplicaciones prácticas como conteo de personas, análisis de flujo de clientes, y optimización de espacios comerciales.

¿Cómo funciona un sistema de conteo de personas?
Un sistema efectivo de conteo de personas mediante visión artificial combina varias tecnologías clave:

Detección con YOLO: Este algoritmo (You Only Look Once) identifica a cada persona en la imagen con alta precisión.

Seguimiento de centroides: Esta técnica asocia cada detección a una trayectoria específica, permitiendo seguir a las personas a través del tiempo y el espacio.

Líneas virtuales de cruce: Se implementan líneas invisibles en la imagen que, al ser atravesadas, permiten contar entradas y salidas desde diferentes direcciones.

```
# Ejemplo conceptual de implementación
import cv2
import numpy as np

# Configurar detector YOLO
yolo = cv2.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")

# Definir líneas virtuales para conteo
linea_entrada = [(200, 0), (200, 400)]
linea_salida = [(400, 0), (400, 400)]

# Función para tracking de centroides
def seguir_centroides(detecciones, centroides_previos):
    # Algoritmo de asociación
    # ...
    return centroides_actualizados
```
    
Este enfoque integrado permite no solo contar personas con precisión, sino también analizar patrones de movimiento, tiempos de permanencia y zonas de mayor interés.

Aplicaciones prácticas de Computer Vision en entornos comerciales
La visión por computadora está generando un impacto significativo en diversos sectores empresariales. Las empresas están utilizando esta tecnología para:

Automatizar procesos que antes requerían supervisión humana constante
Optimizar operaciones basándose en datos visuales precisos
Tomar decisiones más inteligentes respaldadas por análisis en tiempo real
Transformando imágenes en datos accionables
El verdadero poder de la visión artificial reside en su capacidad para convertir contenido visual en información que impulsa acciones concretas:

Un centro comercial puede rediseñar sus espacios basándose en mapas de calor de tráfico de clientes
Una tienda puede ajustar la disposición de productos según el análisis visual del comportamiento del comprador
Una fábrica puede reducir defectos identificando patrones visuales que preceden a fallos de calidad
Estos casos demuestran cómo la tecnología de visión por computadora no solo proporciona datos interesantes, sino información que conduce a mejoras tangibles en resultados de negocio.

La visión por computadora está transformando la forma en que las empresas operan, ofreciendo nuevas perspectivas basadas en datos visuales que antes eran imposibles de obtener a escala. Dominar estas tecnologías abre un mundo de posibilidades para profesionales y organizaciones que buscan mantenerse a la vanguardia de la innovación tecnológica. ¿Qué aplicación de Computer Vision podría revolucionar tu industria? Comparte tus ideas y experiencias en los comentarios.

## Clase 2
Resumen

La inteligencia artificial está revolucionando la forma en que interactuamos con imágenes y videos, permitiendo análisis más profundos y precisos que nunca antes. Las soluciones de visión por computadora están transformando industrias enteras, desde la seguridad hasta la manufactura, ofreciendo capacidades que antes parecían imposibles. Descubramos cómo Securi Vision AI está liderando esta transformación con sus innovadoras tecnologías.

¿Cuáles son los tres pilares fundamentales de la visión por computadora?
Securi Vision AI ha desarrollado soluciones que transforman radicalmente el análisis de imágenes y videos, estructurando su enfoque en tres pilares esenciales:

¿En qué consiste el procesamiento y detección de objetos?
El primer pilar combina dos capacidades fundamentales:

Procesamiento: engloba todas aquellas técnicas que permiten modificar y transformar imágenes para mejorar su calidad o resaltar características específicas.

Detección de objetos: esta tecnología permite identificar y localizar objetos específicos dentro de una imagen, enmarcándolos en rectángulos (conocidos como bounding boxes) y asignándoles etiquetas precisas.

Por ejemplo, un sistema de detección puede identificar simultáneamente a una persona, un perro terrier, un libro e incluso una taza dentro de la misma imagen, asignando a cada elemento su categoría correspondiente.

¿Qué ventajas ofrece la segmentación de imágenes?
El segundo pilar representa un avance significativo respecto a la simple detección:

Segmentación: permite delimitar con precisión los bordes exactos de cada objeto de interés, superando las limitaciones de los rectángulos de detección.
Esta técnica ofrece un nivel de precisión mucho mayor, ya que se adapta a la forma real del objeto en lugar de utilizar formas geométricas simples. La segmentación puede aplicarse a personas, animales y objetos diversos, proporcionando una comprensión más detallada de la escena.

¿Cómo funciona el análisis de poses y comportamiento?
El tercer pilar se adentra en el análisis detallado del movimiento humano:

Estimación de poses (pose estimation): esta tecnología identifica y rastrea los ángulos y posiciones de diferentes partes del cuerpo humano en imágenes o videos.
Esta capacidad resulta fundamental para aplicaciones como el análisis de movimiento deportivo, la ergonomía en entornos laborales o la detección de comportamientos anómalos en sistemas de seguridad.

¿Qué consideraciones computacionales y de negocio debemos tener en cuenta?
Al implementar soluciones de visión por computadora, es crucial considerar varios factores técnicos y estratégicos:

Duración del video: videos más largos requieren mayor capacidad de procesamiento.

Resolución de las imágenes: imágenes de alta resolución proporcionan más detalle pero exigen más recursos computacionales.

Necesidades de hardware: dependiendo de los requisitos de velocidad y eficiencia, puede ser necesario decidir entre:

GPU: para paralelizar procesos y lograr análisis en tiempo real.
CPU convencional: suficiente para procesamiento diferido o menos intensivo.
¿Qué enfoques de implementación existen?
Securi Vision AI adapta sus soluciones a diferentes escenarios de negocio:

Procesamiento diferido: en algunos casos, se graba el video durante el día y se procesa durante la noche en equipos convencionales, optimizando recursos cuando no se requiere análisis inmediato.

Procesamiento en tiempo real: para aplicaciones críticas, se conecta el sistema directamente a las cámaras para obtener resultados instantáneos, lo que requiere mayor potencia computacional.

La elección entre estos enfoques depende fundamentalmente de las necesidades específicas del negocio, el presupuesto disponible y los requisitos de tiempo de respuesta.

¿Cómo aplica Securi Vision AI estas tecnologías en casos reales?
Securi Vision AI implementa estas técnicas avanzadas para resolver problemas concretos de sus clientes. Cada uno de los tres pilares mencionados puede aplicarse de forma independiente o combinada según las necesidades específicas:

Detección de objetos: ideal para inventarios automáticos, control de acceso o conteo de personas.

Segmentación: perfecta para aplicaciones médicas, control de calidad industrial o realidad aumentada.

Análisis de poses: fundamental en seguridad laboral, análisis deportivo o sistemas de vigilancia avanzados.

Estas tecnologías no solo automatizan procesos que antes requerían intervención humana, sino que también permiten descubrir patrones y anomalías imposibles de detectar a simple vista.

La visión por computadora está transformando radicalmente nuestra capacidad para extraer información valiosa de imágenes y videos. Las soluciones como las que ofrece Securi Vision AI están abriendo nuevas posibilidades en innumerables campos. ¿Qué aplicaciones de estas tecnologías te resultan más interesantes? Comparte tus ideas y experiencias en los comentarios.

## Clase 3
Resumen

El procesamiento de imágenes con OpenCV se ha convertido en una herramienta fundamental para el análisis visual en entornos comerciales. Esta tecnología permite a las empresas obtener información valiosa sobre el comportamiento de sus clientes, optimizar la distribución de productos y mejorar la experiencia de compra. A continuación, exploraremos cómo implementar soluciones de visión por computadora para un caso real de análisis de flujo de clientes en una tienda.

¿Qué es OpenCV y por qué es importante para el análisis visual?
OpenCV (Open Source Computer Vision Library) es una biblioteca de código abierto diseñada específicamente para la manipulación y análisis de imágenes y video. Esta potente herramienta ofrece numerosas ventajas que la convierten en la elección preferida para proyectos de visión por computadora:

Procesamiento en tiempo real optimizado para ejecutar algoritmos complejos a alta velocidad
Compatibilidad multiplataforma con Python, Java y C++
Amplio conjunto de herramientas que incluyen detección de bordes, reconocimiento facial y segmentación de imágenes
Comunidad activa que constantemente genera tutoriales y documentación actualizada
Para comenzar a trabajar con OpenCV en Python, necesitamos instalar la biblioteca y realizar algunas importaciones básicas:

```
# Instalación (si no está instalada)
# pip install opencv-python matplotlib

# Importaciones necesarias
import cv2
import matplotlib.pyplot as plt
¿Cómo cargar y manipular imágenes con OpenCV?
Carga básica de imágenes
El primer paso para trabajar con imágenes es cargarlas correctamente. OpenCV proporciona métodos simples pero potentes para esta tarea:

# Definir la ruta de la imagen
path = "data/centro_comercial.jpg"

# Cargar la imagen
image = cv2.imread(path)

# Verificar si la imagen se cargó correctamente
if image is not None:
    # Convertir de BGR a RGB para visualización con Matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Mostrar la imagen
    plt.figure()
    plt.title("Imagen capturada por CCTV")
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.show()
else:
    print("No se pudo cargar la imagen")
```
    
Es importante destacar que OpenCV maneja las imágenes en formato BGR (Blue, Green, Red), mientras que la mayoría de las otras bibliotecas, como Matplotlib, utilizan el formato RGB (Red, Green, Blue). Por esta razón, es necesario realizar una conversión de color cuando se trabaja con ambas bibliotecas simultáneamente.

Captura y visualización de video en tiempo real
Para capturar video desde una webcam, OpenCV ofrece una interfaz sencilla:

```
# Iniciar la captura de video (0 para la primera cámara, 1 para la segunda, etc.)
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("No se pudo abrir la cámara")
else:
    while True:
        # Capturar frame por frame
        ret, frame = cap.read()
        
        if not ret:
            print("No se pudo leer el frame")
            break
            
        # Mostrar el frame resultante
        cv2.imshow('Frame CCTV', frame)
        
        # Salir si se presiona 'q'
        if cv2.waitKey(1) == ord('q'):
            break
            
    # Liberar el objeto de captura y cerrar ventanas
    cap.release()
    cv2.destroyAllWindows()
```
    
Carga de videos pregrabados
De manera similar, podemos cargar y reproducir videos almacenados en el disco:

```
# Cargar un video desde archivo
video_path = "data/video_tienda.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("No se pudo abrir el video")
else:
    while True:
        # Capturar frame por frame
        ret, frame = cap.read()
        
        if not ret:
            print("No se pudo leer el frame")
            break
            
        # Mostrar el frame resultante
        cv2.imshow('Video CCTV', frame)
        
        # Salir si se presiona 'q'
        if cv2.waitKey(25) == ord('q'):
            break
            
    # Liberar el objeto de captura y cerrar ventanas
    cap.release()
    cv2.destroyAllWindows()
```
    
¿Qué técnicas de mejora y anotación podemos aplicar a las imágenes?
Ajuste de brillo y contraste
Para mejorar la calidad visual de las imágenes, especialmente en condiciones de iluminación variable, podemos ajustar el brillo y el contraste:

```
# Ajustar brillo y contraste
contrast_factor = 1.2  # Mayor que 1 aumenta el contraste
brightness_value = 30  # Valor positivo aumenta el brillo

# Aplicar transformación
enhanced_image = cv2.convertScaleAbs(image, alpha=contrast_factor, beta=brightness_value)

# Convertir para visualización
enhanced_image_rgb = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB)

# Mostrar imagen mejorada
plt.figure()
plt.title("Imagen con brillo y contraste ajustados")
plt.imshow(enhanced_image_rgb)
plt.axis('off')
plt.show()

# Guardar la imagen procesada
cv2.imwrite("ccv_imagen_correccion.jpg", enhanced_image)
Corrección de color y normalización
Para ajustes más avanzados de color, podemos trabajar en el espacio de color HSV (Hue, Saturation, Value):

import numpy as np

# Convertir a HSV
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Aumentar la saturación
hsv_image[:, :, 1] = hsv_image[:, :, 1] * 1.5  # Multiplicar el canal de saturación

# Asegurar que los valores estén dentro del rango válido
hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1], 0, 255)

# Convertir de vuelta a BGR y luego a RGB para visualización
saturated_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
saturated_image_rgb = cv2.cvtColor(saturated_image, cv2.COLOR_BGR2RGB)

# Mostrar imagen con saturación ajustada
plt.figure()
plt.title("Imagen con saturación ajustada")
plt.imshow(saturated_image_rgb)
plt.axis('off')
plt.show()

# Guardar la imagen procesada
cv2.imwrite("ajuste_saturacion.jpg", saturated_image)
Anotaciones en imágenes
Una de las funcionalidades más útiles de OpenCV es la capacidad de añadir anotaciones a las imágenes, como líneas, rectángulos y texto:

# Crear una copia de la imagen original
annotated_image = image.copy()

# Dibujar una línea
cv2.line(annotated_image, (50, 50), (200, 50), (255, 0, 0), 3)  # Azul en BGR

# Dibujar un rectángulo
cv2.rectangle(annotated_image, (100, 100), (300, 200), (0, 255, 0), 2)  # Verde en BGR

# Añadir texto
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(annotated_image, "Zona de alto tráfico", (100, 90), font, 0.7, (0, 0, 255), 2)  # Rojo en BGR

# Convertir para visualización
annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

# Mostrar imagen anotada
plt.figure()
plt.title("Imagen con anotaciones")
plt.imshow(annotated_image_rgb)
plt.axis('off')
plt.show()
Comparación de todas las técnicas aplicadas
Para visualizar el impacto de las diferentes técnicas, podemos mostrar todas las imágenes procesadas juntas:

plt.figure(figsize=(15, 10))

# Imagen original
plt.subplot(2, 2, 1)
plt.title("Imagen original")
plt.imshow(image_rgb)
plt.axis('off')

# Imagen con brillo y contraste ajustados
plt.subplot(2, 2, 2)
plt.title("Brillo y contraste ajustados")
plt.imshow(enhanced_image_rgb)
plt.axis('off')

# Imagen con saturación ajustada
plt.subplot(2, 2, 3)
plt.title("Saturación ajustada")
plt.imshow(saturated_image_rgb)
plt.axis('off')

# Imagen con anotaciones
plt.subplot(2, 2, 4)
plt.title("Imagen anotada")
plt.imshow(annotated_image_rgb)
plt.axis('off')

plt.tight_layout()
plt.show()
```

El procesamiento de imágenes con OpenCV ofrece un mundo de posibilidades para el análisis visual en entornos comerciales. Desde el seguimiento del flujo de clientes hasta la identificación de zonas de alto tráfico, estas técnicas pueden proporcionar información valiosa para la toma de decisiones estratégicas. ¿En qué situaciones específicas aplicarías estas técnicas de procesamiento de imágenes? Comparte tus ideas y experiencias en los comentarios.

## Clase 4
Resumen

La seguridad visual mediante análisis de video se ha convertido en una herramienta fundamental para entender el comportamiento de los clientes en espacios comerciales. A través de técnicas de procesamiento de imágenes y detección de movimiento, podemos generar mapas de calor que revelan patrones de interés y flujo de personas, proporcionando información valiosa para la toma de decisiones estratégicas en retail. Veamos cómo implementar esta tecnología y qué beneficios ofrece para optimizar la disposición de productos y mejorar la experiencia del cliente.

¿Cómo crear mapas de calor a partir de videos de seguridad?
El desafío de Vision Security para ILAC consiste en analizar un video de los pasillos de una tienda para determinar dónde se detienen los clientes, cuánto tiempo permanecen en cada posición y qué están observando. El producto final es un mapa de calor que visualiza estas concentraciones de actividad.

Para este proyecto, podemos utilizar Google Colab o trabajar localmente. Google Colab ofrece algunas ventajas, como el acceso a GPU para procesamiento más rápido, aunque tiene limitaciones para la visualización de video en tiempo real. Sin embargo, para el análisis de videos pregrabados, funciona perfectamente.

El proceso básico incluye:

Cargar el video mediante OpenCV.
Utilizar técnicas de detección de movimiento para identificar áreas de actividad.
Acumular esta información en un mapa de calor.
Normalizar y visualizar los resultados.
¿Qué herramientas necesitamos para el análisis de video?
Para implementar esta solución, necesitamos las siguientes bibliotecas:

```
import cv2
import numpy as np
import matplotlib.pyplot as plt
```

OpenCV nos proporciona un método especialmente útil para detectar movimiento entre frames a lo largo del video. Este método puede extraer el fondo y resaltar solo lo que se está moviendo, ya sean personas, animales o incluso objetos como cortinas.

El método recibe tres parámetros principales:

History: número de frames utilizados para distribuir el fondo
Sensibilidad: para detectar cambios (distinguiendo entre movimientos menores y significativos)
Detección de sombras: para considerar o ignorar las sombras en el análisis

¿Cómo se genera y se interpreta un mapa de calor?
El proceso para generar el mapa de calor consiste en:
1. Iniciar un acumulador en cero.
2. Sustraer el fondo de cada frame para determinar las áreas de movimiento.
3. Acumular la máscara de movimiento a lo largo del tiempo.
4. Superponer esta información acumulada sobre el video original.

   
El resultado es un mapa donde las áreas de color rojo más intenso representan lugares donde los clientes permanecieron más tiempo. Por ejemplo, si un cliente estuvo parado mucho tiempo en una posición específica, esa área aparecerá con un rojo más intenso en el mapa.

```
# Ejemplo conceptual del código
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
heatmap = np.zeros((height, width), dtype=np.float32)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Aplicar sustracción de fondo
    fgmask = fgbg.apply(frame)
    
    # Acumular la máscara en el mapa de calor
    heatmap += fgmask
```
    
### ¿Cómo normalizar los mapas de calor para una mejor interpretación?
En entornos con mucho flujo de personas, como centros comerciales, el mapa de calor podría saturarse y verse completamente rojo. Para solucionar esto, se aplica una normalización:

Se toma el valor mínimo y se lleva a cero.
El valor máximo obtenido en el mapa se lleva a 255.
Esto permite una mejor visualización de las diferencias relativas en la concentración de actividad. Además, podemos utilizar diferentes escalas de color, como "viridis" (azul-violeta), para mejorar la interpretación visual.

```
# Normalización del mapa de calor
normalized_heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
Esta normalización nos ayuda a identificar áreas de mayor interés para los clientes, lo que puede informar decisiones sobre la ubicación de productos o la reorganización del espacio comercial.
```

¿Qué limitaciones tiene esta técnica y cómo superarlas?
Una limitación importante de este enfoque es que detecta cualquier movimiento, no solo el de las personas. Esto puede generar resultados engañosos cuando hay objetos en movimiento en la escena.

Por ejemplo, en el segundo video analizado (un parque con personas), una cuerda en movimiento generó una concentración alta en el mapa de calor, desviando la atención del flujo real de personas.

Este problema ocurre porque el método de sustracción de fondo detecta cualquier cambio en la escena, sin distinguir entre tipos de objetos. Para aplicaciones centradas en el comportamiento humano, necesitamos filtrar específicamente el movimiento de personas.

La solución a este problema se encuentra en técnicas más avanzadas de visión por computadora, como la detección de personas mediante modelos de aprendizaje profundo. Estos modelos pueden identificar específicamente figuras humanas y seguir su movimiento, ignorando otros objetos en movimiento.

El análisis de video para seguridad y marketing ofrece información valiosa sobre el comportamiento de los clientes en espacios comerciales. Mediante la creación de mapas de calor, podemos identificar áreas de mayor interés y optimizar la disposición de productos. Aunque la técnica básica tiene limitaciones, como la detección indiscriminada de movimiento, existen soluciones avanzadas que permiten enfocarse específicamente en el comportamiento humano. ¿Has implementado alguna vez análisis de video en tu negocio? ¿Qué insights has obtenido? Comparte tu experiencia en los comentarios.

## Clase 5
Resumen

La segmentación de imágenes con YOLO representa una de las técnicas más avanzadas en visión por computadora, permitiendo identificar y delimitar objetos específicos en tiempo real. Esta tecnología no solo detecta objetos, sino que también crea máscaras precisas que los separan del fondo, ofreciendo aplicaciones revolucionarias en campos como la robótica, la vigilancia y el análisis de video.

### ¿Qué es YOLO y por qué es importante para la segmentación de imágenes?
YOLO (You Only Look Once) es un algoritmo de detección de objetos que analiza una imagen en una sola pasada, lo que le permite ser extremadamente rápido. Actualmente, la empresa Ultralytics está desarrollando nuevas versiones de YOLO, ofreciendo una librería de código abierto con una gran comunidad de soporte.

Las características principales de YOLO incluyen:
- Procesamiento en tiempo real
- Segmentación de imágenes
- Detección de objetos
- Identificación de puntos característicos del cuerpo

La popularidad de YOLO se debe a su eficiencia y versatilidad, permitiendo implementaciones en diversos dispositivos, desde potentes GPUs hasta sistemas con recursos limitados como CPUs estándar.

### ¿Cómo implementar la segmentación con YOLO en tiempo real?
Para implementar la segmentación con YOLO necesitamos algunas herramientas fundamentales:
- OpenCV para el procesamiento de imágenes
- NumPy para operaciones matemáticas
- Ultralytics para acceder a los modelos YOLO

Si aún no tienes instalada la librería Ultralytics, puedes hacerlo con un simple comando de instalación mediante pip.

Selección del modelo adecuado
Para la segmentación, utilizaremos YOLOv11 en su versión "nano", que es el modelo más pequeño disponible. Esta elección es ideal cuando trabajamos con CPUs en lugar de GPUs, ya que requiere menos recursos computacionales.

```
# Importar las librerías necesarias
import cv2
import numpy as np
from ultralytics import YOLO

# Cargar el modelo YOLOv11 nano
model = YOLO('yolov11n.pt')

# Configurar la fuente de video (0 para la cámara principal, 1 para la segunda cámara)
cap = cv2.VideoCapture(1)

# Configurar la resolución
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
```

Procesamiento de frames y medición de latencia
Un aspecto importante al trabajar con segmentación en tiempo real es medir la latencia, es decir, cuánto tiempo tarda el sistema en procesar cada frame:

```
while True:
    # Capturar frame
    ret, frame = cap.read()
    
    # Medir tiempo de inicio
    start_time = time.time()
    
    # Procesar el frame con YOLO
    results = model(frame)
    
    # Calcular latencia
    latency = time.time() - start_time
    
    # Mostrar FPS
    cv2.putText(frame, f"Latencia: {latency:.3f}s", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
```

Visualización de bounding boxes y segmentación
La segmentación con YOLO proporciona dos elementos principales:

1. Bounding boxes: Rectángulos que encierran los objetos detectados
2. Máscaras de segmentación: Áreas coloreadas que delimitan exactamente la forma del objeto

```
# Acceder a las detecciones
for r in results:
    boxes = r.boxes
    masks = r.masks
    
    # Procesar cada detección
    for i, box in enumerate(boxes):
        # Obtener coordenadas del bounding box
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        # Obtener confianza y clase
        confidence = float(box.conf[0])
        class_id = int(box.cls[0])
        
        # Dibujar bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Añadir etiqueta
        label = f"{class_names[class_id]}: {confidence:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
```

Creación y aplicación de máscaras de segmentación
La parte más distintiva de la segmentación es la creación de máscaras que se superponen a la imagen original:

```
# Procesar máscaras de segmentación
if masks is not None:
    for i, mask in enumerate(masks):
        # Redimensionar la máscara al tamaño del frame
        mask_image = mask.data.cpu().numpy()
        mask_image = cv2.resize(mask_image, (frame.shape[1], frame.shape[0]))
        
        # Crear máscara booleana
        mask_bool = mask_image > 0.5
        
        # Generar color aleatorio para la máscara
        color = np.random.randint(0, 255, 3, dtype=np.uint8)
        
        # Aplicar color a la máscara
        colored_mask = np.zeros_like(frame)
        colored_mask[mask_bool] = color
        
        # Combinar máscara con el frame original
        frame = cv2.addWeighted(frame, 1, colored_mask, 0.5, 0)
```

### ¿Cómo filtrar objetos específicos en la segmentación?
Una de las ventajas de YOLO es la capacidad de filtrar objetos específicos según nuestras necesidades. Podemos hacerlo de dos maneras:

Filtrado por confianza
Podemos establecer un umbral de confianza para mostrar solo las detecciones con alta probabilidad:

```
# Configurar umbral de confianza
confidence_threshold = 0.7

# Filtrar detecciones por confianza
results = model(frame, conf=confidence_threshold)
```

Filtrado por clase de objeto
YOLO viene preentrenado para detectar 80 categorías diferentes de objetos. Podemos filtrar para mostrar solo las clases que nos interesan:

```
# Clases de interés (0 = persona, 13 = silla)
classes_of_interest = [0, 13]

# Filtrar por clase y confianza
results = model(frame, conf=0.7, classes=classes_of_interest)
```

El modelo preentrenado de YOLO asigna un número a cada categoría de objeto. Por ejemplo:
- 0: Persona
- 13: Silla
- 41: Taza
- 77: Oso de peluche
  
Esto permite una gran flexibilidad para aplicaciones específicas, como sistemas de seguridad que solo detecten personas o aplicaciones de inventario que se centren en productos particulares.

La segmentación con YOLO representa una herramienta poderosa para el análisis de imágenes, combinando velocidad y precisión en un solo sistema. Su capacidad para procesar video en tiempo real y generar máscaras precisas abre un mundo de posibilidades para desarrolladores e investigadores. ¿Has probado implementar YOLO en alguno de tus proyectos? Comparte tu experiencia en los comentarios.
