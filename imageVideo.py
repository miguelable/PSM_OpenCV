import cv2
import requests
import numpy as np

# Dirección del flujo MJPEG de la ESP32-CAM
url = "http://10.220.151.99:81/"  # Cambia "<IP_DE_LA_ESP32>" por la IP asignada

# Crear una ventana para mostrar el video
cv2.namedWindow("ESP32-CAM Video", cv2.WINDOW_AUTOSIZE)

try:
    # Conectar al flujo de video
    stream = requests.get(url, stream=True)
    if stream.status_code == 200:
        byte_data = b""  # Datos binarios acumulados

        # Leer el flujo MJPEG
        for chunk in stream.iter_content(chunk_size=1024):
            byte_data += chunk
            start = byte_data.find(b'\xff\xd8')  # Inicio de imagen JPEG
            end = byte_data.find(b'\xff\xd9')    # Fin de imagen JPEG

            if start != -1 and end != -1:
                # Extraer el fotograma JPEG
                jpg_data = byte_data[start:end+2]
                byte_data = byte_data[end+2:]

                # Convertir JPEG en imagen OpenCV
                img_array = np.frombuffer(jpg_data, dtype=np.uint8)
                frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                # Mostrar el fotograma
                cv2.imshow("ESP32-CAM Video", frame)

                # Salir con la tecla 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    else:
        print(f"Error al conectarse al flujo MJPEG: {stream.status_code}")
except Exception as e:
    print(f"Error: {e}")

finally:
    # Cerrar todas las ventanas
    cv2.destroyAllWindows()
