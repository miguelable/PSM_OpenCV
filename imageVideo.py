import cv2
import requests
import numpy as np

# Dirección del flujo MJPEG de la ESP32-CAM
url = "http://10.220.151.99:81/"  # Cambia "<IP_DE_LA_ESP32>" por la IP asignada

# Crear una ventana para mostrar el video
cv2.namedWindow("ESP32-CAM Video", cv2.WINDOW_AUTOSIZE)

# Funciones para decodificar la imagen
def convert_jpg_to_image(jpg_data):
    img_array = np.frombuffer(jpg_data, dtype=np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_COLOR) # Convertir JPEG en imagen OpenCV

def blur_image(image, kernel_size):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0) # Aplicar filtro de desenfoque

def convert_image_to_gray(image):
    cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convertir de BGR a RGB
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) # Convertir a escala de grises

def convert_image_to_hsv(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convertir de
    return cv2.cvtColor(image, cv2.COLOR_RGB2HSV) # Convertir a HSV

def close_dilate_mask(mask):
    # Apply close the mask
    kernel = np.ones((7,7), np.uint16)   
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # Dilate the mask
    kernel = np.ones((5,5), np.uint16)
    mask_closed = cv2.dilate(mask_closed, kernel, iterations=1)
    return mask_closed

def get_red_mask(hsv_image):
    # H = 125, S = 20, V = 0, H1 = 179, S1 = 255, V1 = 255
    # H = 120, S = 40, V = 100, H1 = 170, S1 = 255, V1 = 240
    lower_red1 = np.array([0, 50, 50])
    lower_red2 = np.array([10, 255, 255])
    upper_red1 = np.array([150, 50, 50])
    upper_red2 = np.array([179, 255, 255])
    mask_red1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    mask_red = cv2.add(mask_red1, mask_red2)
    # mask_red = close_dilate_mask(mask_red)
    return mask_red

def get_green_mask(hsv_image):
    # H = 50, S = 40, V = 0, H1 = 110, S1 = 255, V1 = 160
    # H = 50, S = 50, V = 0, H1 = 90, S1 = 255, V1 = 190
    lower_green = np.array([30, 20, 20])
    upper_green = np.array([90, 255, 255])
    mask_green = cv2.inRange(hsv_image, lower_green, upper_green)
    # mask_green = close_dilate_mask(mask_green)
    return mask_green

def apply_mask_to_image(image, mask):
    segmented_image = cv2.bitwise_and(image, image, mask=mask)
    return segmented_image

def sobel_filter(image, kernel_size):
    img_sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=kernel_size)
    img_sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=kernel_size) 
    img_sobel = np.sqrt(img_sobel_x**2 + img_sobel_y**2)
    img_sobel = cv2.normalize(img_sobel, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    _, filtered_image = cv2.threshold(img_sobel, 30, 255, cv2.THRESH_BINARY)
    return filtered_image

def find_contours(image):
    image_filter = sobel_filter(image, 11)
    contours, _ = cv2.findContours(image_filter, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours= []
    for contour in contours:
        _, _, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h
        area = cv2.contourArea(contour)
        if area > 4000 and aspect_ratio > 0.5 and aspect_ratio < 1.3:
            filtered_contours.append(contour)
    return filtered_contours

def extract_contour_info(contours):
    contour_info = []

    for contour in contours:
        area = round(cv2.contourArea(contour), 3)
        x, y, w, h = cv2.boundingRect(contour)
        perimeter = round(cv2.arcLength(contour, True), 3)
        # Get the real corners of the contour, could not be the same as the bounding box and could be affine
        # transformed
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int16(box)
        corners = [(corner[0], corner[1]) for corner in box]

        contour_info.append({
            'corners': corners,  # Esquinas en relación a la imagen original
            'center': (round(x + w / 2, 3), round(y + h / 2, 3)),
            'width': round(w, 3),
            'height': round(h, 3),
            'area': area,
            'aspect_ratio': round(w / h, 3) if h != 0 else None,
            'perimeter': perimeter,
            'angle': round(cv2.minAreaRect(contour)[-1], 3)
        })
    return contour_info

def match_contours(red_contours_info, green_contours_info):
    matches = []
    used_red_contours = set()
    used_green_contours = set()

    for red_contour in red_contours_info:
        # Verificar si este contorno rojo ya está emparejado
        if id(red_contour) in used_red_contours:
            continue

        best_match = None
        best_angle_match = float('inf')  # Para encontrar el mejor ángulo
        best_score = float('inf')       # Para desempatar usando área y perímetro

        for green_contour in green_contours_info:
            # Verificar si este contorno verde ya está emparejado
            if id(green_contour) in used_green_contours:
                continue
            # Calcular distancia entre centros
            center_distance = np.linalg.norm(
                np.array(red_contour['center']) - np.array(green_contour['center'])
            )
            # Descartar si la distancia excede 3.5 veces la anchura del contorno rojo
            if center_distance > 3.2 * red_contour['width']:
                continue
            # Descartar si la distancia es menor a 2.8 veces la anchura del contorno rojo
            if center_distance < 2.8 * red_contour['width']:
                continue
            # Calcular diferencia de ángulo
            angle_diff = abs(red_contour['angle'] - green_contour['angle'])
            # Descartar si la diferencia de ángulo es mayor a 10 grados
            # if angle_diff > 20:
            #     continue
            # Calcular diferencias adicionales para desempatar
            area_diff = abs(red_contour['area'] - green_contour['area'])
            perimeter_diff = abs(red_contour['perimeter'] - green_contour['perimeter'])
            # Crear puntaje de desempate
            score = area_diff + perimeter_diff
            # Actualizar el mejor match
            if angle_diff < best_angle_match or (angle_diff == best_angle_match and score < best_score):
                best_angle_match = angle_diff
                best_score = score
                best_match = green_contour
        if best_match:
            matches.append((red_contour, best_match))
            used_red_contours.add(id(red_contour))
            used_green_contours.add(id(best_match))
    return matches

def cut_bounding_box(matched_contours, image):
    extracted_images = []

    for red_contour, green_contour in matched_contours:
        # Obtener los puntos de los contornos
        red_points = np.array(red_contour['corners'])
        green_points = np.array(green_contour['corners'])
        # Combinar los puntos de ambos contornos
        all_points = np.vstack((red_points, green_points))
        # Calcular la bounding box que contenga todos los puntos
        x_min, y_min = np.min(all_points, axis=0)
        x_max, y_max = np.max(all_points, axis=0)
        # Convertir los centros a tuplas de enteros
        red_center = tuple(map(int, red_contour['center']))
        # Calcular el ángulo entre los centros
        x0, y0 = red_contour['center']
        x1, y1 = green_contour['center']
        angle = np.arctan2(y1 - y0, x1 - x0) * 180 / np.pi
        # Rotar la imagen original para que la línea entre los centros sea horizontal
        M = cv2.getRotationMatrix2D(red_center, angle, 1)
        rotated_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
        # Transformar las coordenadas de los contornos según la rotación
        ones = np.ones((red_points.shape[0], 1))
        red_points_homogeneous = np.hstack([red_points, ones])
        green_points_homogeneous = np.hstack([green_points, ones])
        transformed_red_points = (M @ red_points_homogeneous.T).T
        transformed_green_points = (M @ green_points_homogeneous.T).T
        # Combinar los puntos transformados
        transformed_all_points = np.vstack((transformed_red_points, transformed_green_points))
        # Calcular la nueva bounding box en la imagen rotada
        x_min, y_min = np.min(transformed_all_points, axis=0).astype(int)
        x_max, y_max = np.max(transformed_all_points, axis=0).astype(int)
        # Validar las coordenadas para asegurarse de que estén dentro de los límites
        h, w = rotated_image.shape[:2]
        x_min, x_max = max(0, x_min), min(w, x_max)
        y_min, y_max = max(0, y_min), min(h, y_max)
        # Verificar que las dimensiones del recorte sean válidas
        if x_min < x_max and y_min < y_max:
            cropped_image = rotated_image[y_min:y_max, x_min:x_max]

            if cropped_image.size > 0:  # Comprobar que el recorte no sea vacío
                extracted_images.append(cropped_image)
            else:
                print(f"Warning: Empty crop for contours {red_contour['center']} and {green_contour['center']}")
        else:
            print(f"Warning: Invalid crop coordinates for contours {red_contour['center']} and {green_contour['center']}")
    return extracted_images

def threshold_image(image, threshold = 2):
    image_threshold_gaussian = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, threshold)
    # Hacer los bordes más suaves con erosión
    kernel = np.ones((3, 3), np.uint8)
    close_image = cv2.morphologyEx(image_threshold_gaussian,cv2.MORPH_CLOSE, kernel)
    erode_image = cv2.erode(close_image, kernel, iterations=1)
    return erode_image

def segment_contours(contours, image_shape):
    square_contours = []
    rectangular_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        area_ratio = area / (image_shape[0] * image_shape[1])
        _, _, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h
        if 0.5 <= aspect_ratio <= 1.5 and area_ratio > 0.06:
            square_contours.append(contour)
        else:
            rectangular_contours.append(contour)
    return square_contours, rectangular_contours

def filter_inside_contours(contours):
    filtered_contours = []
    for contour in contours:
        parent_contour = None
        for other_contour in contours:
            if contour is not other_contour:
                if all(cv2.pointPolygonTest(other_contour, (int(pt[0][0]), int(pt[0][1])), False) > 0 for pt in contour):
                    if parent_contour is None or cv2.contourArea(other_contour) > cv2.contourArea(parent_contour):
                        parent_contour = other_contour
        if parent_contour is None:
            filtered_contours.append(contour)
    return filtered_contours

def get_contours(thresholded_image, image):
    contours, _ = cv2.findContours(thresholded_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = []
    image_area = image.shape[0] * image.shape[1]
    min_influence = 0.01
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        influence = area / image_area
        if 200 <= area <= 25000 and influence >= min_influence:
            # If the contour is centered close to the border, ignore it
            if x > 10 and y > 10 and x + w < image.shape[1] - 10 and y + h < image.shape[0] - 10:
                filtered_contours.append(contour)
    
    _, rectangular_contours = segment_contours(filtered_contours, image.shape)
    # Filter contours that are inside others
    filtered_contours = filter_inside_contours(rectangular_contours)
    return filtered_contours

def separate_contours_by_segments(contours, image_width):
    segments = [[] for _ in range(4)]
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        center_x = x + w // 2
        segment = center_x // (image_width // 4)
        segments[segment].append(contour)
    return segments

def order_contours(segments):
    ordered_segments = []
    for segment in segments:
        if len(segment) <= 1:
            ordered_segments.append(segment)
        elif len(segment) == 2:
            _, _, w1, h1 = cv2.boundingRect(segment[0])
            _, _, w2, h2 = cv2.boundingRect(segment[1])
            if w1 > h1 and w2 > h2:
                ordered_segments.append(sorted(segment, key=lambda c: cv2.boundingRect(c)[1]))
            elif w1 < h1 and w2 < h2:
                ordered_segments.append(sorted(segment, key=lambda c: cv2.boundingRect(c)[0]))
    return ordered_segments

def get_area_ratio(contours, image):
    image_area = (image.shape[0] * image.shape[1]) / 4
    area_ratios = []
    for contour in contours:
        area = cv2.contourArea(contour)
        area_ratio = area / image_area
        area_ratios.append(round(area_ratio, 3))
    return area_ratios

def get_segment_info(ordered_segments, image):
    segment_info = []
    for segment in ordered_segments:
        num_contours = len(segment)
        orientations = ['horizontal' if cv2.boundingRect(c)[2] > cv2.boundingRect(c)[3] else 'vertical' for c in segment]
        area_ratios = get_area_ratio(segment, image)
        if len(segment) == 2:
            area_ratio_relation = round(cv2.contourArea(segment[0]) / cv2.contourArea(segment[1]), 3)
        else:
            area_ratio_relation = None
        segment_info.append({
            'num_contours': num_contours,
            'orientation': orientations,
            'area_ratios': area_ratios,
            'area_ratios_relation': area_ratio_relation
        })
    return segment_info

def decode_number(segment_info):
    segment_number = ''
    for i in range(4):
        # Validar si el segmento tiene la estructura esperada
        if i >= len(segment_info):
            segment_number += 'X'
            continue
        
        num_contours = segment_info[i]['num_contours']
        orientation = segment_info[i]['orientation']
        area_ratios = segment_info[i]['area_ratios']
        area_ratios_relation = segment_info[i]['area_ratios_relation']
        if num_contours == 0:
            segment_number += '0'
        elif num_contours == 1:
            if orientation[0] == 'horizontal':
                segment_number += '8'
            else:
                if area_ratios[0] < 0.15:
                    segment_number += '1'
                else:
                    segment_number += '5'
        elif num_contours == 2:
            if orientation[0] == 'horizontal':
                if area_ratios_relation > 1.2:
                    segment_number += '7'
                elif area_ratios_relation < 0.8:
                    segment_number += '9'
                else:
                    segment_number += '3'
            else:
                if area_ratios_relation > 1.2:
                    segment_number += '6'
                elif area_ratios_relation < 0.8:
                    segment_number += '4'
                else:
                    segment_number += '2'
        else:
            segment_number += 'X'
    return segment_number


def segmentar_imagen(frame):
    # blur the image
    blurred_image = blur_image(frame, 7)
    # Convertir la imagen a HSV
    hsv_image = convert_image_to_hsv(blurred_image)
    gray_image = convert_image_to_gray(blurred_image)
    # Obtener las máscaras de los colores rojo y verde
    mask_red = get_red_mask(hsv_image)
    mask_green = get_green_mask(hsv_image)
    # Aplicar las máscaras a la imagen original
    segmented_red = apply_mask_to_image(gray_image, mask_red)
    segmented_green = apply_mask_to_image(gray_image, mask_green)
    # Obtener los contornos de las máscaras
    red_contours = find_contours(segmented_red)
    green_contours = find_contours(segmented_green)
    # Extraer información de los contornos
    red_contours_info = extract_contour_info(red_contours)
    green_contours_info = extract_contour_info(green_contours)
    # Emparejar los contornos rojos y verdes
    matched_contours = match_contours(red_contours_info, green_contours_info)
    # Recortar las imágenes de los contornos emparejados
    extracted_images = cut_bounding_box(matched_contours, frame)
    # Devolver las imágenes recortadas
    return extracted_images, matched_contours

def procesar_imagen(frame, extracted_images):
    numbers = []
    for i, image in enumerate(extracted_images):
        # Convertir la imagen a blanco y negro
        gray_image = convert_image_to_gray(image)
        # Blur the image
        blurred_image = (gray_image, 11)
        # Threshold the image
        thresholed_image = threshold_image(blurred_image)
        # Find contours in the thresholded image
        filtered_contours = get_contours(thresholed_image, image)
        # Separate contours by segment
        separated_contours = separate_contours_by_segments(filtered_contours, image.shape[1])
        # Order contours
        ordered_contours = order_contours(separated_contours)
        # Extract segment information
        segment_info = get_segment_info(ordered_contours, image)
        # Extract numbers from the segment information
        number = decode_number(segment_info)
        numbers.append(number)
    return numbers

# Dibujar la bounding box que contenga todos los puntos de los contornos emparejados
def dibujar_bounding_box(contornos_emparejados, imagen, numero):
    # Crear una copia de la imagen original para dibujar la bounding box
    imagen_con_bounding_box = np.copy(imagen)
    for i, (red_contour, green_contour) in enumerate(contornos_emparejados):
            # Obtener los puntos de los contornos
            red_points = np.array(red_contour['corners'])
            green_points = np.array(green_contour['corners'])

            # Combinar los puntos de ambos contornos
            all_points = np.vstack((red_points, green_points))

            # Calcular la bounding box que contenga todos los puntos
            x_min, y_min = np.min(all_points, axis=0)
            x_max, y_max = np.max(all_points, axis=0)

            # Dibujar la bounding box
            cv2.rectangle(imagen_con_bounding_box, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 0, 0), 2)
            # Dibujar el número
            cv2.putText(imagen_con_bounding_box, numero[i], (int(x_min), int(y_min - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)     
    return imagen_con_bounding_box

def get_image_decoded(frame):
    extracted_images, matched_contours = segmentar_imagen(frame)
    numbers = procesar_imagen(frame, extracted_images)
    imagen_con_bounding_box = dibujar_bounding_box(matched_contours, frame, numbers)
    return imagen_con_bounding_box

# Obtención de imagenes a traves del microcontrolador

try:
    # Conectar al flujo de video
    stream = requests.get("http://10.220.151.99:81/", stream=True)
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

                # Decodificar la imagen
                imagen_decodificada = get_image_decoded(frame)

                # Mostrar el fotograma
                cv2.imshow("ESP32-CAM Video", imagen_decodificada)

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


#  Descomentar para usar la camara en vez del microcontrolador

# cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     print("Cannot open camera")
#     exit()

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Can't receive frame (stream end?). Exiting ...")
#         break
#     #Decodificar la imagen
#     imagen_decodificada = get_image_decoded(frame)

#     # image_hsv = convert_image_to_hsv(frame)
#     # mask_red = get_red_mask(image_hsv)
#     # mask_green = get_green_mask(image_hsv)

#     # image_segmented_red = apply_mask_to_image(frame, mask_red)
#     # image_segmented_green = apply_mask_to_image(frame, mask_green)

#     # Mostrar el fotograma
#     cv2.imshow("ESP32-CAM Video", imagen_decodificada)
#     if cv2.waitKey(1) == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()