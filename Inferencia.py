#Importacion de librerias
import cv2
import os
from ultralytics import YOLO
# Importacion del seguimiento de las Manos.
import SeguimientoManos as sm

# Lectura de la camara
cap = cv2.VideoCapture(0)
# Cambios de resolucion.
cap.set(3, 1280)
cap.set(4, 720)

#Leer modelo
model = YOLO('alfa.pt')

# Declaracion de detector
detector = sm.detectormanos(Confdeteccion=0.9)

while True:

    # Realizar la lectura de la camara.
    ret, frame = cap.read()

    # Extraer informacion de la mano.
    frame = detector.encontrarmanos(frame, dibujar= False)

    # Posicion de solo una mano.
    lista1, bbox, mano = detector.encontrarposicion(frame, ManoNum=0, color=[0,255,0])

    #SI hay mano
    if mano == 1:
        # Extraer informacion del recuadro.
        xmin, ymin, xmax, ymax = bbox

        # Asignamos margen
        xmin = xmin - 40
        ymin = ymin - 40
        xmax = xmax + 40
        ymax = ymax + 40

        # Recorte de la mano.
        recorte = frame[ymin:ymax, xmin:xmax]

        # Redimensionamiento
        recorte = cv2.resize(recorte, (800,800), interpolation=cv2.INTER_CUBIC)

        # Extraer resiltados
        resultados = model.predict(recorte, conf=0.55)

        #Si hay resultados
        if len(resultados) != 0:
            for results in resultados:
                masks = results.masks
                coordenadas = masks

                anotaciones = resultados[0].plot()



        cv2.imshow("RECORTE", anotaciones)




    # Mostar Frames por segundo.
    cv2.imshow("LENGUAJE DE VOCALES", frame)

    # Leer teclado.
    t = cv2.waitKey(1)
    if t == 27:
        break

cap.release()
cv2.destroyAllWindows()