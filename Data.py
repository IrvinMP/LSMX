# Importacion de Librerias.
import cv2
import os
# Importacion del seguimiento de las Manos.
import SeguimientoManos as sm

# Creacion de la carpeta
nombre = 'Prueba'
direccion = 'C:/Users/magnu/OneDrive/Escritorio/LenguajeMX/pythonProject1/data'
carpeta = direccion + '/' + nombre

# En caso de la carpeta no este creada.
if not os.path.exists(carpeta):
    print("CARPETA CREADA: ", carpeta)
    # Se crea la carpeta.
    os.makedirs(carpeta)

# Lectura de la camara
cap = cv2.VideoCapture(0)
# Cambios de resolucion.
cap.set(3, 1280)
cap.set(4, 720)

# Contador
cont = 0

# Declaracion de detector
detector = sm.detectormanos(Confdeteccion=0.9)

while True:

    # Realizar la lectura de la camara.
    ret, frame = cap.read()

    # Extraer informacion de la mano.
    frame = detector.encontrarmanos(frame, dibujar= False)

    # Posicion de solo una mano.
    lista1, bbox, mano = detector.encontrarposicion(frame, ManoNum=0,color=[0,255,0])

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
        #recorte = cv2.resize(recorte, (640,640), interpolation=cv2.INTER_CUBIC)

        # Almacenar imagenes.
        cv2.imwrite(carpeta + "/Z_{}.jpg".format(cont), recorte)

        # Aumento del contador.
        cont = cont + 1

        cv2.imshow("RECORTE", recorte)




    # Mostar Frames por segundo.
    cv2.imshow("LENGUAJE DE VOCALES", frame)

    # Leer teclado.
    t = cv2.waitKey(1)
    if t == 27 or cont == 100:
        break

cap.release()
cv2.destroyAllWindows()