import cv2
import os
from ultralytics import YOLO
import SeguimientoManos as sm
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

# Función para cargar el modelo YOLO desde un archivo seleccionado
def cargar_modelo(archivo_modelo):
    global model
    model = YOLO(archivo_modelo)

# Función para seleccionar el modelo YOLO desde el menú
def seleccionar_modelo():
    archivo_modelo = tk.filedialog.askopenfilename(
        title="Seleccionar archivo de modelo",
        filetypes=(("Archivos de modelo", "*.pt"), ("Todos los archivos", "*.*"))
    )
    if archivo_modelo:
        cargar_modelo(archivo_modelo)

# Función para salir de la aplicación
def salir():
    cap.release()
    cv2.destroyAllWindows()
    root.quit()

# Lectura de la cámara
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Leer modelo inicial
model = YOLO('alfa.pt')

# Declaración de detector
detector = sm.detectormanos(Confdeteccion=0.9)

# Configurar la ventana de tkinter
root = tk.Tk()
root.title("Lenguaje de Vocales")
root.geometry("1280x720")

# Crear el menú
menubar = tk.Menu(root)
root.config(menu=menubar)

# Crear el menú de archivo
archivo_menu = tk.Menu(menubar, tearoff=0)
menubar.add_cascade(label="Archivo", menu=archivo_menu)
archivo_menu.add_command(label="Seleccionar Modelo", command=seleccionar_modelo)
archivo_menu.add_separator()
archivo_menu.add_command(label="Salir", command=salir)

# Etiqueta para mostrar el video
label_frame = ttk.Label(root)
label_frame.grid(row=0, column=0, padx=10, pady=10)

label_recorte = ttk.Label(root)
label_recorte.grid(row=0, column=1, padx=10, pady=10)

def update_frame():
    ret, frame = cap.read()

    # Extraer informacion de la mano.
    frame = detector.encontrarmanos(frame, dibujar=False)

    # Posicion de solo una mano.
    lista1, bbox, mano = detector.encontrarposicion(frame, ManoNum=0, color=[0, 255, 0])

    anotaciones = frame
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
        recorte = cv2.resize(recorte, (640, 640), interpolation=cv2.INTER_CUBIC)

        # Extraer resultados
        resultados = model.predict(recorte, conf=0.55)

        # Si hay resultados
        if len(resultados) != 0:
            for results in resultados:
                masks = results.masks
                coordenadas = masks

                anotaciones = resultados[0].plot()

        # Convertir la imagen recortada a formato PIL y luego a ImageTk para tkinter
        recorte_rgb = cv2.cvtColor(recorte, cv2.COLOR_BGR2RGB)
        recorte_pil = Image.fromarray(recorte_rgb)
        recorte_tk = ImageTk.PhotoImage(image=recorte_pil)
        label_recorte.imgtk = recorte_tk
        label_recorte.configure(image=recorte_tk)

    # Convertir la imagen anotada a formato PIL y luego a ImageTk para tkinter
    anotaciones_rgb = cv2.cvtColor(anotaciones, cv2.COLOR_BGR2RGB)
    anotaciones_pil = Image.fromarray(anotaciones_rgb)
    anotaciones_tk = ImageTk.PhotoImage(image=anotaciones_pil)
    label_frame.imgtk = anotaciones_tk
    label_frame.configure(image=anotaciones_tk)

    # Actualizar el frame después de un corto intervalo
    root.after(10, update_frame)

# Iniciar la actualización del frame
update_frame()

# Correr la aplicación de tkinter
root.mainloop()

# Liberar la cámara y cerrar ventanas de OpenCV cuando se cierra tkinter
cap.release()
cv2.destroyAllWindows()