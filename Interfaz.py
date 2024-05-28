import tkinter as tk
from PIL import Image, ImageTk
import cv2
import threading
import sys

print("Cargando...")

# Función para abrir una nueva ventana
def abrir_nueva_ventana():
    # Configuración de la nueva ventana
    nueva_ventana = tk.Toplevel()
    nueva_ventana.geometry("1280x720")
    nueva_ventana.title("LSMX")
    nueva_ventana.resizable(width=False, height=False)
    nueva_ventana.configure(bg="#2C3E50")

    # Estilo del marco para los elementos de la ventana
    estilo_marco = {
        "bd": 2,
        "relief": "groove",
        "bg": "#34495E",
        "fg": "white",
        "font": ("Helvetica", 12)
    }

    # Creación de etiquetas para mostrar video e imagen ilustrativa
    label_video = tk.Label(nueva_ventana, **estilo_marco)
    label_video.pack(side="left", padx=20, pady=20)

    label_imagen_ilustrativa = tk.Label(nueva_ventana, **estilo_marco)
    label_imagen_ilustrativa.pack(side="right", padx=20, pady=20)

    # Carga y configuración de la imagen ilustrativa
    imagen_ilustrativa = Image.open("LSM.gif")
    imagen_ilustrativa = imagen_ilustrativa.resize((450, 450), Image.LANCZOS)
    imagen_ilustrativa_tk = ImageTk.PhotoImage(imagen_ilustrativa)
    label_imagen_ilustrativa.imgtk = imagen_ilustrativa_tk
    label_imagen_ilustrativa.configure(image=imagen_ilustrativa_tk)

    # Función para cerrar la ventana
    def cerrar_programa():
        nueva_ventana.destroy()
        ventana.destroy()
        sys.exit()

    # Botón para cerrar la ventana
    boton_cerrar = tk.Button(nueva_ventana, text="Cerrar", command=cerrar_programa,
                             font=("Helvetica", 14), bg="#E74C3C", fg="white", bd=0, relief="ridge")
    boton_cerrar.pack(side="bottom", pady=20)

    ventana.withdraw()  # Oculta la ventana principal mientras se muestra la nueva ventana

    # Función para procesar el video en tiempo real
    def procesar_video():
        from ultralytics import YOLO  # Importa el modelo YOLO para detección de objetos
        import SeguimientoManos as sm  # Importa un módulo para seguimiento de manos

        # Configuración de la captura de video
        cap = cv2.VideoCapture(0)
        cap.set(3, 1280)
        cap.set(4, 720)

        # Inicialización del modelo YOLO y del detector de manos
        model = YOLO('alfa.pt')
        detector = sm.detectormanos(Confdeteccion=0.9)

        # Función para actualizar el frame del video
        def actualizar_frame():
            ret, frame = cap.read()
            if not ret:
                return

            # Procesamiento de detección de manos
            frame = detector.encontrarmanos(frame, dibujar=False)
            lista1, bbox, mano = detector.encontrarposicion(frame, ManoNum=0, color=[0, 255, 0])

            # Procesamiento adicional si se detecta una mano
            anotaciones = frame
            if mano == 1:
                xmin, ymin, xmax, ymax = bbox
                margen = 80
                xmin = max(0, xmin - margen)
                ymin = max(0, ymin - margen)
                xmax = min(frame.shape[1], xmax + margen)
                ymax = min(frame.shape[0], ymax + margen)

                recorte = frame[ymin:ymax, xmin:xmax]
                recorte = cv2.resize(recorte, (800, 800), interpolation=cv2.INTER_CUBIC)
                resultados = model.predict(recorte, conf=0.55)

                if len(resultados) != 0:
                    for results in resultados:
                        anotaciones = resultados[0].plot()

            # Conversión del frame para mostrar en la interfaz
            anotaciones_rgb = cv2.cvtColor(anotaciones, cv2.COLOR_BGR2RGB)
            anotaciones_pil = Image.fromarray(anotaciones_rgb)
            anotaciones_tk = ImageTk.PhotoImage(image=anotaciones_pil)

            label_video.imgtk = anotaciones_tk
            label_video.configure(image=anotaciones_tk)

            # Verifica si la ventana sigue abierta para continuar la actualización del frame
            if nueva_ventana.winfo_exists():
                label_video.after(10, actualizar_frame)
            else:
                cap.release()
                cv2.destroyAllWindows()

        actualizar_frame()

    threading.Thread(target=procesar_video).start()  # Inicia el procesamiento de video en un hilo separado

# Configuración de la ventana principal
ventana = tk.Tk()
ventana.geometry("1000x800+200+10")
ventana.title("Menu Principal")
ventana.resizable(width=False, height=False)
ventana.configure(bg="#2C3E50")

# Configuración del fondo de la ventana principal
fondo = tk.PhotoImage(file="Iniciar Demo.png")
fondo1 = tk.Label(ventana, image=fondo)
fondo1.place(x=0, y=0, relwidth=1, relheight=1)

# Botón para abrir la nueva ventana
boton_abrir_ventana = tk.Button(ventana, text="Iniciar Modo Practica", font=("Georgia", 25), fg="white", bg="#3498DB",
                                borderwidth=0, command=abrir_nueva_ventana)
boton_abrir_ventana.place(relx=0.5, rely=0.5, anchor="center")

ventana.mainloop()  # Inicia el bucle principal de la interfaz gráfica
