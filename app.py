import gradio as gr
import cv2
import numpy as np
from modelo_ia import DetectorBananos
from ventana import resumen_clasificacion

# Cargamos el modelo una sola vez al iniciar la web
print("Cargando modelo...")
detector = DetectorBananos()
print("¡Modelo cargado!")

def analizar_imagen(imagen):
    if imagen is None:
        return None, "Por favor, sube una imagen."
    
    # Convertir la imagen de Gradio (RGB) a BGR (OpenCV)
    img_bgr = cv2.cvtColor(imagen, cv2.COLOR_RGB2BGR)
    
    # Guardar temporalmente para que tu función original la procese
    ruta_temp = "temp_image.jpg"
    cv2.imwrite(ruta_temp, img_bgr)
    
    # Llamar a tu función original
    resultado = detector.procesar_imagen(ruta_temp)
    
    if resultado["error"]:
        return None, resultado["error"]
    
    # Convertir la imagen resultante a RGB para mostrarla en la web
    img_resultado = cv2.cvtColor(resultado["imagen"], cv2.COLOR_BGR2RGB)
    
    # Generar el resumen (usando la función de ventana.py)
    texto_resumen = resumen_clasificacion(resultado["conteos"])
    
    return img_resultado, texto_resumen

# Interfaz web
interfaz = gr.Interface(
    fn=analizar_imagen,
    inputs=gr.Image(type="numpy", label="Sube una imagen de bananos"),
    outputs=[
        gr.Image(label="Resultado del análisis"),
        gr.Textbox(label="Resumen")
    ],
    title="🍌 Control de Calidad de Bananos",
    description="Sube una imagen para detectar y clasificar bananos (maduros, verdes, podridos)."
)

if __name__ == "__main__":
    interfaz.launch()