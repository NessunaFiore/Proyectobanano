import streamlit as st
import cv2
import numpy as np
from PIL import Image
from modelo_ia import DetectorBananos

# --- Configuración de la página ---
st.set_page_config(page_title="Control de Calidad de Bananos", page_icon="🍌", layout="wide")

st.title("🍌 Control de Calidad de Bananos")
st.write("Integrantes: Abraham Ariza, Isac Cruz y Juan Suárez Brango")
st.write("Sube una imagen para detectar y clasificar bananos (maduros, verdes, podridos).")

# --- Función copiada de ventana.py para evitar importar PySide6 ---
def resumen_clasificacion(conteos):
    total = sum(conteos.values())
    if total == 0:
        return "No se detectaron plátanos en la imagen."

    nombres = {1: "maduros", 2: "verdes", 3: "podridos"}
    mayor = max(conteos.values())
    dominantes = [
        nombres[clase] for clase, cantidad in conteos.items()
        if cantidad == mayor
    ]
    if len(dominantes) > 1:
        return (
            "La muestra presenta un empate entre plátanos "
            f"{', '.join(dominantes)}."
        )
    if mayor > total / 2:
        return (
            "En la imagen, la muestra corresponde en su mayoría "
            f"a plátanos {dominantes[0]}."
        )
    return (
        "En la imagen, el tipo de plátano más frecuente es: "
        f"{dominantes[0]}."
    )

# --- Carga del modelo (con caché) ---
@st.cache_resource
def cargar_modelo():
    print("Cargando modelo...")
    detector = DetectorBananos()
    print("Modelo cargado exitosamente.")
    return detector

# Mostrar un spinner mientras carga el modelo la primera vez
with st.spinner("Cargando modelos de Inteligencia Artificial (esto puede tardar unos minutos la primera vez)..."):
    detector = cargar_modelo()

# --- Interfaz de usuario ---
archivo_subido = st.file_uploader("Elige una imagen...", type=["jpg", "jpeg", "png"])

if archivo_subido is not None:
    # Leer la imagen
    imagen = Image.open(archivo_subido).convert("RGB")
    
    # (Se eliminó la línea que mostraba la imagen original aquí)
    
    # Guardar temporalmente para tu función procesar_imagen
    ruta_temp = "temp_image.jpg"
    imagen.save(ruta_temp)
    
    with st.spinner("Analizando imagen..."):
        resultado = detector.procesar_imagen(ruta_temp)
        
        if resultado["error"]:
            st.error(resultado["error"])
        else:
            # Mostrar SOLO la imagen procesada
            img_resultado = cv2.cvtColor(resultado["imagen"], cv2.COLOR_BGR2RGB)
            st.image(img_resultado, caption="Resultado del análisis", use_container_width=True)
            
            # Mostrar resumen (texto verde)
            texto_resumen = resumen_clasificacion(resultado["conteos"])
            st.success(texto_resumen)
            
            
