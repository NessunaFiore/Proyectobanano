import streamlit as st
import cv2
import numpy as np
from PIL import Image
from modelo_ia import DetectorBananos
from ventana import resumen_clasificacion

# Configuración de la página
st.set_page_config(page_title="Control de Calidad de Bananos", page_icon="🍌", layout="wide")

st.title("🍌 Control de Calidad de Bananos")
st.write("Sube una imagen para detectar y clasificar bananos (maduros, verdes, podridos).")

# Cargar el modelo (se guarda en caché para no recargarlo cada vez)
@st.cache_resource
def cargar_modelo():
    return DetectorBananos()

detector = cargar_modelo()

# Componente para subir imagen
archivo_subido = st.file_uploader("Elige una imagen...", type=["jpg", "jpeg", "png"])

if archivo_subido is not None:
    # Leer la imagen
    imagen = Image.open(archivo_subido).convert("RGB")
    st.image(imagen, caption="Imagen original", use_column_width=True)
    
    # Guardar temporalmente para tu función procesar_imagen
    ruta_temp = "temp_image.jpg"
    imagen.save(ruta_temp)
    
    with st.spinner("Analizando imagen..."):
        resultado = detector.procesar_imagen(ruta_temp)
        
        if resultado["error"]:
            st.error(resultado["error"])
        else:
            # Mostrar resultado
            img_resultado = cv2.cvtColor(resultado["imagen"], cv2.COLOR_BGR2RGB)
            st.image(img_resultado, caption="Resultado del análisis", use_column_width=True)
            
            # Mostrar resumen
            texto_resumen = resumen_clasificacion(resultado["conteos"])
            st.success(texto_resumen)
            
            # Mostrar conteos detallados
            st.write("**Conteo detallado:**")
            col1, col2, col3 = st.columns(3)
            col1.metric("Maduros", resultado["conteos"].get(1, 0))
            col2.metric("Verdes", resultado["conteos"].get(2, 0))
            col3.metric("Podridos", resultado["conteos"].get(3, 0))