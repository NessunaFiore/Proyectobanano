from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QPushButton, QLabel, QFileDialog, QTextEdit)
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt
import cv2


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


class VentanaPrincipal(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Control de Calidad de Bananos 🍌")
        self.resize(900, 600)
        self.detector = None # Se asignará desde main.py
        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout_principal = QHBoxLayout(central_widget)

        # --- Panel Izquierdo (Imagen) ---
        layout_izquierdo = QVBoxLayout()
        self.btn_cargar = QPushButton("Cargar Imagen (JPG/PNG)")
        self.btn_cargar.clicked.connect(self.cargar_imagen)
        
        self.lbl_imagen = QLabel("Sube una imagen para comenzar")
        self.lbl_imagen.setAlignment(Qt.AlignCenter)
        self.lbl_imagen.setStyleSheet("border: 2px dashed #aaa; background-color: #f0f0f0; font-size: 16px;")
        self.lbl_imagen.setMinimumSize(600, 500)
        
        layout_izquierdo.addWidget(self.btn_cargar)
        layout_izquierdo.addWidget(self.lbl_imagen)

        # --- Panel Derecho (Información) ---
        layout_derecho = QVBoxLayout()
        lbl_titulo_info = QLabel("Resultados del Análisis")
        lbl_titulo_info.setStyleSheet("font-size: 16px; font-weight: bold;")
        
        self.txt_resultados = QTextEdit()
        self.txt_resultados.setReadOnly(True)
        self.txt_resultados.setStyleSheet("font-size: 14px; padding: 10px;")
        
        layout_derecho.addWidget(lbl_titulo_info)
        layout_derecho.addWidget(self.txt_resultados)

        layout_principal.addLayout(layout_izquierdo, stretch=2)
        layout_principal.addLayout(layout_derecho, stretch=1)

    def cargar_imagen(self):
        archivo, _ = QFileDialog.getOpenFileName(self, "Seleccionar imagen", "Muestra", "Imágenes (*.png *.jpg *.jpeg)")
        
        if not archivo:
            return

        self.txt_resultados.setText("Procesando imagen, por favor espera...")
        from PySide6.QtWidgets import QApplication
        QApplication.processEvents()

        if self.detector:
            resultado = self.detector.procesar_imagen(archivo)
            
            if resultado["error"]:
                self.txt_resultados.setText(f"{resultado['error']}")
                self.mostrar_imagen(archivo)
            else:
                self.mostrar_imagen_cv2(resultado["imagen"])
                
                texto = resumen_clasificacion(resultado["conteos"])
                self.txt_resultados.setText(texto)

    def mostrar_imagen(self, ruta):
        pixmap = QPixmap(ruta)
        self.lbl_imagen.setPixmap(pixmap.scaled(self.lbl_imagen.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def mostrar_imagen_cv2(self, img_np):
        img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        bytes_per_line = ch * w
        q_img = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        self.lbl_imagen.setPixmap(pixmap.scaled(self.lbl_imagen.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))