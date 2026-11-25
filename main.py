#Main
import sys
from PySide6.QtWidgets import QApplication
from ventana import VentanaPrincipal
from modelo_ia import DetectorBananos

if __name__ == '__main__':
    print("Iniciando aplicación...")
    print("Cargando modelo de Inteligencia Artificial (esto puede tardar unos segundos)...")
    
    # 1. Iniciar la interfaz
    app = QApplication(sys.argv)
    ventana = VentanaPrincipal()
    
    # 2. Cargar el modelo y pasárselo a la ventana
    try:
        detector = DetectorBananos()
        ventana.detector = detector
        print("¡Modelo cargado con éxito!")
    except Exception as e:
        print(f"Error cargando el modelo: {e}")
        
    ventana.show()
    sys.exit(app.exec())