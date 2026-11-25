# modelo_ia.py
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torchvision.transforms.functional as F
from PIL import Image
import numpy as np
import cv2
import os
from ultralytics import YOLO


def clasificar_por_color(pixeles_rgb, etiqueta_modelo):
    if len(pixeles_rgb) == 0:
        return etiqueta_modelo

    fruta_hsv = cv2.cvtColor(
        pixeles_rgb.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV
    ).reshape(-1, 3)
    h, s, v = fruta_hsv.T

    # Ignora brillos casi blancos, pero conserva sombras oscuras para detectar manchas.
    pixeles_validos = (s >= 20) | (v < 55)
    if not np.any(pixeles_validos):
        return etiqueta_modelo

    h, s, v = h[pixeles_validos], s[pixeles_validos], v[pixeles_validos]
    proporcion_marron = np.mean(
        (h >= 5) & (h <= 25) & (s >= 60) & (v < 145)
    )
    proporcion_negro = np.mean((v < 55) & (s < 80))
    proporcion_verde = np.mean(
        (h >= 28) & (h <= 90) & (s >= 25) & (v >= 40)
    )
    proporcion_amarillo = np.mean(
        (h >= 8) & (h <= 38) & (s >= 30) & (v >= 60)
    )

    if proporcion_marron >= 0.45 or proporcion_negro >= 0.05:
        return 3
    if proporcion_marron >= 0.15 or proporcion_negro >= 0.02:
        return etiqueta_modelo
    if (
        proporcion_amarillo >= 0.4
        and proporcion_amarillo - proporcion_verde >= 0.25
    ):
        return 1
    if (
        proporcion_verde >= 0.4
        and proporcion_verde - proporcion_amarillo >= 0.25
    ):
        return 2
    return etiqueta_modelo


def dibujar_contorno(img, mascara, color):
    contornos, _ = cv2.findContours(
        mascara.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    poligonos = [
        cv2.approxPolyDP(
            contorno, 0.005 * cv2.arcLength(contorno, True), True
        )
        for contorno in contornos
        if cv2.contourArea(contorno) > 0
    ]
    if poligonos:
        overlay = img.copy()
        overlay[mascara] = color
        cv2.addWeighted(overlay, 0.28, img, 0.72, 0, img)
        grosor = max(3, int(round(min(img.shape[:2]) * 0.003)))
        cv2.polylines(img, poligonos, True, color, grosor)


def obtener_mascara(imagen_rgb, mascara_modelo, caja):
    alto, ancho = mascara_modelo.shape
    x1, y1, x2, y2 = np.rint(caja).astype(int)
    x1, x2 = max(0, x1), min(ancho, x2)
    y1, y2 = max(0, y1), min(alto, y2)
    area_caja = max(0, x2 - x1) * max(0, y2 - y1)

    if np.count_nonzero(mascara_modelo) >= max(16, int(area_caja * 0.005)):
        return mascara_modelo
    if x2 - x1 < 3 or y2 - y1 < 3:
        return mascara_modelo

    roi = imagen_rgb[y1:y2, x1:x2]
    alto_roi, ancho_roi = roi.shape[:2]
    escala = min(1.0, 800 / max(alto_roi, ancho_roi))
    if escala < 1:
        roi = cv2.resize(
            roi,
            (round(ancho_roi * escala), round(alto_roi * escala)),
            interpolation=cv2.INTER_AREA,
        )

    alto_roi, ancho_roi = roi.shape[:2]
    if ancho_roi < 3 or alto_roi < 3:
        return mascara_modelo
    mascara_gc = np.zeros((alto_roi, ancho_roi), dtype=np.uint8)
    modelo_fondo = np.zeros((1, 65), dtype=np.float64)
    modelo_frente = np.zeros((1, 65), dtype=np.float64)
    cv2.grabCut(
        roi,
        mascara_gc,
        (1, 1, ancho_roi - 2, alto_roi - 2),
        modelo_fondo,
        modelo_frente,
        5,
        cv2.GC_INIT_WITH_RECT,
    )
    frente = np.isin(mascara_gc, (cv2.GC_FGD, cv2.GC_PR_FGD))
    if not np.any(frente):
        return mascara_modelo

    cantidad_componentes, componentes, estadisticas, _ = (
        cv2.connectedComponentsWithStats(frente.astype(np.uint8), 8)
    )
    if cantidad_componentes > 1:
        componente_mayor = 1 + np.argmax(
            estadisticas[1:, cv2.CC_STAT_AREA]
        )
        frente = componentes == componente_mayor

    if frente.shape != (y2 - y1, x2 - x1):
        frente = cv2.resize(
            frente.astype(np.uint8),
            (x2 - x1, y2 - y1),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

    mascara = np.zeros((alto, ancho), dtype=bool)
    mascara[y1:y2, x1:x2] = frente
    return mascara


class DetectorBananos:
    def __init__(self, ruta_pesos="maskrcnn_bananos.pth"):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        # 1. Cargar YOLO
        self.yolo = YOLO('yolov8n.pt')
        
        # 2. Cargar Mask R-CNN
        self.model = self._crear_modelo(4)
        self.model.load_state_dict(torch.load(ruta_pesos, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def _crear_modelo(self, num_classes):
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)
        return model

    def procesar_imagen(self, ruta_imagen, umbral=0.3):
        img = Image.open(ruta_imagen).convert("RGB")
        
        # --- NUEVA LÓGICA PARA IMÁGENES DE BAJA RESOLUCIÓN ---
        ancho, alto = img.size
        min_dim = 800  # Tamaño mínimo del lado más corto
        
        if ancho < min_dim or alto < min_dim:
            escala = min_dim / min(ancho, alto)
            nuevo_ancho = int(ancho * escala)
            nuevo_alto = int(alto * escala)
            
            # Redimensionamos con interpolación de alta calidad
            img = img.resize((nuevo_ancho, nuevo_alto), Image.Resampling.LANCZOS)
            print(f"Imagen redimensionada de {ancho}x{alto} a {nuevo_ancho}x{nuevo_alto} para mejorar la detección.")
        # -----------------------------------------------------

        img_tensor = F.to_tensor(img).to(self.device)
        img_np = np.array(img)
        
        with torch.no_grad():
            prediccion = self.model([img_tensor])[0]
            
        img_resultado = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        conteos = {1: 0, 2: 0, 3: 0}
        
        for i in range(len(prediccion['scores'])):
            score = prediccion['scores'][i].item()
            if score > umbral:
                caja = prediccion['boxes'][i].cpu().numpy()
                x1, y1 = map(int, caja[:2])
                
                # Análisis de Color
                mask = prediccion['masks'][i, 0].cpu().numpy() > 0.5
                mask = obtener_mascara(img_np, mask, caja)
                pixeles_fruta = img_np[mask]
                
                label = clasificar_por_color(
                    pixeles_fruta, prediccion['labels'][i].item()
                )

                if label in conteos:
                    conteos[label] += 1
                    color = (
                        (0, 255, 255) if label == 1
                        else (0, 255, 0) if label == 2
                        else (0, 0, 255)
                    )
                    clase_nombre = (
                        "maduro" if label == 1
                        else "verde" if label == 2
                        else "podrido"
                    )
                    
                    dibujar_contorno(img_resultado, mask, color)
                    cv2.putText(img_resultado, f"{clase_nombre} {score:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return {"error": None, "imagen": img_resultado, "conteos": conteos}
