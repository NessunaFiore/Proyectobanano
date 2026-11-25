import unittest

import cv2
import numpy as np

from modelo_ia import clasificar_por_color, dibujar_contorno, obtener_mascara
from ventana import resumen_clasificacion


class ClasificarPorColorTests(unittest.TestCase):
    def test_clasifica_distintos_tonos_amarillos_como_maduros(self):
        tonos_amarillos = np.array(
            [[235, 205, 50], [245, 220, 90], [210, 170, 45]],
            dtype=np.uint8,
        )

        self.assertEqual(clasificar_por_color(tonos_amarillos, 2), 1)

    def test_clasifica_tonos_verdes_como_inmaduros(self):
        tonos_verdes = np.array(
            [[95, 175, 55], [125, 185, 65], [70, 130, 45]],
            dtype=np.uint8,
        )

        self.assertEqual(clasificar_por_color(tonos_verdes, 1), 2)

    def test_clasifica_manchas_marrones_y_negras_como_podrido(self):
        tonos = np.array(
            [[95, 55, 25]] * 5 + [[25, 20, 20]] * 2 + [[235, 205, 50]] * 3,
            dtype=np.uint8,
        )

        self.assertEqual(clasificar_por_color(tonos, 1), 3)

    def test_conserva_clase_del_modelo_si_el_color_es_ambiguo(self):
        grises = np.array([[128, 128, 128], [145, 145, 145]], dtype=np.uint8)

        self.assertEqual(clasificar_por_color(grises, 2), 2)

    def test_dibuja_contorno_de_mascara_y_no_un_rectangulo(self):
        imagen = np.zeros((24, 24, 3), dtype=np.uint8)
        mascara = np.zeros((24, 24), dtype=bool)
        mascara[5:19, 7:17] = True

        dibujar_contorno(imagen, mascara, (0, 255, 255))

        self.assertTrue(np.any(imagen[10, 10]))
        np.testing.assert_array_equal(imagen[5, 10], (0, 255, 255))
        self.assertFalse(np.any(imagen[:3, :]))

    def test_crea_mascara_si_el_modelo_devuelve_una_mascara_vacia(self):
        imagen = np.full((100, 100, 3), 255, dtype=np.uint8)
        cv2.ellipse(imagen, (50, 50), (25, 35), 0, 0, 360, (30, 180, 240), -1)
        mascara_vacia = np.zeros((100, 100), dtype=bool)

        mascara = obtener_mascara(
            imagen, mascara_vacia, np.array([10, 10, 90, 90])
        )

        self.assertTrue(mascara[50, 50])
        self.assertFalse(mascara[0, 0])

    def test_resumen_muestra_la_mayoria_sin_cantidades(self):
        resumen = resumen_clasificacion({1: 5, 2: 2, 3: 1})

        self.assertEqual(
            resumen,
            "En la imagen, la muestra corresponde en su mayoría a plátanos maduros.",
        )
        self.assertNotIn("5", resumen)

    def test_resumen_indica_empate_y_caso_sin_detecciones(self):
        self.assertIn("empate", resumen_clasificacion({1: 2, 2: 2, 3: 0}))
        self.assertEqual(
            resumen_clasificacion({1: 0, 2: 0, 3: 0}),
            "No se detectaron plátanos en la imagen.",
        )


if __name__ == "__main__":
    unittest.main()
