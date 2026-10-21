# TP2 - Reconstrucción 3D (I308 Visión Artificial, UdeSA)

Reconstrucción 3D métrica de un objeto (caracol marino) a partir de un par
estéreo ELP-USB3D1080P02-H120, usando un tablero ChArUco como marco de
referencia global. Pipeline: calibración y rectificación estéreo → disparidad
SGBM → nube local → pose con `solvePnP` → fusión en el marco del tablero →
optimización con Open3D → exportación `.ply` → medición en milímetros.

## Estructura

```
TP2.ipynb              Notebook principal (ejecuta todo el pipeline en orden)
src/
  calibration.py       Calibración estéreo (checkerboard), refinamiento R/T (ChArUco), mapas de rectificación
  disparity.py         Disparidad con StereoSGBM
  reconstruction.py    Nube local (reprojectImageTo3D) y transformación al mundo
  pose.py              Pose de la cámara respecto del ChArUco (solvePnP + Rodrigues)
  pipeline.py          Procesamiento de todas las capturas, una a la vez
  fusion.py            Corrección por plano del tablero, fusión, ICP, optimización y medición
  visualization.py     Figuras (matplotlib) y capturas de Open3D sin ventana
data_propia/
  calib/               Pares left_N.jpg / right_N.jpg con checkerboard 9x6 (cuadros de 28 mm)
  captures/            83 pares del objeto sobre el ChArUco (9x13 cuadros de 42 mm, marcadores de 31.5 mm, DICT_5X5_100)
figures/               Figuras generadas por el notebook (usadas en el informe)
output/caracol.ply     Nube final optimizada, PLY binario (~12 MB, entregable)
output/caracol_malla.ply  Malla de Poisson opcional (~11 MB)
informe/               Informe en LaTeX (informe.tex) y PDF compilado
```

## Instalación

Probado con Python 3.11.2 en Windows 11.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

Versiones usadas (`requirements.txt`): numpy 2.4.6, opencv-contrib-python
5.0.0.93 (hace falta el módulo `aruco`), open3d 0.20.0, matplotlib 3.11.2.

## Uso

Abrir `TP2.ipynb` con el kernel del entorno y ejecutar todas las celdas en
orden (Run All). El notebook:

1. Calibra el par estéreo y calcula los mapas de rectificación (~20 s).
2. Calcula disparidad, nube local y pose de una captura de ejemplo.
3. Procesa las 83 capturas (dos veces: la celda original de 4.2 y la de
   fusion), las fusiona y optimiza. En total ~15-20 min en CPU; el cuello
   de botella es SGBM a 1920x1080.
4. Exporta `output/caracol.ply` (binario) y guarda las figuras en `figures/`.
5. Imprime las dimensiones estimadas del objeto en milímetros
   (resultado: alto ~122 mm, largo ~219 mm, ancho ~202 mm). Para el error
   relativo, completar `REAL_SIZE_MM` en la última celda con las medidas reales.

La corrección por plano usa RANSAC, que en Open3D no es determinista: entre
corridas las medidas varían ~1 %.

Las celdas `show_cloud` / `show_camera_poses` abren ventanas interactivas de
Open3D; en entornos sin pantalla (Colab) usar las celdas que llaman a
`render_geometries`, que guardan PNG sin abrir ventana.

Para ver la nube final: abrir `output/caracol.ply` con MeshLab.

Para compilar el informe: `cd informe && pdflatex informe.tex` (dos veces).

Los archivos `data_propia/stereo_calibration.pkl` y `stereo_maps.pkl` son una
calibración previa; el notebook no los usa (recalibra con OpenCV) pero se
conservan como referencia (`calibration.load_calibration` los lee).
