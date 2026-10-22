# TP2 - Reconstrucción 3D

I308 Visión Artificial - Universidad de San Andrés.

Reconstrucción métrica de un caracol marino mediante calibración estéreo,
disparidad SGBM y pose respecto de un tablero ChArUco, con OpenCV y Open3D.

## Estructura

- `TP2.ipynb`: ejecución del pipeline y visualización de resultados.
- `src/`: funciones de calibración, disparidad, pose, reconstrucción y fusión.
- `data_propia/calib/`: 52 pares de calibración, checkerboard de 9 x 6
  esquinas internas y cuadros de 28 mm.
- `data_propia/captures/`: 83 pares del objeto, ChArUco de 9 x 13 cuadros
  de 42 mm, marcadores de 31.5 mm y diccionario `DICT_5X5_100`.
- `caracol.ply`: nube de puntos optimizada en formato binario.
- `caracol_malla.ply`: malla de Poisson.
- `figures/`: figuras de resultados.
- `informe/`: informe en LaTeX y PDF.

## Instalación

Python 3.11. Versiones de dependencias en `requirements.txt`.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

En Linux o macOS, activar el entorno con `source .venv/bin/activate`.

## Ejecución

Abrir `TP2.ipynb` desde la carpeta del proyecto, seleccionar el kernel del
entorno, reiniciarlo y ejecutar las celdas en orden. Las ventanas interactivas de Open3D
deben cerrarse para continuar. Las visualizaciones requieren OpenGL.

Los parámetros de los tableros, la caja del objeto y los filtros se definen
en el notebook. Las dimensiones se expresan en milímetros. `REAL_SIZE_MM`
permite comparar con medidas físicas cuando se disponga de ellas.

La celda de ICP refina la alineación de las nubes antes del filtrado final;
se puede omitir para conservar la fusión preliminar.
La validación opcional de ICP usa vistas separadas del ajuste y no modifica
las nubes de la reconstrucción.

El notebook recalibra a partir de las imágenes. Los PKL originales se
conservan como referencia y se pueden leer con `load_calibration`.

Para visualizar la nube exportada, abrir `caracol.ply` con MeshLab.

Para compilar el informe, desde `informe/` ejecutar dos veces:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error informe.tex
```
