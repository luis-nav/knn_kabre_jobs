# knn_kabre_jobs

Proyecto de clasificacion y recomendaciones para trabajos del cluster Kabre usando `k`-Nearest Neighbors (`k`NN) implementado desde cero.

El entregable principal es el notebook `kabre_jobs_xai.ipynb`, estructurado con la metodologia CRISP-DM.

## Requisitos

- Python 3.11 o compatible
- Archivo `dataset.csv` ubicado en la raiz del repositorio

## Instalacion rapida

1. Crear y activar un entorno virtual:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Instalar dependencias:

```bash
pip install -r requirements.txt
```

3. Registrar el kernel del entorno virtual en Jupyter:

```bash
python -m ipykernel install --user --name knn-kabre --display-name "knn-kabre"
```

## Dataset

Coloque el archivo `dataset.csv` en la raiz del proyecto:

```text
knn_kabre_jobs/
  dataset.csv
  kabre_jobs_xai.ipynb
```

El notebook espera el dataset con separador `|` y la columna `Submit` en formato de fecha.

Fuente usada en el notebook:

- <https://raw.githubusercontent.com/DylanBC09/Project-1-CeNAT/refs/heads/main/dataset.csv>

## Ejecucion del notebook

1. Inicie JupyterLab:

```bash
jupyter lab
```

2. Abra `kabre_jobs_xai.ipynb`.
3. Seleccione el kernel `knn-kabre` o el kernel del entorno virtual activo.
4. Ejecute las celdas en orden.

## Nota sobre cross-validation

El notebook ya deja seleccionado el valor de `k` utilizado para la evaluacion final (`k=10`).

La celda de cross-validation puede tardar bastante tiempo. Si solo se quiere recorrer el resto del notebook y reproducir las secciones posteriores, no es necesario volver a ejecutarla siempre que se conserve la seleccion de `k` ya documentada en el notebook.

## Estructura del repositorio

- `kabre_jobs_xai.ipynb`: entregable principal con CRISP-DM, EDA, modelado, evaluacion y conclusiones.
- `src/`: modulos auxiliares del proyecto.
- `main.py`: script alternativo de ejecucion del pipeline.

## Alcance actual

La referencia principal del proyecto es el notebook. Las decisiones metodologicas y los resultados reportados deben tomarse de `kabre_jobs_xai.ipynb`.
