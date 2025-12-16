# Proyecto: Predicción de Riesgo de Incumplimiento de Crédito

# MACHINE LEARNING_001D
**Integrantes: Martin Espinoza y Jeremias Fernandez**

---------------------------------------------------------------

# Descripción General

Este proyecto implementa un pipeline completo de Machine Learning para predecir la probabilidad de incumplimiento de crédito (default) de un cliente, utilizando el dataset Home Credit Default Risk.

# El desarrollo sigue estrictamente la metodología CRISP-DM, cubriendo todas sus fases:

-Comprensión del negocio

-Comprensión de los datos

-Preparación de los datos

-Modelado

-Evaluación

-Despliegue (API REST)

**El modelo final se expone mediante una API REST desarrollada con FastAPI, permitiendo evaluar solicitudes de crédito en tiempo real.**

---------------------------------------------------------------

#Objetivo del Proyecto

**Construir un sistema predictivo que:**

-Estime la probabilidad de default de un solicitante.

-Clasifique automáticamente la solicitud en Aprobado, Revisión o Rechazado.

-Permita su uso a través de una API documentada y funcional.


---------------------------------------------------------------
# Estructura del Proyecto


project_root/
│
├── artifacts/                  **Modelos y datasets procesados**
│   ├── best_model.pkl
│   ├── X_train_processed.parquet
│   ├── y_train_processed.parquet
│   └── X_columns.json
│
├── data/                       **Datos originales**
│   ├── application_.parquet
│   ├── bureau.parquet
│   ├── bureau_balance.parquet
│   ├── credit_card_balance.parquet
│   ├── installments_payments.parquet
│   ├── POS_CASH_balance.parquet
│   └── previous_application.parquet
│
├── outputs/                    **Resultados y evidencias**
│   ├── evaluation_report.json
│   ├── classification_report.txt
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── pr_curve.png
│   ├── feature_importance.png
│   ├── payload_aprobado.json
│   ├── payload_revision.json
│   └── payload_rechazado.json
│
├── 01_data_understanding/      **Fase 1: Análisis**
│   └── 01_data_understanding.py
│
├── 02_data_preparation/        **Fase 2: Ingeniería de Características**
│   └── 02_data_preparation.py
│
├── 03_modeling/                **Fase 3: Entrenamiento**
│   └── 03_modeling.py
│
├── 04_evaluation/              **Fase 4: Métricas**
│   └── 04_evaluation.py
│
├── 05_deployment/              **Fase 5: API REST**
│   ├── 05_deployment.py
│   └── payloads/
│
├── requirements.txt
└── README.md

---------------------------------------------------------------

# Metodología CRISP-DM


**Como Fase 1 - Data Understanding**

**En Archivo: 01_data_understanding.py**

-Análisis exploratorio del dataset principal.

-Evaluación de distribución de la variable objetivo (TARGET).

-Identificación de desbalance de clases.

-Análisis de valores nulos y correlaciones.

-Generación de gráficos exploratorios.


**Resultados:**

01_target_distribution.png

02_correlation_matrix.png

-----------------------------------

# Fase 2 - Data Preparation

**Archivo: 02_data_preparation.py**

-Integración de múltiples tablas 1:N a nivel cliente (SK_ID_CURR).

-Feature Engineering avanzado (pagos, atrasos, proporciones, conteos).

-Imputación simple:

--Numéricos → mediana

--Categóricos → moda / “MISSING”

-One-Hot Encoding.

-Alineación final de features.


# Salida:

-X_train_processed.parquet

-y_train_processed.parquet

-X_columns.json

-----------------------------------

# Fase 3 – Modeling

**Archivo: 03_modeling.py**

-División Train/Test estratificada (80/20).

-Manejo de desbalance:

--class_weight="balanced" en RandomForest

--scale_pos_weight en LightGBM

-Validación cruzada StratifiedKFold (5 folds).

-Modelos entrenados:

--RandomForest (baseline)

--LightGBM (modelo campeón)

-Selección por ROC-AUC promedio.

# Modelo final:

-best_model.pkl

-----------------------------------

# Fase 4 – Evaluation

**Archivo: 04_evaluation.py**

-Evaluación en conjunto hold-out.

-Búsqueda de umbral óptimo de decisión.

-Métricas:

--ROC-AUC

--PR-AUC

--F1-Score

--Precision / Recall

-Generación de gráficos y reportes.

# Salida:

-evaluation_report.json

-classification_report.txt

-confusion_matrix.png

-roc_curve.png

-pr_curve.png

-feature_importance.png

-----------------------------------

# Fase 5 – Deployment

**Archivo: 05_deployment.py**

-API REST con FastAPI.

-Carga automática del modelo entrenado.

-Endpoint /predict para inferencia.

-Endpoint /health para monitoreo.

-Clasificación en 3 estados:

--Aprobado

--Revisión

--Rechazado

---------------------------------------------------------------

# Ejecución del Proyecto

---------------------------------------
**Instalación de dependencias**

pip install -r requirements.txt

---------------------------------------

# Ejecución del pipeline completo

python 01_data_understanding.py
python 02_data_preparation.py
python 03_modeling.py **aca es donde mas se demora**
python 04_evaluation.py

---------------------------------------

# Levantar la API

uvicorn 05_deployment:app --reload --host 127.0.0.1 --port 8000

---------------------------------------

# Acceder a la documentación interactiva:

http://127.0.0.1:8000/docs

---------------------------------------

# Uso de la API

---------------------------------------
**Endpoint de salud**

GET /health

**Respuesta**

{
  "status": "ok",
  "model_loaded": true,
  "has_expected_columns": true,
  "threshold": 0.38
}

---------------------------------------

# Endpoint de predicción

POST /predict

**Ejemplo de Request**

{
  "data": {
    "AMT_INCOME_TOTAL": 40000,
    "AMT_CREDIT": 900000,
    "AMT_ANNUITY": 70000,
    "DAYS_BIRTH": -8000,
    "DAYS_EMPLOYED": -100
  }
}


**Ejemplo de Response**

{
  "probabilidad_default": 0.0579,
  "decision": "Aprobado",
  "threshold": 0.38
}

---------------------------------------

# Resultados Relevantes

ROC-AUC Test: 0.78

PR-AUC Test: 0.27

El modelo logra un buen equilibrio entre recall y precision, priorizando la detección de clientes riesgosos.


---------------------------------------

# Conclusión

Este proyecto demuestra un flujo completo, realista y profesional de Machine Learning aplicado a un problema financiero real, integrando:

-Análisis de datos

-Ingeniería de características

-Modelos avanzados

-Evaluación robusta

-Despliegue productivo mediante API

**El sistema está preparado para ser extendido a un entorno productivo o integrado en sistemas de evaluación crediticia.**


---------------------------------------
## Dataset Procesado Final
**NOTA TÉCNICA:** El dataset procesado final (`X_train_processed.parquet`) excede el límite estricto de 100MB de GitHub (incluso comprimido).

Para cumplir con la entrega, el archivo se encuentra disponible para descarga directa en el siguiente enlace seguro:

 **[DESCARGAR DATASET PROCESADO AQUÍ] (https://drive.google.com/file/d/1F6iXSgcQIhvR1W9A2xAjo8dHVWNqyv02/view?usp=sharing)**

*Alternativamente, este archivo puede ser regenerado localmente ejecutando el script:*
`python 02_data_preparation/02_data_preparation.py`

---------------------------------------






#End README.md













