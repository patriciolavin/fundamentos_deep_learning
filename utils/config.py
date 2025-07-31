# utils/config.py
from pathlib import Path

# --- Rutas Principales ---
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
BAD_DATA_DIR = DATA_DIR / 'bad_data'
MODELS_DIR = ROOT_DIR / 'models'
PLOTS_DIR = ROOT_DIR / 'plots'
REPORTS_DIR = ROOT_DIR / 'reports'
LOGS_DIR = ROOT_DIR / 'logs'
TEMPLATES_DIR = ROOT_DIR / 'templates'

# Crear directorios si no existen y validar
for path in [PROCESSED_DATA_DIR, BAD_DATA_DIR, MODELS_DIR, PLOTS_DIR, REPORTS_DIR, LOGS_DIR, TEMPLATES_DIR]:
    path.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        raise FileNotFoundError(f"No se pudo crear el directorio: {path}")

# --- Archivos ---
DATASET_NAME = 'dataset_natalidad.csv'
DATASET_PATH = RAW_DATA_DIR / DATASET_NAME

# Validar existencia del dataset
if not DATASET_PATH.exists():
    raise FileNotFoundError(f"El archivo de datos {DATASET_PATH} no existe.")

FINAL_MODEL_NAME = 'ridge_model_final.joblib'
REPORT_NAME = 'reporte_final_de_analisis.html'

# --- Parámetros de Modelado ---
TARGET_VARIABLE = 'Tasa_Natalidad'
TEST_SIZE = 0.2
RANDOM_STATE = 42
K_FOLDS = 10

# --- Nombres de Columnas ---
# Lista de columnas de características para validación y modelado
FEATURE_COLUMNS = [
    'PIB_per_capita', 'Acceso_Salud', 'Nivel_Educativo',
    'Tasa_Empleo_Femenino', 'Edad_Maternidad', 'Urbanizacion'
]