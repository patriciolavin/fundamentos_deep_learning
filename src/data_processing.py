# src/data_processing.py
import pandas as pd
import numpy as np  # CORRECCIÓN: Importar numpy para validaciones
from utils.logger import log
from utils import config
from src.preprocess import preprocess_data

def load_data():
    """
    Carga los datos desde el archivo CSV definido en la configuración.
    Returns:
        pd.DataFrame: DataFrame con los datos cargados.
    Raises:
        FileNotFoundError: Si el archivo no se encuentra.
    """
    log.info(f"Iniciando la carga de datos desde: {config.DATASET_PATH}")
    try:
        df = pd.read_csv(config.DATASET_PATH)
        log.info(f"Datos cargados exitosamente. Dimensiones: {df.shape}")
        return df
    except FileNotFoundError:
        log.error(f"Error: El archivo {config.DATASET_PATH} no fue encontrado.")
        raise

def clean_data(df):
    """
    Realiza la limpieza de datos, incluyendo nulos y duplicados.
    Args:
        df (pd.DataFrame): DataFrame con los datos crudos.
    Returns:
        pd.DataFrame: DataFrame limpio.
    """
    log.info("Iniciando proceso de limpieza de datos.")
    
    # Validar columnas esperadas
    required_columns = config.FEATURE_COLUMNS + [config.TARGET_VARIABLE]
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        log.error(f"Columnas faltantes en el DataFrame: {missing_cols}")
        raise ValueError(f"Columnas faltantes: {missing_cols}")
    
    # CORRECCIÓN: Validar que las columnas esperadas sean numéricas
    non_numeric_cols = [
        col for col in required_columns 
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col])
    ]
    if non_numeric_cols:
        log.error(f"Columnas no numéricas detectadas: {non_numeric_cols}")
        raise ValueError(f"Columnas no numéricas: {non_numeric_cols}")
    
    # Loguear tipos de datos de las columnas
    log.info("Tipos de datos de las columnas:")
    for col in df.columns:
        log.info(f"Columna: {col}, Tipo: {df[col].dtype}")
    
    # Aplicar preprocesamiento
    df = preprocess_data(df)
    
    # Manejo de duplicados
    initial_rows = len(df)
    df.drop_duplicates(inplace=True)
    rows_after_duplicates = len(df)
    if initial_rows > rows_after_duplicates:
        log.warning(f"Se eliminaron {initial_rows - rows_after_duplicates} filas duplicadas.")
    else:
        log.info("No se encontraron filas duplicadas.")
    
    log.info("Proceso de limpieza de datos finalizado.")
    return df