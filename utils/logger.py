# utils/logger.py
import logging
import sys
from datetime import datetime
from utils import config

# Formato del log
LOG_FORMAT = logging.Formatter(
    '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
LOG_FILE = config.LOGS_DIR / f"pipeline_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"

def get_logger(name):
    """
    Configura y devuelve un logger con handlers para consola y archivo.
    Args:
        name (str): Nombre del logger.
    Returns:
        logging.Logger: Logger configurado con handlers para consola y archivo.
    Raises:
        FileNotFoundError: Si el directorio de logs no existe.
    """
    # Validar existencia del directorio de logs
    if not config.LOGS_DIR.exists():
        raise FileNotFoundError(f"El directorio de logs {config.LOGS_DIR} no existe.")

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Prevenir duplicación de logs si ya está configurado
    if not logger.handlers:
        # Handler para la consola
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(LOG_FORMAT)
        logger.addHandler(console_handler)

        # Handler para el archivo de log
        file_handler = logging.FileHandler(LOG_FILE)
        file_handler.setFormatter(LOG_FORMAT)
        logger.addHandler(file_handler)
        
    return logger

# Instancia global para ser usada en otros módulos
log = get_logger(__name__)