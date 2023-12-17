from pathlib import Path
import logging


# Setting root directory path for easier refernce in script
BASE_DIR = Path(__file__).resolve().parent.parent

# Path to various directories
MODEL_DIR = BASE_DIR / "models"
TRAIN_DIR = BASE_DIR / "dataset/train_set"
TEST_DIR = BASE_DIR / "dataset/test_set"
TEST_IMG_PATH = BASE_DIR / "img"
RESULT_DIR = BASE_DIR / "result"
LOG_DIR = BASE_DIR / "log"


# Configure logging
logging.basicConfig(level=logging.INFO)

# Create a logger and set the level to INFO
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create a FileHandler that writes log messages to a file
log_file_path = LOG_DIR / "logfile.log"  # Replace with the desired path
file_handler = logging.FileHandler(log_file_path)

# Create a Formatter to specify the log message format
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)

# Add the FileHandler to the logger
logger.addHandler(file_handler)
