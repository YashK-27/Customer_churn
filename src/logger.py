import logging
import os
from datetime import datetime
from src.exception import CustomException

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
log_path = os.path.join(os.getcwd(), "logs", LOG_FILE)
os.makedirs(log_path,exist_ok=True)

LOG_FILE_PATH = os.path.join(log_path, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,  # message with severity level INFO or higher will be logged
)



# def main():  
#     logging.info("This is an example log message.")

# if __name__ == '__main__':
#     main()