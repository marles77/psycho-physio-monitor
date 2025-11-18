# ==========================================================
# Main module
# ==========================================================

import os
import datetime
import logging
from logger import logger
import app
import multiprocessing as mp

def setup_logging() -> None:

    log_dir = os.path.join('log')
    os.makedirs(log_dir, exist_ok=True)

    now = datetime.datetime.now()
    date_str = now.strftime("%Y-%m-%d_%H-%M-%S")
    log_fname = os.path.join(log_dir, f'log_{date_str}.log')

    # Configure logging ONCE
    if not logger.handlers:
        file_handler = logging.FileHandler(log_fname, mode='w')
        formatter = logging.Formatter('%(asctime)s %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Optional: log to console (can be commented out)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        logger.info("Logging to: %s", log_fname)

if __name__ == "__main__":
    
    mp.set_start_method("spawn")
    setup_logging()
    logger.info("Main process started")
    app = app.App()
    app.mainloop()
    
