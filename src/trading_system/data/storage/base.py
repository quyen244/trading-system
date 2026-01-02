import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from trading_system.utils.logger import setup_logger
from dotenv import load_dotenv
import numpy as np 

load_dotenv()

logger = setup_logger("StorageEngine")

class StorageEngine:
    def __init__(self, db_url=None):
        if db_url is None:
            self.db_url = os.getenv("TRADING_DB_URL", "")
            logger.info(f"Using default DB URL: {self.db_url}")
            
        self.engine = create_engine(self.db_url)
        self.Session = sessionmaker(bind=self.engine)

        logger.info("Database connection established.")
    def store_data(self):
        pass
    def get_data(self):
        pass
        