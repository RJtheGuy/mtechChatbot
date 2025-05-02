import logging
import sys
from pathlib import Path

def setup_logging():
    """Configure logging for the application"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(Path(__file__).parent.parent / 'logs' / 'chatbot.log')
        ]
    )

# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)