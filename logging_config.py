"""
Logging Configuration for HVAC GraphRAG System
Uses loguru for structured, colorized logging
"""
import sys
from loguru import logger
from config import Config

# Remove default handler
logger.remove()

# Add console handler with colorization
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{line}</cyan> | <level>{message}</level>",
    level=Config.LOG_LEVEL,
    colorize=True
)

# Add file handler with rotation
logger.add(
    Config.LOG_FILE,
    rotation="10 MB",  # Rotate when file reaches 10MB
    retention="30 days",  # Keep logs for 30 days
    compression="zip",  # Compress rotated logs
    level=Config.LOG_LEVEL,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} | {message}",
    enqueue=True  # Thread-safe logging
)

# Export configured logger
__all__ = ['logger']
