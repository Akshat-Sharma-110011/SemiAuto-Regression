import logging
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime
import sys
import colorama
from colorama import Fore, Back, Style

# Initialize colorama (required for Windows)
colorama.init(autoreset=True)

# Constants for log configuration
LOG_DIR = 'logs'
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y-%H_%M_%S')}.log"
MAX_LOG_SIZE = 5 * 1024 * 1024  # 5 MB
BACKUP_COUNT = 3  # Number of backup log files to keep

# Construct log file path
root_dir = os.path.dirname(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
log_dir_path = os.path.join(root_dir, LOG_DIR)
os.makedirs(log_dir_path, exist_ok=True)
log_file_path = os.path.join(log_dir_path, LOG_FILE)


class PrettyFormatter(logging.Formatter):
    """Custom formatter for pretty colored logs with symbols"""

    # Define log format with colors and symbols
    FORMATS = {
        logging.DEBUG: f"{Fore.CYAN}[•] %(asctime)s {Fore.WHITE}| {Fore.CYAN}%(name)-15s {Fore.WHITE}| {Fore.CYAN}DEBUG{Style.RESET_ALL} | %(message)s",
        logging.INFO: f"{Fore.GREEN}[✓] %(asctime)s {Fore.WHITE}| {Fore.GREEN}%(name)-15s {Fore.WHITE}| {Fore.GREEN}INFO{Style.RESET_ALL}  | %(message)s",
        logging.WARNING: f"{Fore.YELLOW}[⚠] %(asctime)s {Fore.WHITE}| {Fore.YELLOW}%(name)-15s {Fore.WHITE}| {Fore.YELLOW}WARN{Style.RESET_ALL}  | %(message)s",
        logging.ERROR: f"{Fore.RED}[✗] %(asctime)s {Fore.WHITE}| {Fore.RED}%(name)-15s {Fore.WHITE}| {Fore.RED}ERROR{Style.RESET_ALL} | %(message)s",
        logging.CRITICAL: f"{Fore.MAGENTA}[!] %(asctime)s {Fore.WHITE}| {Fore.MAGENTA}%(name)-15s {Fore.WHITE}| {Fore.MAGENTA}CRIT{Style.RESET_ALL}  | %(message)s"
    }

    # Plain format for file logging (no colors)
    FILE_FORMATS = {
        logging.DEBUG: "[•] %(asctime)s | %(name)-15s | DEBUG | %(message)s",
        logging.INFO: "[✓] %(asctime)s | %(name)-15s | INFO  | %(message)s",
        logging.WARNING: "[⚠] %(asctime)s | %(name)-15s | WARN  | %(message)s",
        logging.ERROR: "[✗] %(asctime)s | %(name)-15s | ERROR | %(message)s",
        logging.CRITICAL: "[!] %(asctime)s | %(name)-15s | CRIT  | %(message)s"
    }

    def __init__(self, use_color=True):
        super().__init__(datefmt="%Y-%m-%d %H:%M:%S")
        self.use_color = use_color

    def format(self, record):
        # Select colored or plain format based on the destination
        log_format = self.FORMATS.get(record.levelno) if self.use_color else self.FILE_FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)


class SectionLogger:
    """Helper class to create section headers in logs"""

    @staticmethod
    def section(title, logger=None, level=logging.INFO, char='=', length=50):
        """Log a section header with decorative characters"""
        if logger is None:
            logger = logging.getLogger()

        # Create a decorative header
        header = f"\n{char * length}\n{title.center(length)}\n{char * length}"

        if level == logging.INFO:
            logger.info(f"{Fore.BLUE}{header}{Style.RESET_ALL}")
        elif level == logging.DEBUG:
            logger.debug(f"{Fore.CYAN}{header}{Style.RESET_ALL}")
        elif level == logging.WARNING:
            logger.warning(f"{Fore.YELLOW}{header}{Style.RESET_ALL}")
        elif level == logging.ERROR:
            logger.error(f"{Fore.RED}{header}{Style.RESET_ALL}")
        elif level == logging.CRITICAL:
            logger.critical(f"{Fore.MAGENTA}{header}{Style.RESET_ALL}")


# Flag to track if the logger has been configured already
_logger_configured = False


def configure_logger():
    """
    Configures logging with a rotating file handler and a console handler with pretty formatting.
    Only configures once to prevent duplicate handlers.
    """
    global _logger_configured

    # Skip if already configured
    if _logger_configured:
        return

    # Create a custom logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Clear any existing handlers
    if logger.handlers:
        for handler in logger.handlers:
            logger.removeHandler(handler)

    # Console handler with colored formatter
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(PrettyFormatter(use_color=True))
    console_handler.setLevel(logging.INFO)

    # File handler with plain formatter and rotation
    file_handler = RotatingFileHandler(log_file_path, maxBytes=MAX_LOG_SIZE, backupCount=BACKUP_COUNT, encoding="utf-8")
    file_handler.setFormatter(PrettyFormatter(use_color=False))
    file_handler.setLevel(logging.INFO)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Log startup information
    logger.info(f"Logger initialized. Log file: {log_file_path}")

    # Set the flag to indicate logger is configured
    _logger_configured = True


# Configure the logger
configure_logger()

# Export the section logger for use in other modules
section = SectionLogger.section