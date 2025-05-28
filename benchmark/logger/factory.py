import logging


def create_null_logger(name: str = "null_logger") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.CRITICAL)
    null_handler = logging.NullHandler()
    logger.addHandler(null_handler)
    return logger


def create_default_console_logger(logger_name: str = 'log') -> logging.Logger:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    return logger
