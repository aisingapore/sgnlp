import json
import logging
import pathlib
from logging import Formatter, StreamHandler, FileHandler
from logging.config import dictConfig
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from typing import List, Union


LOG_CONFIG = pathlib.Path(__file__).parent / "config" / "logger_default_config.json"


def setup_logging(log_config_path: Union[str, pathlib.Path] = LOG_CONFIG):
    """
    Function to setup initial logging configuration.

    Args:
        log_config_path (Union[str, pathlib.Path], optional): file path to logging config file. Defaults to LOG_CONFIG.
    """
    logger = logging.getLogger("sgnlp")  # Root logger for SGnlp package
    handlers = [handler.name for handler in logger.handlers]
    if "streamHandler" in handlers and "nullHandler" in handlers:
        logger.debug("Logger already configured.")
        return
    if isinstance(log_config_path, str):
        log_config_path = pathlib.Path(log_config_path)
    if log_config_path.exists():
        with open(log_config_path) as f:
            config = json.load(f)
            dictConfig(config)
    else:
        raise ValueError(f"Logging config file {str(log_config_path)} does not exist.")


def _create_stream_handler(handler_config: dict):
    """
    Function to create stream handler.

    Args:
        handler_config (dict): handler config

    Returns:
        logging.StreamHandler: stream handler
    """
    log_handler = StreamHandler()
    log_handler.name = handler_config.get("name", "streamHandler")
    log_handler.setLevel(handler_config.get("level", "DEBUG"))
    log_handler.setFormatter(
        Formatter(
            handler_config.get(
                "formatter", "%(asctime)s - %(levelname)s - %(name)s - %(filename)s - %(lineno)s - %(message)s"
            )
        )
    )
    log_handler.setStream(eval(handler_config.get("args", "()"), vars(logging)))
    return log_handler


def _create_file_handler(handler_config: dict):
    """
    Function to create file handler.

    Args:
        handler_config (dict): _description_
    """
    log_file_path = pathlib.Path(handler_config.get("filename", "logs/sgnlp.log"))
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    log_handler = FileHandler(filename=log_file_path, mode=handler_config.get("mode", "a"))
    log_handler.name = handler_config.get("name", "fileHandler")
    log_handler.setLevel(handler_config.get("level", "DEBUG"))
    log_handler.setFormatter(
        Formatter(
            handler_config.get(
                "formatter", "%(asctime)s - %(levelname)s - %(name)s - %(filename)s - %(lineno)s - %(message)s"
            )
        )
    )
    log_handler.mode = handler_config.get("mode", "a")
    return log_handler


def _create_rotating_file_handler(handler_config: dict):
    """
    Function to create rotating file handler.

    Args:
        handler_config (dict): handler config

    Returns:
        logging.RotatingFileHandler: rotating file handler
    """
    log_file_path = pathlib.Path(handler_config.get("filename", "logs/sgnlp.log"))
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    log_handler = RotatingFileHandler(
        filename=log_file_path,
        mode=handler_config.get("mode", "a"),
        maxBytes=int(handler_config.get("maxBytes", "10485760")),
        backupCount=int(handler_config.get("backupCount", "1")),
    )
    log_handler.name = handler_config.get("name", "rotatingFileHandler")
    log_handler.setLevel(handler_config.get("level", "DEBUG"))
    log_handler.setFormatter(
        Formatter(
            handler_config.get(
                "formatter", "%(asctime)s - %(levelname)s - %(name)s - %(filename)s - %(lineno)s - %(message)s"
            )
        )
    )
    return log_handler


def _create_timed_rotating_file_handler(handler_config: dict):
    """
    Function to create timed rotating file handler.

    Args:
        handler_config (dict): handler config

    Returns:
        logging.TimedRotatingFileHandler: timed rotating file handler
    """
    log_file_path = pathlib.Path(handler_config.get("filename", "logs/sgnlp.log"))
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    log_handler = TimedRotatingFileHandler(
        filename=log_file_path,
        when=handler_config.get("when", "midnight"),
        interval=int(handler_config.get("interval", "1")),
        backupCount=int(handler_config.get("backupCount", "1")),
    )
    log_handler.name = handler_config.get("name", "timedRotatingFileHandler")
    log_handler.setLevel(handler_config.get("level", "DEBUG"))
    log_handler.setFormatter(
        Formatter(
            handler_config.get(
                "formatter", "%(asctime)s - %(levelname)s - %(name)s - %(filename)s - %(lineno)s - %(message)s"
            )
        )
    )
    return log_handler


_create_handler = {
    "streamHandler": _create_stream_handler,
    "fileHandler": _create_file_handler,
    "rotatingFileHandler": _create_rotating_file_handler,
    "timedRotatingFileHandler": _create_timed_rotating_file_handler,
}


def add_handler(handler_type: str, handler_config: Union[dict, None] = None) -> None:
    """
    Function to add handler to logger.
    Supported handler keys: [
        "streamHandler", "fileHandler", "rotatingFileHandler", "timedRotatingFileHandler"]

    Args:
        handler_type (str)): type of handle to add
        handler_config (Union[dict, None]): handler config. Optional.
    """
    if handler_config is None:
        handler_config = {}
    logger = logging.getLogger("sgnlp")  # Root logger for SGnlp package
    if handler_type in _create_handler:
        handler = _create_handler[handler_type](handler_config)
        logger.addHandler(handler)
    else:
        raise ValueError(f"Handler type {handler_type} is not supported.")


def remove_handler(handler_name: str) -> None:
    """
    Function to remove handler from logger.

    Args:
        handler_name (str): name of handler to remove
    """
    logger = logging.getLogger("sgnlp")  # Root logger for SGnlp package
    for handler in logger.handlers:
        if handler.name == handler_name:
            logger.removeHandler(handler)
            break


def get_active_handlers_name() -> List[Union[str, None]]:
    """
    Function to get active handlers name for the 'sgnlp' loggers.

    Returns:
        List[str]: active handlers name
    """
    logger = logging.getLogger("sgnlp")  # Root logger for SGnlp package
    return [handler.name for handler in logger.handlers]
