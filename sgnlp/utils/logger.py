import configparser
import pathlib
import logging
from logging import Formatter, StreamHandler, FileHandler
from logging.config import fileConfig
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from typing import Union


LOG_CONFIG = pathlib.Path(__file__).parent / "config" / "logger_config.conf"
HANDLER_DEFAULT_CONFIG = pathlib.Path(__file__).parent / "config" / "log_handlers_default_config.conf"


def setup_logging(log_config_path: Union[str, pathlib.Path] = LOG_CONFIG):
    """
    Function to setup initial logging configuration.

    Args:
        log_config_path (Union[str, pathlib.Path], optional): file path to logging config file. Defaults to LOG_CONFIG.
    """
    if isinstance(log_config_path, str):
        log_config_path = pathlib.Path(log_config_path)
    if log_config_path.exists():
        fileConfig(log_config_path)
    else:
        raise ValueError(f"Logging config file {str(log_config_path)} does not exist.")


def _read_handler_config(handler_config_path: Union[str, pathlib.Path] = HANDLER_DEFAULT_CONFIG):
    """
    Function to read handler config file.

    Args:
        handler_config_path (Union[str, pathlib.Path], optional): file path to handler config file. Defaults to HANDLER_DEFAULT_CONFIG.

    Returns:
        dict: handler config
    """
    if isinstance(handler_config_path, str):
        handler_config_path = pathlib.Path(handler_config_path)
    if handler_config_path.exists():
        handler_config = configparser.ConfigParser()
        handler_config.read(handler_config_path)
        return handler_config
    else:
        raise ValueError(f"Handler config file {str(handler_config_path)} does not exist.")


def _create_stream_handler(handler_config: Union[dict, configparser.SectionProxy]):
    """
    Function to create stream handler.

    Args:
        handler_config (Union[dict, configparser.SectionProxy]): handler config

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


def _create_file_handler(handler_config: Union[dict, configparser.SectionProxy]):
    """
    Function to create file handler.

    Args:
        handler_config (Union[dict, configparser.SectionProxy]): _description_
    """
    log_handler = FileHandler(
        filename=handler_config.get("filename", "logs/sgnlp.log"), mode=handler_config.get("mode", "a")
    )
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


def _create_rotating_file_handler(handler_config: Union[dict, configparser.SectionProxy]):
    """
    Function to create rotating file handler.

    Args:
        handler_config (Union[dict, configparser.SectionProxy]): handler config

    Returns:
        logging.RotatingFileHandler: rotating file handler
    """
    log_handler = RotatingFileHandler(
        filename=handler_config.get("filename", "logs/sgnlp.log"),
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


def _create_timed_rotating_file_handler(handler_config: Union[dict, configparser.SectionProxy]):
    """
    Function to create timed rotating file handler.

    Args:
        handler_config (Union[dict, configparser.SectionProxy]): handler config

    Returns:
        logging.TimedRotatingFileHandler: timed rotating file handler
    """
    log_handler = TimedRotatingFileHandler(
        filename=handler_config.get("filename", "logs/sgnlp.log"),
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


def _add_handler(logger: logging.Logger, handler_config: Union[dict, configparser.SectionProxy]):
    """
    Function to add handler to logger.

    Args:
        logger (logging.Logger): logger
        handler_config (Union[dict, configparser.SectionProxy]): handler config
    """
    handler_type = handler_config.get("type", "streamHandler")
    if handler_type in _create_handler:
        handler = _create_handler[handler_type](handler_config)
        logger.addHandler(handler)
    else:
        raise ValueError(f"Handler type {handler_type} is not supported.")
