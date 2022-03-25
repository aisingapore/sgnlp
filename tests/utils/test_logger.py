import logging
import pathlib
import shutil
import tempfile
import unittest
from logging import StreamHandler, FileHandler, NullHandler
from logging.handlers import TimedRotatingFileHandler, RotatingFileHandler

from sgnlp.utils.logger import setup_logging, add_handler, remove_handler, get_active_handlers_name


class TestLoggerTestCase(unittest.TestCase):
    def setUp(self) -> None:
        root_logger = logging.getLogger()
        root_logger.manager.loggerDict.clear()  # Clear all loggers
        with tempfile.TemporaryDirectory() as tmp_dir:
            self.tmp_dir = pathlib.Path(tmp_dir)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_setup_logging(self):
        setup_logging()
        logger = logging.getLogger("sgnlp")
        self.assertEqual(len(logger.handlers), 2)

        handlers = [handler.name for handler in logger.handlers]
        self.assertIn("streamHandler", handlers)
        self.assertIn("nullHandler", handlers)
        print(logger.handlers[0].name)
        self.assertTrue(
            isinstance([handler for handler in logger.handlers if handler.name == "streamHandler"][0], StreamHandler)
        )
        self.assertTrue(
            isinstance([handler for handler in logger.handlers if handler.name == "nullHandler"][0], NullHandler)
        )

    def test_add_file_handler(self):
        add_handler("fileHandler", {"filename": self.tmp_dir / "test.log"})
        logger = logging.getLogger("sgnlp")
        self.assertEqual(len(logger.handlers), 1)

        handlers = [handler.name for handler in logger.handlers]
        self.assertIn("fileHandler", handlers)
        self.assertTrue(
            isinstance([handler for handler in logger.handlers if handler.name == "fileHandler"][0], FileHandler)
        )

    def test_add_rotating_file_handler(self):
        add_handler("rotatingFileHandler", {"filename": self.tmp_dir / "test.log"})
        logger = logging.getLogger("sgnlp")
        self.assertEqual(len(logger.handlers), 1)

        handlers = [handler.name for handler in logger.handlers]
        self.assertIn("rotatingFileHandler", handlers)
        self.assertTrue(
            isinstance(
                [handler for handler in logger.handlers if handler.name == "rotatingFileHandler"][0],
                RotatingFileHandler,
            )
        )

    def test_add_timed_rotating_file_handler(self):
        add_handler("timedRotatingFileHandler", {"filename": self.tmp_dir / "test.log"})
        logger = logging.getLogger("sgnlp")
        self.assertEqual(len(logger.handlers), 1)

        handlers = [handler.name for handler in logger.handlers]
        self.assertIn("timedRotatingFileHandler", handlers)
        self.assertTrue(
            isinstance(
                [handler for handler in logger.handlers if handler.name == "timedRotatingFileHandler"][0],
                TimedRotatingFileHandler,
            )
        )

    def test_remove_handler(self):
        add_handler("streamHandler", {})
        logger = logging.getLogger("sgnlp")
        self.assertEqual(len(logger.handlers), 1)
        remove_handler("streamHandler")
        self.assertEqual(len(logger.handlers), 0)

    def test_get_active_handlers_name(self):
        setup_logging()
        self.assertEqual(get_active_handlers_name(), ["streamHandler", "nullHandler"])
