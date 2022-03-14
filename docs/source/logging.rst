Logging
=======

Logging in SGnlp
~~~~~~~~~~~~~~~~

When the SGnlp package is being imported for the first time, a logger named 'sgnlp' is created as the base logger for the SGnlp package.

By default, a logging.StreamHandler and logging.NullHandler are added to the logger. The StreamHandler is used to print messages to the console with a level of DEBUG and the NullHandler is used to suppress all messages in the event where all other handlers have been removed.


Logging for contributors
~~~~~~~~~~~~~~~~~~~~~~~~

For contributors to the SGnlp package, please first obtain a script specific logger via the example code shown below.

Since the script specific logger is created within the 'sgnlp' module, it will be automatically added as a child logger to the 'sgnlp' logger and in turn inherit all the handlers from the 'sgnlp' logger.

.. code:: python

    import logging
    logger = logging.getLogger(__name__)

Please use this logger for logging throughout the script.

.. code:: python

    logger.debug('This is a debug message')
    logger.info('This is an info message')
    logger.warning('This is a warning message')
    logger.error('This is an error message')
    logger.critical('This is a critical message')

    # Sample logged messages
    # 2022-01-01 00:00:00,000 - DEBUG - sgnlp.models.ufd.modeling - modeling.py - 10 - This is a debug message
    # 2022-01-01 00:00:00,000 - INFO - sgnlp.models.ufd.modeling - modeling.py - 10 - This is an info message
    # 2022-01-01 00:00:00,000 - WARNING - sgnlp.models.ufd.modeling - modeling.py - 10 - This is a warning message
    # 2022-01-01 00:00:00,000 - ERROR - sgnlp.models.ufd.modeling - modeling.py - 10 - This is an error message
    # 2022-01-01 00:00:00,000 - CRITICAL - sgnlp.models.ufd.modeling - modeling.py - 10 - This is a critical message

As shown in the sample output above, the default log format is `%(asctime)s - %(levelname)s - %(name)s - %(filename)s - %(lineno)d - %(message)s`.


Logging for 3rd party package
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When using 'sgnlp' as a 3rd party package, the user can choose to add additional handlers to the 'sgnlp' logger or remove unwanted handlers including the default `StreamHandler` and `NullHandler`.

Query name of all active handlers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To query all active handlers assigned to the 'sgnlp' logger, use the following code.

Calling the `get_active_handlers_name` function will return a list of handler's name which can be use for removing the handlers.

.. code:: python

    from sgnlp.utils.logger import get_active_handlers_name

    get_active_handlers_name()

    # ['streamHandler', 'nullHandler']


Add Handlers
~~~~~~~~~~~~

To add a handler to the 'sgnlp' logger, the user can use the following code.

Currently, StreamHandler, FileHandler, RotatingFileHandler and TimedRotatingFileHandler are supported.

Please note that all fields for the config are optional, for handlers with output files, default path is 'logs/sgnlp.log' if "filename" field is not specfied.

For detailed explanation of each handlers config, please refer to the official Python 3 documentation (https://docs.python.org/3/library/logging.handlers.html).

.. code:: python

    from sgnlp.utils.logger import add_handler

    # Add a streamhandler to the 'sgnlp' logger
    stream_config = {
        "name": "streamhandler",
        "level": "DEBUG",
        "formatter": "%(asctime)s - %(message)s",
    }

    # Note use "streamHandler" key to add a StreamHandler
    add_handler("streamHandler", stream_config)")

    # Add a filehandler to the 'sgnlp' logger
    file_config = {
        "name": "filehandler",
        "level": "DEBUG",
        "formatter": "%(asctime)s - %(message)s",
        "filename": "logs/sgnlp.log",
        "mode": "a",
    }

    # Note use "fileHandler" key to add a FileHandler
    add_handler("fileHandler", file_config)")

    # Add a rotating filehandler to the 'sgnlp' logger
    rotating_file_config = {
        "name": "rotatingfilehandler",
        "level": "DEBUG",
        "formatter": "%(asctime)s - %(message)s",
        "filename": "logs/sgnlp.log",
        "mode": "a",
        "maxBytes": 10485760,
        "backupCount": 5,
    }

    # Note use "rotatingFileHandler" key to add a RotatingFileHandler
    add_handler("rotatingFileHandler", rotating_file_config)")

    # Add a timed rotating filehandler to the 'sgnlp' logger
    timed_rotating_file_config = {
        "name": "timedrotatingfilehandler",
        "level": "DEBUG",
        "formatter": "%(asctime)s - %(message)s",
        "filename": "logs/sgnlp.log",
        "backupCount": 5,
        "when": "midnight",
        "interval": 1,
        "utc": True,
    }

    # Note use "timedRotatingFileHandler" key to add a TimedRotatingFileHandler
    add_handler("timedRotatingFileHandler", timed_rotating_file_config)")


Remove Handlers
~~~~~~~~~~~~~~

To remove an active handler from the 'sgnlp' package, first query the name of all active handlers using the `get_active_handlers_name` function.

Next use the `remove_handler` function to remove the handler.

.. code:: python

    from sgnlp.utils.logger import remove_handler

    # Remove the 'streamHandler' handler
    remove_handler("streamHandler")

    # Remove the 'fileHandler' handler
    remove_handler("fileHandler")

    # Remove the 'rotatingFileHandler' handler
    remove_handler("rotatingFileHandler")

    # Remove the 'timedRotatingFileHandler' handler
    remove_handler("timedRotatingFileHandler")

It is advisable not to remove the default NullHandler when there are no other active handlers present in the 'sgnlp' logger.
This is to ensure that no log message from the 'sgnlp' package will be sent when all other handlers have been removed.
