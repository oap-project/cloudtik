import logging
import os
import sys
import threading
from logging.handlers import RotatingFileHandler

from typing import Callable

from cloudtik.core._private.core_utils import open_log

_default_handler = None


def setup_logger(logging_level, logging_format):
    """Setup default logging."""
    logger = logging.getLogger("cloudtik")
    if type(logging_level) is str:
        logging_level = logging.getLevelName(logging_level.upper())
    logger.setLevel(logging_level)
    global _default_handler
    if _default_handler is None:
        _default_handler = logging.StreamHandler()
        logger.addHandler(_default_handler)
    _default_handler.setFormatter(logging.Formatter(logging_format))
    # Setting this will avoid the message
    # is propagated to the parent logger.
    logger.propagate = False


def setup_component_logger(*,
                           logging_level,
                           logging_format,
                           log_dir,
                           filename,
                           max_bytes,
                           backup_count,
                           logger_name=""):
    """Configure the root logger that is used for CloudTik's python components.

    For example, it should be used for controller, and log monitor.
    The only exception is workers. They use the different logging config.

    Args:
        logging_level(str | int): Logging level in string or logging enum.
        logging_format(str): Logging format string.
        log_dir(str): Log directory path.
        filename(str): Name of the file to write logs.
        max_bytes(int): Same argument as RotatingFileHandler's maxBytes.
        backup_count(int): Same argument as RotatingFileHandler's backupCount.
        logger_name(str, optional): used to create or get the correspoding
            logger in getLogger call. It will get the root logger by default.
    Returns:
        logger (logging.Logger): the created or modified logger.
    """
    logger = logging.getLogger(logger_name)
    if type(logging_level) is str:
        logging_level = logging.getLevelName(logging_level.upper())
    assert filename, "filename argument should not be None."
    assert log_dir, "log_dir should not be None."
    handler = logging.handlers.RotatingFileHandler(
        os.path.join(log_dir, filename),
        maxBytes=max_bytes,
        backupCount=backup_count)
    handler.setLevel(logging_level)
    logger.setLevel(logging_level)
    handler.setFormatter(logging.Formatter(logging_format))
    logger.addHandler(handler)
    return logger


"""
All components underneath here is used specifically for the default_worker.py.
"""


class StandardStreamInterceptor:
    """Used to intercept stdout and stderr.

    Intercepted messages are handled by the given logger.

    NOTE: The logger passed to this method should always have
          logging.INFO severity level.

    Example:
        >>> from contextlib import redirect_stdout
        >>> logger = logging.getLogger("my_logger")
        >>> hook = StandardStreamHook(logger)
        >>> with redirect_stdout(hook):
        >>>     print("a") # stdout will be delegated to logger.

    Args:
        logger: Python logger that will receive messages streamed to
                the standard out/err and delegate writes.
        intercept_stdout(bool): True if the class intercepts stdout. False
                         if stderr is intercepted.
    """

    def __init__(self, logger, intercept_stdout=True):
        self.logger = logger
        assert len(self.logger.handlers) == 1, (
            "Only one handler is allowed for the interceptor logger.")
        self.intercept_stdout = intercept_stdout

    def write(self, message):
        """Redirect the original message to the logger."""
        self.logger.info(message)
        return len(message)

    def flush(self):
        for handler in self.logger.handlers:
            handler.flush()

    def isatty(self):
        # Return the standard out isatty. This is used by colorful.
        fd = 1 if self.intercept_stdout else 2
        return os.isatty(fd)

    def close(self):
        handler = self.logger.handlers[0]
        handler.close()

    def fileno(self):
        handler = self.logger.handlers[0]
        return handler.stream.fileno()


class StandardFdRedirectionRotatingFileHandler(RotatingFileHandler):
    """RotatingFileHandler that redirects stdout and stderr to the log file.

    It is specifically used to default_worker.py.

    The only difference from this handler vs original RotatingFileHandler is
    that it actually duplicates the OS level fd using os.dup2.
    """

    def __init__(self,
                 filename,
                 mode="a",
                 maxBytes=0,
                 backupCount=0,
                 encoding=None,
                 delay=False,
                 is_for_stdout=True):
        super().__init__(
            filename,
            mode=mode,
            maxBytes=maxBytes,
            backupCount=backupCount,
            encoding=encoding,
            delay=delay)
        self.is_for_stdout = is_for_stdout
        self.switch_os_fd()

    def doRollover(self):
        super().doRollover()
        self.switch_os_fd()

    def get_original_stream(self):
        if self.is_for_stdout:
            return sys.stdout
        else:
            return sys.stderr

    def switch_os_fd(self):
        # Old fd will automatically closed by dup2 when necessary.
        os.dup2(self.stream.fileno(), self.get_original_stream().fileno())


def configure_log_file(out_file, err_file):
    # If either of the file handles are None, there are no log files to
    # configure since we're redirecting all output to stdout and stderr.
    if out_file is None or err_file is None:
        return
    stdout_fileno = sys.stdout.fileno()
    stderr_fileno = sys.stderr.fileno()
    # C++ logging requires redirecting the stdout file descriptor. Note that
    # dup2 will automatically close the old file descriptor before overriding
    # it.
    os.dup2(out_file.fileno(), stdout_fileno)
    os.dup2(err_file.fileno(), stderr_fileno)
    # We also manually set sys.stdout and sys.stderr because that seems to
    # have an effect on the output buffering. Without doing this, stdout
    # and stderr are heavily buffered resulting in seemingly lost logging
    # statements. We never want to close the stdout file descriptor, dup2 will
    # close it when necessary and we don't want python's GC to close it.
    sys.stdout = open_log(
        stdout_fileno, unbuffered=True, closefd=False)
    sys.stderr = open_log(
        stderr_fileno, unbuffered=True, closefd=False)


class StandardStreamDispatcher:
    def __init__(self):
        self.handlers = []
        self._lock = threading.Lock()

    def add_handler(self, name: str, handler: Callable) -> None:
        with self._lock:
            self.handlers.append((name, handler))

    def remove_handler(self, name: str) -> None:
        with self._lock:
            new_handlers = [pair for pair in self.handlers if pair[0] != name]
            self.handlers = new_handlers

    def emit(self, data):
        with self._lock:
            for pair in self.handlers:
                _, handle = pair
                handle(data)


global_standard_stream_dispatcher = StandardStreamDispatcher()
