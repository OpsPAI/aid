import os
import logging
from logging.handlers import RotatingFileHandler


def setupLogging(logDir=None, name=None, logFmt="lite"):
    """
    Set up logging

    :logDir: the dir to store log file, if not specified then it will not log to file
    :name: name of logger, you need to call getLogger when the name is set, used for surpressing unrelated logs
    :logFmt: can be lite or full
    """
    assert logFmt in ["lite", "full"], "Undefined log format"
    assert name is not None or logFmt == "lite", "You need to specify logger name in full log format"

    # Predefiend log format
    # https://www.programcreek.com/python/example/192/logging.Formatter
    log_file_format = {
        "full": "%(asctime)s - %(levelname)s - %(name)s - %(filename)s:%(funcName)s:%(lineno)d - %(message)s",
        "lite": "%(asctime)s - %(levelname)s - %(filename)s - %(message)s"
    }

    log_console_format = {
        "full": "[%(levelname)s] %(name)s - %(filename)s:%(funcName)s:%(lineno)d - %(message)s",
        "lite": "[%(levelname)s] %(filename)s - %(message)s"
    }

    # Main logger
    mainLogger = logging.getLogger(name)
    mainLogger.setLevel(logging.DEBUG)

    # Console logger (must)
    consoleHandler = logging.StreamHandler()
    consoleHandler.setLevel(logging.INFO)
    consoleHandler.setFormatter(logging.Formatter(log_console_format[logFmt]))
    mainLogger.addHandler(consoleHandler)

    # File logger (optional)
    if logDir:
        if not os.path.isdir(logDir):
            os.mkdir(logDir)

        debugFileHandler = RotatingFileHandler(
            os.path.join(logDir, f'{name+"_" if name else ""}debug.log'), maxBytes=10**6, backupCount=5)
        debugFileHandler.setLevel(logging.DEBUG)
        debugFileHandler.setFormatter(
            logging.Formatter(log_file_format[logFmt]))
        mainLogger.addHandler(debugFileHandler)

        errorFileHandler = RotatingFileHandler(
            os.path.join(logDir, f'{name+"_" if name else ""}warning.log'), maxBytes=10**6, backupCount=5)
        errorFileHandler.setLevel(logging.WARNING)
        errorFileHandler.setFormatter(
            logging.Formatter(log_file_format[logFmt]))
        mainLogger.addHandler(errorFileHandler)

    return mainLogger


if __name__ == '__main__':
    setupLogging("logs")
    logging.info("information")
    logging.debug("oops bug")
    logging.warning("warning")

    setupLogging("logs", "somename", "full")
    logger = logging.getLogger("somename")
    logger.info("information")
    logger.debug("oops bug")
    logger.warning("warning")
