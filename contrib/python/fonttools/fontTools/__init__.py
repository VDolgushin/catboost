import logging
from fontTools.misc.loggingTools import configLogger

log = logging.getLogger(__name__)

version = __version__ = "4.48.1"

__all__ = ["version", "log", "configLogger"]
