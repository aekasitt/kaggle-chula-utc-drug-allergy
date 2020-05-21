#!/usr/bin/python
# coding:utf-8
# Copyright (C) 2019-2020 All rights reserved.
# FILENAME:  logger.py
# VERSION: 	 1.0
# CREATED: 	 2020-04-29 00:09
# AUTHOR: 	 Aekasitt Guruvanich <aekazitt@gmail.com>
# DESCRIPTION:
#
# HISTORY:
#*************************************************************
"""Module defining class Logger"""
import sys
import threading
import re
from logging import FileHandler, Filter, Formatter, \
      getLogger, INFO, StreamHandler
from logging.handlers import TimedRotatingFileHandler
### Local Modules ###
from helpers.singleton import Singleton

# Define Color Escape Codes for Logging
BLUE='\033[1;94m'     # Sky Blue ANSI Escape Code
CYAN='\033[1;36m'     # Light Cyan ANSI Escape Code
GREEN='\033[1;92m'    # Green ANSI Escape Code
NC='\033[1;0m'        # No Colour ANSI Escape Code
RED='\033[1;31m'      # Red ANSI Escape Code
YELLOW='\033[1;93m'   # Yellow ANSI Escape Code
ColorEnum = {
  'blue': BLUE,
  'cyan': CYAN,
  'green': GREEN,
  'red': RED,
  'yellow': YELLOW
}
class CleanAnsiFilter(Filter):
  # 7-bit C1 ANSI sequences
  ANSI_ESCAPE = re.compile(r'''
      \x1B  # ESC
      (?:   # 7-bit C1 Fe (except CSI)
          [@-Z\\-_]
      |     # or [ for CSI, followed by a control sequence
          \[
          [0-?]*  # Parameter bytes
          [ -/]*  # Intermediate bytes
          [@-~]   # Final byte
      )
  ''', re.VERBOSE)

  def clean_ansi(self, message:str) -> str:
    if message is None: return ''
    return CleanAnsiFilter.ANSI_ESCAPE.sub('', str(message))

  def filter(self, record):
    record.clean_message = self.clean_ansi(record.msg)
    return True

class Logger(Singleton):
  """Class used Log to sys.stdout"""

  def __init__(self, log_name, is_cut=True):
    self.logger = getLogger(log_name)
    self.logger.addFilter(CleanAnsiFilter())

    ### Logs to File in /logs folder ###
    handler = None
    if is_cut:
      handler = TimedRotatingFileHandler(
          filename='logs/{}.log'.format(log_name),
          when='D', interval=1, backupCount=10)
          
    else:
      handler = FileHandler('logs/{}.log'.format(log_name))
    fmt = Formatter('%(asctime)s - [%(levelname)s] %(clean_message)s', datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(fmt)
    self.logger.addHandler(handler)

    ### Logs to Stream ###
    stream_handler = StreamHandler(sys.stdout)
    stream_fmt = Formatter(f'{CYAN}[%(levelname)s]{NC} %(message)s')
    stream_handler.setFormatter(stream_fmt)
    self.logger.addHandler(stream_handler)
    self.logger.setLevel(INFO)

  def shutdown(self):
    """Release all of Logger instance handler and then release Logger."""
    try:
      for handler in list(self.logger.handlers):
        self.logger.removeHandler(handler)
        del handler
      if self.logger is not None:
        del self.logger
    except AttributeError:
      pass

  def info(self, msg, *args, **kwargs):
    """
    Log a message with severity 'INFO' on the root logger. If the logger has
    no handlers, call basicConfig() to add a console handler with a pre-defined
    format.
    """
    if len(args) > 0:
      args_dict = dict(args[0])
      color = args_dict.get('color', None)
      if color is not None:
        color = ColorEnum.get(color, NC)
        msg = f'{color}{msg}{NC}'
        args = args[1:]
    self.logger.info(msg, *args, **kwargs)

  def infoSuccess(self, msg, *args, **kwargs):
    args += (dict(color='green'),)
    self.info(msg, *args, **kwargs)

  def infoWarn(self, msg, *args, **kwargs):
    args += (dict(color='yellow'),)
    self.info(msg, *args, **kwargs)
  
  def infoSafe(self, msg, *args, **kwargs):
    args += (dict(color='blue'),)
    self.info(msg, *args, **kwargs)

  def infoDanger(self, msg, *args, **kwargs):
    args += (dict(color='red'),)
    self.info(msg, *args, **kwargs)

  # When a method is not found, shortcuts it to instance's `logger` member's Method
  def __getattr__(self, name):
    return getattr(self.logger, name)
