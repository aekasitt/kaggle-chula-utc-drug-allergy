#!/usr/bin/python
# coding:utf-8
# Copyright (C) 2019-2020 All rights reserved.
# FILENAME:  singleton.py
# VERSION: 	 1.0
# CREATED: 	 2020-05-06 19:20
# AUTHOR: 	 Aekasitt Guruvanich <aekazitt@gmail.com>
# DESCRIPTION:
#
# HISTORY:
#*************************************************************
"""Module defining Parent-Class Singleton"""
from abc import ABCMeta, abstractmethod
import threading

class Singleton(object):
  __metaclass__ = ABCMeta
  instance = None
  mutex = threading.Lock()
  
  @classmethod
  def get_instance(cls, *args, **kwargs):
    """Get or Create Singleton instance"""
    if cls.instance is None:
      cls.mutex.acquire()
      if cls.instance is None:
        cls.instance = cls(*args, **kwargs)
      cls.mutex.release()
    return cls.instance

  @classmethod
  def release_instance(cls):
    """Release Singleton instance"""
    if cls.instance is not None:
      cls.instance.shutdown()
    cls.instance = None
  
  def __call__(self):
    raise TypeError('Singletons must be accessed through `get_instance()`.')

  @abstractmethod
  def shutdown(self):
    """Must be defined in children class"""
