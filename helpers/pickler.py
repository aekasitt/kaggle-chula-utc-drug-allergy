#!/usr/bin/env python
# coding: utf-8
# Copyright (C) 2005-2020 All rights reserved.
# FILENAME: 	 pickler.py
# VERSION: 	   1.0
# CREATED: 	   2020-03-19 15:12
# AUTHOR: 	   Aekasitt Guruvanich <aekazitt@gmail.com>
# DESCRIPTION:
#
# HISTORY:
#*************************************************************
"""Module definining Pickler class accessing saved pickles"""
# import asyncio
import os
import sys
from pickle import dump, load, PicklingError, UnpicklingError, HIGHEST_PROTOCOL
### Local Modules ###
from helpers.singleton import Singleton

class Pickler(Singleton):
  def delete(self, name:str, folder:str='pickles') -> bool:
    """
    Delete a saved pickle from pickles/ folder
    """
    filepath = '%s/%s.pkl' % (folder, name)
    try:
      os.remove(filepath)
      return True
    except OSError: # File Not Found
      return False

  def load(self, name:str='undefined', folder:str='pickles') -> object:
    """
    Load an Object saved from pickles/ folder
    """
    filename = '%s.pkl' % name
    filepath = '%s/%s' % (folder, filename)
    try:
      with open(filepath, 'rb') as file:
        try:
          return load(file)
        except UnicodeDecodeError:
          import hashlib
          return hashlib.sha1(file.read()).hexdigest()
    except (IOError, EOFError): # File Not Found
      return None
    except UnpicklingError:
      return None

  def save(self, obj:object, name:str='undefined', folder:str='pickles') -> bool:
    """
    Save an Object as pickle in pickles/ folder
    """
    filename = '%s.pkl' % name
    filepath = '%s/%s' % (folder, filename)
    try:
      with open(filepath, 'wb') as handle:
        dump(obj, handle, protocol=HIGHEST_PROTOCOL)
      return True
    except PicklingError:
      return False
