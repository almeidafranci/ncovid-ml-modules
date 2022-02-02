#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 17:43:02 2022

@author: davi
"""

import json
from urllib.request import urlopen
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import data_manner
from models import lstm_manner
import configs_manner as configs

rd = pd.read_csv("https://raw.githubusercontent.com/wcota/covid19br/master/cases-brazil-states.csv", index_col='date')
 
# Opening JSON file
# https://gist.githubusercontent.com/henriquejensen/1032c47a44d2cddaa2ef47fc531025db/raw/c58fdc848baf2a1fb53e617a0ad4e9754ec68e35/json-estados-brasileiros
#f = open('../dbs/json-estados-brasileiros')
 
# returns JSON object as
# a dictionary
response = urlopen("https://gist.githubusercontent.com/henriquejensen/1032c47a44d2cddaa2ef47fc531025db/raw/c58fdc848baf2a1fb53e617a0ad4e9754ec68e35/json-estados-brasileiros")
data = json.loads(response.read())