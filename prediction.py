#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 19:00:29 2020

@author: dcoster
"""

import pandas as pd 

data_file = "Datasets/Data.csv"
dataset = pd.read_csv(data_file)
X = dataset.il