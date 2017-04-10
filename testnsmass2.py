#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 17:19:59 2017

@author: Carnec
"""

import networkx as nx
import random
import matplotlib.pyplot as plt
import csv

#with open('airports_data.csv', newline='') as csvfile: 
#        airports = csv.reader(csvfile)
#        for row in airports:
#            print(row)

nodes = open('/Users/Carnec/Documents/NSM2/airports_data.txt')

#with open(nodes, 'r', encoding='utf-8') as f:
#
#    for line in f.readlines():
#        entries = line.replace('"',"").rstrip().split(",")