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


        
    
edges = 'edges.txt'
with open(edges, 'r', encoding="utf-8") as f:
    for line in f.readlines():
        entries = line.replace('"',"").rstrip().split(",")


    