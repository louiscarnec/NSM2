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


        
    
airportdata = 'sub_us_airports.csv'   
edges = 'edges.txt'

G = nx.Graph()

duplicate_count = 0
edge_count = 0 
error_count = 0
line_num = 0 

USairports = set()   
with open(airportdata, 'r') as f:
    csvreader = csv.reader(f)
    for row in csvreader:
            if row[0] not in USairports:
                USairports.add(row[0])
            G.add_node(int(row[0]),country=row[3],name=row[1], IATA = row[4], population= int(row[11]))

with open(edges, 'r', encoding="utf-8") as f:
    for line in f.readlines():
        entries = line.replace('"',"").rstrip().split(",")
        if entries[3] and entries[5] in USairports:
            try:
                if G.has_edge(int(entries[3]),int(entries[5])) or G.has_edge(int(entries[5]),int(entries[3])):
                    duplicate_count += 1
                else:
                    if line_num > 1:
                        vertex1 = int(entries[3])
                        vertex2 = int(entries[5])
                        G.add_edge(vertex1, vertex2)
                        G.edge[vertex1][vertex2]['IATA1'] = entries[2]
                        G.edge[vertex1][vertex2]['IATA2'] = entries[4]
                        edge_count += 1
            except ValueError:
                # The value doesn't exist
                error_count += 1
                pass
        else:
            pass
            line_num += 1
        

    
                    