#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 13:07:13 2017

@author: Carnec
"""

import networkx as nx
import random
import matplotlib.pyplot as plt
import csv
import math
import scipy



n = 20 # number of nodes
pn = 0.2 # per-edge probability of existing
p = 0.1 # probability of acquiring infection from a single neighbour, per time-step
i = 1 # number of nodes initially infected
td = 4 # td time-steps after infection, the individual dies
nsteps = 10 # how many time-steps to run

def infection_init(G):
    """Make a graph with some infected nodes."""
    for u in G.nodes():
        G.node[u]["state"] = 0
        G.node[u]["color"] = 'green'
        
    init = random.sample(G.nodes(), i)
    for u in init:
        G.node[u]["state"] = 1
        G.node[u]["color"] = 'blue'

def step(G):
    """Given a graph G, run one time-step."""
    new_state = {}
    for u, d in G.nodes(data=True):
        new_state[u] = infection_update(d["state"],
                                        (G.node[u2]["state"] for u2 in G.neighbors(u)))
    for u in G.nodes():
        G.node[u]["state"] = new_state[u]
        if G.node[u]["state"] < 0 :        #if the nodes is dead
            G.node[u]["color"] = 'red'     #set the node colour to red
            
def infection_update(s1, ss):
    """Update the state of node s1, given the states of its neighbours ss."""

    if s1 < 0:
        return s1 # s1 < 0 means node has died, so no change
    if s1 > td:
        return -1 # t time steps after infection, node dies
    if s1 > 0:
        return s1 + 1 # one time-step has elapsed

    # if not yet infected, each infected neighbour is a new risk!
    for s in ss:
        if s > 0: # neighbour s is infected but still alive
            if random.random() < p:
                # with probability p, become infected
                return 1
    return 0

def run():
    G = nx.erdos_renyi_graph(n, pn)

    infection_init(G)
    print("Time proportion_alive proportion_infected")
    for i in range(nsteps):
        step(G)
        palive = sum(G.node[i]["state"] >= 0 for i in G.nodes()) / n
        pinf = sum(G.node[i]["state"] > 0 for i in G.nodes()) / n
        print("%2d %.2f %.2f" % (i, palive, pinf))

    viz(G)
    
def viz(G):   
    #nx.draw(G)
    pos=nx.spring_layout(G)
    node_colors = []
    
    for i in G.nodes():
        node_colors.append(G.node[i]["color"])
    
    
    nx.draw_networkx_nodes(G,
            pos,
            linewidths=0.5,
            node_size=50,
            with_labels=False,
            node_color = node_colors)    
   
    nx.draw_networkx_edges(G,pos,edgelist=G.edges(),
            width=0.5,
            #edge_color=d_edge_colors,
            alpha=0.5,
            arrows=False)
       
    
    """    
    nx.draw_networkx_nodes(G,pos,
                       G.nodes(),
                       node_color='black',
                       node_size=50,
                   alpha=.9)
    """
    plt.show()

    
def real_world_airport_graph(nodes, edges):
    """ This function creates a graph using a databse of aiports and their associated routes.
    
    Airports are represented by nodes and routes by edges.
    
    """
    
    G = nx.Graph()
    
    duplicate_count = 0
    edge_count = 0 
    error_count = 0
    line_num = 0 
            
    nodes = 'airports_data.txt'        
    with open(nodes, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            entries = line.replace('"',"").rstrip().split(",")
            G.add_node(int(entries[0]),country=entries[3],name=entries[1], IATA = entries[4])
    
    edges = 'edges.txt'
    with open(edges, 'r', encoding="utf-8") as f:
        for line in f.readlines():
            entries = line.replace('"',"").rstrip().split(",")
            try:
                if G.has_edge(int(entries[3]),int(entries[5])):
                    duplicate_count += 1
                else:
                    if line_num > 1:
                        vertex1 = int(entries[3])
                        vertex2 = int(entries[5])
                        G.add_edge(vertex1, vertex2 )
                        G.edge[vertex1][vertex2]['IATA'] = entries[2]
                        G.edge[vertex1][vertex2]['IATA'] = entries[4]
                        edge_count += 1
            except ValueError:
                # The value doesn't exist
                error_count += 1
                pass
            line_num += 1
    return G
    
def largest_connected_component(G):
    largest = max(nx.connected_component_subgraphs(G), key=len)
    return largest
    
def nCk(n, k):
    # n choose k
    return scipy.misc.comb(n, k)
    
def graph_properties(G):
    n = G.order()
    m = G.size()
    d = G.degree()
    sqrt_n = math.sqrt(n)
    try:
        diam = nx.diameter(G)
    except:
        diam = math.inf
    return (
        n,
        m,
        m / nCk(n, 2), # density
        nx.average_clustering(G),
        diam,
        len([u for u in G.nodes() if d[u] > sqrt_n]), # num nodes high degree
        max(len(c) for c in nx.connected_components(G)) # len(largest comp)
    )

        
    
if __name__ == "__main__":
#    run()
    nodes = 'airports_data.txt'
    edges = 'edges.csv.txt'
    G = real_world_airport_graph(nodes, edges)
    order, size, density, cluster_coeff, diameter, num_nodes_deg_n, largest_comp = graph_properties(G)
