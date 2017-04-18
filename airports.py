#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 13:07:13 2017

@authors: Louis Carnec, Vijay Katta and Adedayo Adelowokan
"""

import networkx as nx
import random
import matplotlib.pyplot as plt
import csv
import math
import scipy



def infection_init(G,simulation,init):
    """Make a graph with some infected nodes."""
    for u in G.nodes():
        G.node[u]["state"] = 0
        G.node[u]["color"] = 'green'
        
        G.node[init]["state"] = 1
        G.node[init]["color"] = 'yellow'

def step(G):
    """Given a graph G, run one time-step."""
    new_state = {}
    for u, d in G.nodes(data=True):
        new_state[u] = infection_update(d["state"],
                                       ((G.node[u2]["state"],G.edge[u][u2]['weight']) for u2 in G.neighbors(u)))
        
    for u in G.nodes():
        G.node[u]["state"] = new_state[u]
        if G.node[u]["state"] < 0 :        #if the nodes is dead
            G.node[u]["color"] = 'red'     #set the node colour to red
    
    #recover from infection
    for u in G.nodes():
        if G.node[u]["state"] > 0 :        #if the nodes is infected
            if random.random() < rp/G.node[u]["state"] :
                G.node[u]["state"] = 0     # reset to susceptible
                G.node[u]["color"] = 'blue'     #set the node colour to blue

                
def infection_update(s1, ss_w):
    """Update the state of node s1, given the states of its neighbours ss."""   
    if s1 < 0:
        return s1 # s1 < 0 means node has died, so no change
    if s1 > td:
        return -1 # t time steps after infection, node dies
    if s1 > 0:
        return s1 + 1 # one time-step has elapsed

    # if not yet infected, each infected neighbour is a new risk!
    for nnodestate,edgeweight in ss_w:
        if nnodestate > 0: # neighbour s is infected but still alive
            if random.random() < p * edgeweight : 
                #nodes with higher weight have have higher chance of spreading the disease
                return 1
    
    return 0
                   
def normalize_edge_weight(G):
    mx_weight =0
    
    for (u,v,d) in G.edges(data=True):   # get the maximum edge weight
        if d['weight'] > mx_weight:
            mx_weight =d['weight'] 
    
    for (u,v,d) in G.edges(data=True):   
         d['weight'] = (d['weight'] / mx_weight) # divide each edge by the maximum edge weight
         
    return G
    
def run(G,simulation):
    pos=nx.spring_layout(G)
    
    if simulation == "random":    
        init = random.choice(G.nodes())
    if simulation == "node_deg":
        deg = G.degree()
        init = random.choice([n for n in deg if deg[n] > large_airports])

    infection_init(G,simulation,init)
    
    print("Source Airport : ", init)
    print("Airport closeness centrality: ", nx.closeness_centrality(G,init))
    print("Airport clustering: ", nx.clustering(G,init))
    print("Airport eccentricity: ", nx.eccentricity(G,init))

    
    print("TimeStep Susceptible(%), Alive(%), Infected(%),Pdead(%)")
    for i in range(nsteps):
        step(G)
        psus = sum(G.node[i]["state"] == 0 for i in G.nodes()) / nx.number_of_nodes(G)
        palive = sum(G.node[i]["state"] >= 0 for i in G.nodes()) / nx.number_of_nodes(G)
        pinf = sum(G.node[i]["state"] > 0 for i in G.nodes()) / nx.number_of_nodes(G)
        prec = sum(G.node[i]["color"] > 'blue' for i in G.nodes()) / nx.number_of_nodes(G)
        pdead = sum(G.node[i]["state"] < 0 for i in G.nodes()) / nx.number_of_nodes(G)
        print("Step:%2d, psusceptible: %.2f  , palive: %.2f , pInfected: %.2f , pRecovered: %.2f, pDead: %.2f" % (i,psus, palive, pinf, prec,pdead))
        viz(G,pos)    
    
def viz(G,pos):   
    node_colors = []
    edge_colors = []

    for i in G.nodes():
        if G.node[i]["state"] == 1 :        #if the nodes is infected
            G.node[i]["color"] = 'yellow'     #set the node colour to yellow
        node_colors.append(G.node[i]["color"])
    
    for u,v,e in G.edges(data=True):
        if e['weight'] < 0.5 : 
           e["color"] = 'black'     #set the edge colour 
        elif  (e['weight'] >= 0.5 ) and (e['weight'] <= 0.75) :
           e["color"] = 'darkred'     #set the edge colour 
        else:
           e["color"] = 'lime'     #set the edge colour 
        edge_colors.append(e["color"])  
        
    nx.draw_networkx_nodes(G,
            pos,
            linewidths=0.5,
            node_size=20,
            with_labels=False,
            node_color = node_colors
            )    
   
    nx.draw_networkx_edges(G,pos,edgelist=G.edges(),
            width=0.5,
            edge_color=edge_colors,
            alpha=0.5,
            arrows=False)
       
    
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

    USairports = set()  
    
    with open(nodes, 'r') as f:
        csvreader = csv.reader(f)
        for row in csvreader:
            #print(row[4])
            if row[4] != '':
                if row[4] not in USairports:
                    USairports.add(row[4])
                G.add_node(int(row[0]),country=row[3],name=row[1], IATA = row[4], population= int(row[11]))
                #print(row[11])
    with open(edges, 'r', encoding="utf-8") as f:
        for line in f.readlines():
            entries = line.replace('"',"").rstrip().split(",")
            if entries[2] in USairports:
                if entries[4] in USairports:
                    try:
                        if G.has_edge((entries[2]),(entries[4])):
                            duplicate_count += 1
                        else:
                            if line_num > 1:
                                vertex1 = (entries[2])
                                vertex2 = (entries[4])
                                #print((entries[2],entries[4]))
                                #print((entries[3],entries[5]))
                                edge_weight = (int(entries[3]) + int(entries[5]))/2
                                G.add_edge(vertex1, vertex2 ,weight= edge_weight)
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
    
    n = 30 # number of nodes
    pn = 0.2 # per-edge probability of existing
    p = 0.1 # probability of acquiring infection from a single neighbour, per time-step
    rp = 0.4 # probability of recovery 
    i = 1 # number of nodes initially infected
    td = 4 # td time-steps after infection, the individual dies
    nsteps = 25 # how many time-steps to run
    large_airports = 50 # picking the initial edpidemic spreading airport to have degree greater than n
    
    G = nx.erdos_renyi_graph(n, pn)
    
    for u,v,d in G.edges(data=True):
        d['weight']=random.random()
   
    #print(G.edges(data=True))
#    run(G,"random")
    
    nodes = 'sub_us_airports.csv' 
    edges = 'edges.txt'
    G = real_world_airport_graph(nodes, edges)
    
    #remove nodes which have degree zero OR Keep nodes with degree > 0
    deg = G.degree()
    #to_remove = [n for n in deg if deg[n] == 0]
    to_keep = [n for n in deg if deg[n] != 0]
    G= G.subgraph(to_keep)
    G= normalize_edge_weight(G)
    
    order, size, density, cluster_coeff, diameter, num_nodes_deg_n, largest_comp = graph_properties(G)
    print(graph_properties(G))

    run(G,"node_deg")
  