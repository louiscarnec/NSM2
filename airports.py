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
import numpy as np
import operator
import matplotlib.pyplot as plt
import heapq




def infection_init(G,init):
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
    elif simulation == "node_deg":
        deg = G.degree()
        init = random.choice([n for n in deg if deg[n] > large_airports])
    else:
        init = simulation
        

    infection_init(G,init)
    
    print("Source Airport : ", init)
    print("Airport closeness centrality: ", nx.closeness_centrality(G,init))
    print("Airport clustering: ", nx.clustering(G,init))
#    print("Airport eccentricity: ", nx.eccentricity(G,init))

    
    w = nsteps #creating a matrix with % of infected, susceptible, recovered to plot 
    h = 4
    stats_matrix = [[0 for x in range(h)] for y in range(w)] 

    print("TimeStep Susceptible(%), Alive(%), Infected(%),Pdead(%)")
    for i in range(nsteps):
        step(G)
        psus = sum(G.node[i]["state"] == 0 for i in G.nodes()) / nx.number_of_nodes(G)
        palive = sum(G.node[i]["state"] >= 0 for i in G.nodes()) / nx.number_of_nodes(G)
        pinf = sum(G.node[i]["state"] > 0 for i in G.nodes()) / nx.number_of_nodes(G)
        prec = sum(G.node[i]["color"] > 'blue' for i in G.nodes()) / nx.number_of_nodes(G)
        pdead = sum(G.node[i]["state"] < 0 for i in G.nodes()) / nx.number_of_nodes(G)
        print("Step:%2d, psusceptible: %.2f  , palive: %.2f , pInfected: %.2f , pRecovered: %.2f, pDead: %.2f" % (i,psus, palive, pinf, prec,pdead))
        
        stats_matrix[i][0] = psus
        stats_matrix[i][1] = pinf
        stats_matrix[i][2] = prec
        stats_matrix[i][3] = pdead

        viz(G,pos) 
        
     
#    stats(G,init)
    
    return G, init, stats_matrix
    

    
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
            linewidths=0.7,
            node_size=20,
            with_labels=False,
            node_color = node_colors
            )    
   
    nx.draw_networkx_edges(G,pos,edgelist=G.edges(),
            width=0.7,
            edge_color=edge_colors,
            alpha=0.5,
            arrows=False)
       
    
    plt.show()

    
def real_world_airport_graph(nodes, edges):
    """ This function creates a graph using a database of aiports and their associated routes.
    
    Airports are represented by nodes and routes by edges."""
    
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
        max(len(c) for c in nx.connected_components(G)), # len(largest comp)
#        nx.average_shortest_path_length(G)
    )
    
def short_path_test(G,n):
    SP = nx.single_source_dijkstra_path_length(G, n) 
    
    key, value = max(SP.items(), key=lambda x:x[1])    
    
    print("Longest Shortest Path,", key, "of length ", value)
    
    status = G.node[key]['color']

    if status == 'green':
        print("not infected")
    elif status == 'blue':
        print("recovered")
    else:
        print("dead")
    
    return status
    
def max_btcentrality(G):
    bcent = nx.betweenness_centrality(G)
    return max(bcent.items(), key=lambda x:x[1:])
    
def min_btcentrality(G):
    bcent = nx.betweenness_centrality(G)
    return min(bcent.items(), key=lambda x:x[1:])   
    
def max_degcentrality(G):
    degcent = nx.degree_centrality(G)
    return max(degcent.items(), key=lambda x:x[1:]) 
    
def min_degcentrality(G):
    degcent = nx.degree_centrality(G)
    return min(degcent.items(), key=lambda x:x[1:])    

def gDiameterTest(n,nsteps):
    diameterList = [["pn"],["diameter"],["inftime"],["rectime"],["deadtime"]]
    
    timeinf = 0
    timerec = 0
    timedead = 0
    
    for i in range(10):
        print("Erdos-Renyi Graph with ", i ," probability of edge existing")
        frac = i/100
        G = nx.erdos_renyi_graph(n, frac)
        for u,v,d in G.edges(data=True):
            d['weight']=random.random()
        try:
            diameter = nx.diameter(G)
        except nx.NetworkXError:
            diameter = math.inf
            
        G, init, matrix = run(G,'random')
        
        for nstep in range(nsteps):
            if matrix[nstep][1] == 1.0:
                timeinf = nstep
                break
            else:
                timeinf = None
            if matrix[nstep][2] == 0.0:
                timerec = nstep
                break
            else:
                timerec = None
            if matrix[nstep][3] == 1.0:
                timedead = nstep  
                break
            else:
                timedead = None
                
        diameterList[0].append(frac)
        diameterList[1].append(diameter)
        diameterList[2].append(timeinf)
        diameterList[3].append(timerec)
        diameterList[4].append(timedead)
        
        plotting(matrix, nsteps, "Erdos-Renyi with pn" + str(frac))

    return diameterList
    
def subgraph(G,nsteps):
    
    "subgraph with only edges of weight >0.75"
    SG_largeweight = nx.Graph([(u,v,d) for u,v,d in Greal.edges(data=True) if d['weight']>0.75] )
    
    "subgraph with only edges of weight <0.5"
    SG_lowweight = nx.Graph([(u,v,d) for u,v,d in Greal.edges(data=True) if d['weight']<0.5] )
        
    "subgraph with 20 largest degree-centrality nodes"
    l = heapq.nlargest(20, zip(nx.degree_centrality(Greal).values(),nx.degree_centrality(Greal)))
    locs = []
    for i in range(len(l)): locs.append(l[i][1])
    SG_top20DC = nx.subgraph(G, [i for i in locs])
    
    "subgraph with 20 lowest degree-centrality nodes"
    l = heapq.nsmallest(100, zip(nx.degree_centrality(Greal).values(),nx.degree_centrality(Greal)))
    locs = []
    for i in range(len(l)): locs.append(l[i][1])
    SG_low20DC = nx.subgraph(G, [i for i in locs])
    
    "Minimum Spanning Tree"
    minspan = nx.minimum_spanning_tree(G)

        
    return SG_largeweight, SG_lowweight, SG_top20DC, SG_low20DC, minspan

    
def resultssubgraph(G,nsteps,sim_str):
    
    subgraphlist = [["diameter"],["inftime"],["rectime"],["deadtime"]]
    
    timeinf = 0
    timerec = 0
    timedead = 0
    
    try:
        diameter = nx.diameter(G)
    except nx.NetworkXError:
        diameter = math.inf
       
    G, init, matrix = run(G,'random')
        
    for nstep in range(nsteps):
        if matrix[nstep][1] == 1.0:
            timeinf = nstep
            break
        else:
            timeinf = None
        if matrix[nstep][2] == 0.0:
            timerec = nstep
            break
        else:
            timerec = None
        if matrix[nstep][3] == 1.0:
            timedead = nstep  
            break
        else:
            timedead = None
            
    subgraphlist[0].append(diameter)
    subgraphlist[1].append(timeinf)
    subgraphlist[2].append(timerec)
    subgraphlist[3].append(timedead) 
    
    
    return subgraphlist
            
        
    
    
#    print("---")
#    print("Source node : ", init)
#    print("Degree: ", nx.degree(G,init))
#    print("Closeness Centrality: ", nx.closeness_centrality(G,init))
#
#    
#    for i in G.nodes():
#        if G.node[i]["state"] == 0:
#            print("---")
#            print("Susceptible node : ", i)
#            print("Degree: ", nx.degree(G,i))
#            print("Closeness Centrality: ", nx.closeness_centrality(G,i))
#
#
#        elif G.node[i]["state"] >= 0:
#            print("---")
#            print("Alive node : ",i) 
#            print("Degree: ", nx.degree(G,i))
#            print("Closeness Centrality: ", nx.closeness_centrality(G,i))
#            
#        elif G.node[i]["state"] > 0:
#            print("---")
#            print("Infected node : ", i) 
#            print("Degree: ", nx.degree(G,i))
#            print("Closeness Centrality: ", nx.closeness_centrality(G,i))
#        

def plotting(matrix, nsteps, sim_str):
    plt.plot([i for i in range(len(matrix))],[matrix[i][0] for i in range(len(matrix))],'--go',label = '% Susceptible')
    plt.plot([i for i in range(len(matrix))],[matrix[i][1] for i in range(len(matrix))],'--y^',label = '% Infected')
    plt.plot([i for i in range(len(matrix))],[matrix[i][2] for i in range(len(matrix))],'--bs',label = '% Recovered')
    plt.plot([i for i in range(len(matrix))],[matrix[i][3] for i in range(len(matrix))],'--ro',label = '% Closed/Dead')
    plt.title(str(sim_str))
    plt.xlabel('Time Step')
    plt.ylabel('%')
    plt.xlim(-0.5, nsteps)
    plt.ylim(-0.1, 1.1)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
       ncol=2, fancybox=True, borderaxespad=0.)
    plt.show()


        
    
if __name__ == "__main__":
    
    "Overall Parameters"
    n = 143 # number of nodes
    pn = 0.04 # per-edge probability of existing
    i = 1 # number of nodes initially infected
    nsteps = 25 # how many time-steps to run
    large_airports = 50 # picking the initial edpidemic spreading airport to have degree greater than n
    
    "Create Real World Graph"
    nodes = 'sub_us_airports.csv' 
    edges = 'edges.txt'
    Greal = real_world_airport_graph(nodes, edges)
    #remove nodes which have degree zero OR Keep nodes with degree > 0
    deg = Greal.degree()
    #to_remove = [n for n in deg if deg[n] == 0]
    to_keep = [n for n in deg if deg[n] != 0]
    Gsub = Greal.subgraph(to_keep)
    Greal = normalize_edge_weight(Gsub)
    
    """Simulation 1 - Airports cannot die or recover"""
    
    print("Simulation 1 - Airports cannot die or recover")
    
    p = 0.4 # probability of acquiring infection from a single neighbour, per time-step
    rp = 0 # probability of recovery 
    td = math.inf # td time-steps after infection, the individual dies

    print("Testing on graphs with increasing probability of edge existence between nodes" )    
    Diameter_test_data = gDiameterTest(n,nsteps)
    
    print("Testing on subgraphs of the real-world airport graph")
    SG_largeweight, SG_lowweight, SG_top20DC, SG_low20DC, minspan = subgraph(Greal,nsteps)
    print("Subgraph of 20 largest edge weight")
    largwlist = resultssubgraph(SG_largeweight,nsteps,"Subgraph 20 largest edge weights")
    print("Subgraph of 20 lowest edge weight")
    lowwlist = resultssubgraph(SG_lowweight,nsteps,"Subgraph 20 lowest edge weights")
    print("Subgraph of 20 largest degree centrality nodes")
    top20list = resultssubgraph(SG_top20DC,nsteps,"Subgraph 20 largest degree centrality nodes")
    print("Subgraph of 20 lowest degree centrality nodes")
    low20list = resultssubgraph(SG_low20DC,nsteps,"Subgraph 20 lowest degree centrality nodes")
    print("Minimum Spanning Tree")
    minspanlist = resultssubgraph(minspan,nsteps,"Minimum Spanning Tree")
 
    print("Test using nodes with maximimum and minimum betweenness centrality")
    maxkeybc, maxvalbc = max_btcentrality(Greal)
    minkeybc, minvalbc = min_btcentrality(Greal)
    print("Max Betweenness Centrality Source Node")
    Gmaxb, init, matrix_maxbc = run(Greal, maxkeybc)
    plotting(matrix_maxbc, nsteps, "Max Betweeness Centrality Source Node")
    print("Min Betweenness Centrality Source Node")
    Gminb, init, matrix_minbc = run(Greal, minkeybc)
    plotting(matrix_minbc, nsteps, "Min Betweeness Centrality Source Node")

    print("Test using nodes with maximimum and minimum degree centrality")
    maxkeydegc, maxvaldegc = max_degcentrality(Greal)
    minkeydegc, minvaldegc = min_degcentrality(Greal)
    print("Max Degree Centrality Source Node")
    Gmaxdeg, init, matrix_maxdegc = run(Greal, maxkeydegc)
    plotting(matrix_maxdegc, nsteps, "Max Degree Centrality Source Node")
    print("Min Degree Centrality Source Node")
    Gmindeg, init, matrix_mindegc = run(Greal, minkeydegc)
    plotting(matrix_mindegc, nsteps, "Min Degree Centrality Source Node")
    
    print("Test using center of graph")
    center = nx.center(Greal)
    Gcenter, init, matrix_center = run(Greal, center[0])
    plotting(matrix_center, nsteps, "Initial Node: Center of Graph")


#    "Simulation 2 - Airports can die and cannot recover"
#    
#    p = 0.5 # probability of acquiring infection from a single neighbour, per time-step
#    rp = 0 # probability of recovery 
#    td = 4 # td time-steps after infection, the individual dies
#    Diameter_test_data = gDiameterTest(n,nsteps)
#
#
#    "Simulation 3 - Airports can die and recover"
#    p = 0.5 # probability of acquiring infection from a single neighbour, per time-step
#    rp = 0.2 # probability of recovery 
#    td = 4 # td time-steps after infection, the individual dies
#
#    
#
#   


#    print(short_path_test(G_er, init_er))
#    
#    print("----------")
#    print("Real World Graph")
#    nodes = 'sub_us_airports.csv' 
#    edges = 'edges.txt'
#    Greal = real_world_airport_graph(nodes, edges)
#    #remove nodes which have degree zero OR Keep nodes with degree > 0
#    deg = Greal.degree()
#    #to_remove = [n for n in deg if deg[n] == 0]
#    to_keep = [n for n in deg if deg[n] != 0]
#    Gsub= Greal.subgraph(to_keep)
#    Greal= normalize_edge_weight(Gsub)
#    
#    order, size, density, cluster_coeff, diameter, num_nodes_deg_n, largest_comp, av_shortest_path = graph_properties(Greal)
#    print(graph_properties(Greal))
#
#    G_real, init_real = run(Greal,"node_deg") 
#    print(short_path_test(G_real, init_real))