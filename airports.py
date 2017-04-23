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
from collections import defaultdict
from tabulate import tabulate



def infection_init(G,init): #initiate the state and color of node
    """Make a graph with some infected nodes."""
    for u in G.nodes():
        G.node[u]["state"] = 0 #Susceptible nodes
        G.node[u]["color"] = 'green'
        
    G.node[init]["state"] = 1 #Initial infected node
    G.node[init]["color"] = 'yellow'

            

def step(G): #Running time step, call infection_update to get changes to be applied to the state of nodes
    """Given a graph G, run one time-step."""
    new_state = {}
    for u, d in G.nodes(data=True): #get new state for each node
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
    
                   
def normalize_edge_weight(G): #normalise edge weights
    mx_weight =0
    
    for (u,v,d) in G.edges(data=True):   # get the maximum edge weight
        if d['weight'] > mx_weight:
            mx_weight =d['weight'] 
    
    for (u,v,d) in G.edges(data=True):   
         d['weight'] = (d['weight'] / mx_weight) # divide each edge by the maximum edge weight
         
    return G
    
def run(G,simulation): #run simulation   
    
    pos=nx.spring_layout(G)
    
    if simulation == "random": #simulation using a random node as initial infected node    
        init = random.choice(G.nodes())
    elif simulation == "node_deg": #simulation using initial infected node with deg > value
        deg = G.degree()
        init = random.choice([n for n in deg if deg[n] > large_airports])
    else:
        init = simulation #simulate using node provided as parameter
        

    infection_init(G,init) #initial G with initial states
    
#    print("Source Airport : ", init)
#    print("Airport closeness centrality: ", nx.closeness_centrality(G,init))
#    print("Airport clustering: ", nx.clustering(G,init))
#    print("Airport eccentricity: ", nx.eccentricity(G,init))
    pinf_all_step = math.inf
    pdead_all_step = math.inf
    pinf_all_step_check =0
    pdead_all_step_check=0
    w = nsteps #creating a matrix with % of infected, susceptible, recovered to plot 
    h = 4
    stats_matrix = [[0 for x in range(h)] for y in range(w)] 

#    print("TimeStep Susceptible(%), Alive(%), Infected(%),Pdead(%)")
    for i in range(nsteps):
        step(G)
        psus = sum(G.node[i]["state"] == 0 for i in G.nodes()) / nx.number_of_nodes(G)
        palive = sum(G.node[i]["state"] >= 0 for i in G.nodes()) / nx.number_of_nodes(G)
        pinf = sum(G.node[i]["state"] > 0 for i in G.nodes()) / nx.number_of_nodes(G)
        prec = sum(G.node[i]["color"] > 'blue' for i in G.nodes()) / nx.number_of_nodes(G)
        pdead = sum(G.node[i]["state"] < 0 for i in G.nodes()) / nx.number_of_nodes(G)
#        print("Step:%2d, psusceptible: %.2f  , palive: %.2f , pInfected: %.2f , pRecovered: %.2f, pDead: %.2f" % (i,psus, palive, pinf, prec,pdead))
        
        stats_matrix[i][0] = psus #append matrix
        stats_matrix[i][1] = pinf
        stats_matrix[i][2] = prec
        stats_matrix[i][3] = pdead
        
        if (pinf ==1 and pinf_all_step_check ==0 ):
            pinf_all_step = i
            pinf_all_step_check =1
        if (pdead ==1 and pdead_all_step_check ==0 ):
            pdead_all_step = i
            pdead_all_step_check = 1
            
    #viz(G,pos) #Visualise graph at each time step
        
        
     
#    stats(G,init)
    
    return G, init, stats_matrix,pinf_all_step,pdead_all_step
    
def viz(G,pos):   #Function to visualise graph at each time step
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

    
def real_world_airport_graph(nodes, edges): #.txt and .csv airport data to graph 
    """ This function creates a graph using a database of aiports and their associated routes.
    
    Airports are represented by nodes and routes by edges."""
    
    G = nx.Graph() #initiate graph
    
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
            row = line.replace('"',"").rstrip().split(",")
            if row[2] in USairports:
                if row[4] in USairports:
                    try:
                        if G.has_edge((row[2]),(row[4])):
                            duplicate_count += 1
                        else:
                            if line_num > 1:
                                vertex1 = (row[2])
                                vertex2 = (row[4])
                                edge_weight = (int(row[3]) + int(row[5]))/2
                                G.add_edge(vertex1, vertex2 ,weight= edge_weight)
                                edge_count += 1
                    except ValueError:
                        # The value doesn't exist
                        error_count += 1
                        pass
                    line_num += 1
    return G
    
def stat(G,init): #Return statistics dic
    stats = {}
    stats['Initial Node'] = init
    stats['Node Degree'] = nx.degree(G,init)
    stats['Closness Centrality'] = nx.closeness_centrality(G,init)
           
    psus = sum(G.node[i]["state"] == 0 for i in G.nodes()) / nx.number_of_nodes(G)
    stats['Percentage Susceptible'] = psus
    pinf = sum(G.node[i]["state"] > 0 for i in G.nodes()) / nx.number_of_nodes(G)
    stats['Percentage Infected'] = pinf
    pdead = sum(G.node[i]["state"] < 0 for i in G.nodes()) / nx.number_of_nodes(G)
    stats['Percentage Dead'] = pdead

    meandegsus = (np.mean([nx.degree(G,i) for i in G.nodes() if G.node[i]["state"] == 0]))
    stats['Mean Degree of Susceptible Nodes'] = meandegsus
    meanclossus = (np.mean([nx.closeness_centrality(G,i) for i in G.nodes() if G.node[i]["state"] == 0]))
    stats['Mean Closeness Centrality of Susceptible Nodes'] = meanclossus

    meandeginf = (np.mean([nx.degree(G,i) for i in G.nodes() if G.node[i]["state"] > 0]))
    stats['Mean Degree of Infected Nodes'] = meandeginf
    meanclosinf = (np.mean([nx.closeness_centrality(G,i) for i in G.nodes() if G.node[i]["state"] > 0]))
    stats['Mean Closeness Centrality of Infected Nodes'] = meanclosinf

    meandegdead = (np.mean([nx.degree(G,i) for i in G.nodes() if G.node[i]["state"] < 0]))
    stats['Mean Degree of Dead Nodes'] = meandegdead
    meanclosdead = (np.mean([nx.closeness_centrality(G,i) for i in G.nodes() if G.node[i]["state"] < 0]))
    stats['Mean Closeness Centrality of Dead Nodes'] = meanclosdead
    return stats

    
    
def largest_connected_component(G): #return largest connected components
    largest = max(nx.connected_component_subgraphs(G), key=len)
    return largest
    
def nCk(n, k):
    # n choose k
    return scipy.misc.comb(n, k)
    
def graph_properties(G): #return graph properties
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
    
def max_btcentrality(G): #return node with max betweenness centrality
    bcent = nx.betweenness_centrality(G)
    return max(bcent.items(), key=lambda x:x[1:])
    
def min_btcentrality(G):#return node with min betweenness centrality
    bcent = nx.betweenness_centrality(G)
    return min(bcent.items(), key=lambda x:x[1:])   
    
def max_degcentrality(G): #return node with max degree centrality
    degcent = nx.degree_centrality(G)
    return max(degcent.items(), key=lambda x:x[1:]) 
    
def min_degcentrality(G):#return node with min degree centrality
    degcent = nx.degree_centrality(G)
    return min(degcent.items(), key=lambda x:x[1:])    

def gDiameterTest(n,nsteps,simulationnumber): #test a range of erdos_renyi graphs
    diameterList = [["pn"],["diameter"],["inftime"],["rectime"],["deadtime"]]
    statlist = []

    frac=0

    timeinf = 0
    timerec = 0
    timedead = 0
    
    p = [1,3,5,9]#[1,3,5,7,9,11] #1,3,5,9

    w = 9 #creating a matrix with % of infected, susceptible, recovered to plot 
    h = len(p)
    stats_matrix = [[0 for x in range(w)] for y in range(h)] 
    for i in range(len(p)):
        stats_matrix[i][0]=i
        stats_matrix[i][1]=p[i]

    for i in range(len(p)):
        print("Erdos-Renyi Graph with ", frac ," probability of edge existing")
        frac = p[i]/100
        G = nx.erdos_renyi_graph(n, frac)
        properties = graph_properties(G)
        for j in range(len(properties)):
            stats_matrix[i][j+2]=properties[j]
        for u,v,d in G.edges(data=True):
            d['weight']=random.random()
        try:
            diameter = nx.diameter(G)
        except nx.NetworkXError:
            diameter = math.inf
            
        G, init, matrix,pinf_all_step,pdead_all_step = run(G,'random')
        
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
        
        #plotting(matrix, nsteps, "Erdos-Renyi with pn" + str(frac),simulationnumber)
        #print("properties:",properties)
        statlist.append(i)
        statlist.append(stat(G,init).items())

    return diameterList, stats_matrix,G,pinf_all_step,pdead_all_step
    
def subgraph(G,nsteps): #Groduce subgraphs
    
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

    
def resultssubgraph(G,nsteps,sim_str,simulationnumber): #return statistics for subgraphs
    
    subgraphlist = [["diameter"],["inftime"],["rectime"],["deadtime"]]
    
    timeinf = 0
    timerec = 0
    timedead = 0
    
    try:
        diameter = nx.diameter(G)
    except nx.NetworkXError:
        diameter = math.inf
       
    G, init, matrix,pinf_all_step,pdead_all_step = run(G,'random')
        
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
    
    statistics = stat(G,init)
    #print(matrix)
    #plotting(matrix, nsteps, sim_str, simulationnumber)

    return subgraphlist, graph_properties(G)
            
def testingPrint(simulationnumber,graphtype): #range of tests for each simulation
#    print("---")
#
    print("Testing on graphs with increasing probability of edge existence between nodes" )    
    Diameter_test_data, graphproperties, ERG,pinf_all_step,pdead_all_step = gDiameterTest(n,nsteps,simulationnumber)

    if graphtype == 'generated' :
        air_graph = ERG
    else :
        air_graph = Greal    
    
    print("pinf_all_step = ",pinf_all_step," pdead_all_step = ",pdead_all_step, " Diameter = ",nx.diameter(air_graph))
    

#    print(graphproperties)
#    print(Diameter_test_data)
#    print(stats)
#    
#    print("---")
#    print("Testing on real-world Airport Graph")
#    G, init, matrix ,pinf_all_step,pdead_all_step = run(air_graph,'random')
#    stat(G,init)
#    
#    print("Testing on subgraphs of the real-world airport graph")
#    SG_largeweight, SG_lowweight, SG_top20DC, SG_low20DC, minspan = subgraph(air_graph,nsteps)
#  
#    print("Subgraph of 20 largest edge weight")
#    largwlist,stats = resultssubgraph(SG_largeweight,nsteps,"Subgraph largest edge weights",simulationnumber)
#    
#    print("Subgraph of 20 lowest edge weight")
#    print(stats)
#    
#    lowwlist, stats = resultssubgraph(SG_lowweight,nsteps,"Subgraph lowest edge weights",simulationnumber)
#    print("Subgraph of 20 largest degree centrality nodes")
#    print(stats)
#    top20list, stats = resultssubgraph(SG_top20DC,nsteps,"Subgraph 20 largest degree centrality nodes",simulationnumber)
#    print("Subgraph of 20 lowest degree centrality nodes")
#    print(stats)
#    low20list, stats = resultssubgraph(SG_low20DC,nsteps,"Subgraph 20 lowest degree centrality nodes",simulationnumber)
#    print("Minimum Spanning Tree")
#    minspanlist, stats = resultssubgraph(minspan,nsteps,"Minimum Spanning Tree",simulationnumber)
#    print(stats)


        
    print("---")
    
    print("Test using nodes with maximimum and minimum betweenness centrality",simulationnumber)
    maxkeybc, maxvalbc = max_btcentrality(air_graph)
    minkeybc, minvalbc = min_btcentrality(air_graph)
    print("Max Betweenness Centrality Source Node")
    Gmaxb, init, matrix_maxbc,pinf_all_step,pdead_all_step = run(air_graph, maxkeybc)
    plotting(matrix_maxbc, nsteps, "Max Betweeness Centrality Source Node",simulationnumber)
    print(stat(Gmaxb,init))
    print("pinf_all_step = ",pinf_all_step," pdead_all_step = ",pdead_all_step, " Diameter = ",nx.diameter(air_graph))

    print("Min Betweenness Centrality Source Node")
    Gminb, init, matrix_minbc ,pinf_all_step,pdead_all_step = run(air_graph, minkeybc)
    plotting(matrix_minbc, nsteps, "Min Betweeness Centrality Source Node",simulationnumber)
    print(stat(Gminb,init))
    print("pinf_all_step = ",pinf_all_step," pdead_all_step = ",pdead_all_step, " Diameter = ",nx.diameter(air_graph))

    
    print("---")

    print("Test using nodes with maximimum and minimum degree centrality")
    maxkeydegc, maxvaldegc = max_degcentrality(air_graph)
    minkeydegc, minvaldegc = min_degcentrality(air_graph)
    
    print("Max Degree Centrality Source Node")
    Gmaxdeg, init, matrix_maxdegc,pinf_all_step,pdead_all_step = run(air_graph, maxkeydegc)
    plotting(matrix_maxdegc, nsteps, "Max Degree Centrality Source Node",simulationnumber)
    print(stat(Gmaxdeg,init))
    print("pinf_all_step = ",pinf_all_step," pdead_all_step = ",pdead_all_step, " Diameter = ",nx.diameter(air_graph))
    
    print("Min Degree Centrality Source Node")
    Gmindeg, init, matrix_mindegc ,pinf_all_step,pdead_all_step= run(air_graph, minkeydegc)
    plotting(matrix_mindegc, nsteps, "Min Degree Centrality Source Node",simulationnumber)
    print(stat(Gmindeg,init))
    print("pinf_all_step = ",pinf_all_step," pdead_all_step = ",pdead_all_step," Diameter = ",nx.diameter(air_graph))
    
    print("---")
    
    print("Test using center of graph")
    center = nx.center(air_graph)
    Gcenter, init, matrix_center ,pinf_all_step,pdead_all_step= run(air_graph, center[0])
    plotting(matrix_center, nsteps, "Initial Node: Center of Graph",simulationnumber)
    print(stat(Gcenter,init))
    print("pinf_all_step = ",pinf_all_step," pdead_all_step = ",pdead_all_step, " Diameter = ",nx.diameter(air_graph)," Length = ",len(air_graph))  
#    
#    


def plotting(matrix, nsteps, sim_str, simulationnumber): #plot percentage of nodes in eahc state over time
    if simulationnumber == 'sim1':
        plt.plot([i for i in range(len(matrix))],[matrix[i][0] for i in range(len(matrix))],'--go',label = '% Susceptible')
        plt.plot([i for i in range(len(matrix))],[matrix[i][1] for i in range(len(matrix))],'--y^',label = '% Infected')
    elif simulationnumber == 'sim2':  
        plt.plot([i for i in range(len(matrix))],[matrix[i][0] for i in range(len(matrix))],'--go',label = '% Susceptible')
        plt.plot([i for i in range(len(matrix))],[matrix[i][1] for i in range(len(matrix))],'--y^',label = '% Infected')
        plt.plot([i for i in range(len(matrix))],[matrix[i][3] for i in range(len(matrix))],'--ro',label = '% Closed/Dead')
    elif simulationnumber == 'sim3':  
        plt.plot([i for i in range(len(matrix))],[matrix[i][0] for i in range(len(matrix))],'--go',label = '% Susceptible')
        plt.plot([i for i in range(len(matrix))],[matrix[i][1] for i in range(len(matrix))],'--y^',label = '% Infected')
        plt.plot([i for i in range(len(matrix))],[matrix[i][3] for i in range(len(matrix))],'--ro',label = '% Closed/Dead')
        #plt.plot([i for i in range(len(matrix))],[matrix[i][2] for i in range(len(matrix))],'--bs',label = '% Recovered')
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
    n =1000 # number of nodes
    pn = 0.01 # per-edge probability of existing
    i = 1 # number of nodes initially infected
    nsteps = 50 # how many time-steps to run
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
    
    """Simulation 1 - Airports cannot die"""
    
    print("Simulation 1 - Airports cannot die ")
    
    p = 0.4 # probability of acquiring infection from a single neighbour, per time-step
    rp = 0 # probability of recovery 
    td = math.inf # td time-steps after infection, the individual dies
    
#    testingPrint('sim1','real')
#    testingPrint('sim1','generated')

    "Simulation 2 - Airports can die/close"
    
    p = 0.4 # probability of acquiring infection from a single neighbour, per time-step
    rp = 0 # probability of recovery 
    td = 4 # td time-steps after infection, the individual dies

#    testingPrint('sim2','real')
    testingPrint('sim2','generated')


#    "Simulation 3 - Airports can die and recover"
#    p = 0.4 # probability of acquiring infection from a single neighbour, per time-step
#    rp = 0.2 # probability of recovery 
#    td = 4 # td time-steps after infection, the individual dies

#    testingPrint('sim3')
    

   


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
#    G_real, init_real,pinf_all_step,pdead_all_step = run(Greal,"node_deg") 
#    print(short_path_test(G_real, init_real))