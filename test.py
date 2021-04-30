import networkx as nx
from parse import read_input_file, write_output_file, get_config_from_output
from utils import is_valid_solution, calculate_score
import sys
from os.path import basename, normpath
import glob

from collections import defaultdict
from copy import deepcopy

from mip import *

from pysat.examples.hitman import Hitman

from itertools import islice
import time
import random
import math

def solve(G, prevVertices, prevEdges):
    start = time.time()

    originalGraph = G
    G = originalGraph.copy()

    numNodes = G.number_of_nodes()
    s = 0
    t = numNodes - 1

    # correctly identify the size of the problem
    k = 15
    c = 1
    if (numNodes > 30):
        k = 50
        c = 3
    if (numNodes > 50):
        k = 100
        c = 5
    
    bestSeenEdges = prevEdges
    bestSeenVertices = prevVertices
    removedEdges = prevEdges
    removedVertices = prevVertices
    G.remove_edges_from(removedEdges)
    G.remove_nodes_from(removedVertices)
    tempG = G.copy()

    shortestPathLength = nx.dijkstra_path_length(G, s, t)
    print(shortestPathLength)

    p = 0.5
    temperature = 5
    iterations = 5000

    for T in range(temperature):
        for i in range(iterations):
            # at start tempG == G
            action = pickAction(len(removedEdges), len(removedVertices), k, c)
            if (i % 100 == 0):
                print(nx.is_connected(G), end="", flush=True)
            if action == 'e':
                edge = random.choice(removedEdges)
                randEndpoint = edge[random.choice([0,1])]
                newEdge = random.choice([(randEndpoint, neighbor) for neighbor in G.neighbors(randEndpoint)])
                tempG.add_edge(edge[0], edge[1], weight=originalGraph.edges[edge[0], edge[1]]['weight'])
                tempG.remove_edge(newEdge[0], newEdge[1])
                if (not nx.is_connected(tempG)):
                    # reset tempG to G
                    tempG.remove_edge(edge[0], edge[1])
                    tempG.add_edge(newEdge[0], newEdge[1], weight=originalGraph.edges[newEdge[0], newEdge[1]]['weight'])
                else:
                    Gscore = score(G, s, t)
                    tempGscore = score(tempG, s, t)
                    if tempGscore > Gscore or probCheck(Gscore, tempGscore, shortestPathLength, p, T):
                        # set G to tempG
                        G.add_edge(edge[0], edge[1], weight=originalGraph.edges[edge[0], edge[1]]['weight'])
                        G.remove_edge(newEdge[0], newEdge[1])
                        removedEdges.remove(edge)
                        removedEdges.append(newEdge)
                    else: 
                        # reset tempG to G
                        tempG.remove_edge(edge[0], edge[1])
                        tempG.add_edge(newEdge[0], newEdge[1], weight=originalGraph.edges[newEdge[0], newEdge[1]]['weight'])
            elif action == 'v':
                vertex = random.choice(removedVertices)

                nodesList = list(G.nodes)
                nodesList.remove(s)
                nodesList.remove(t)
                randVertex = random.choice(nodesList)

                newVertices = removedVertices.copy()
                newVertices.remove(vertex)
                newVertices.append(randVertex)
                
                tempG = originalGraph.copy()
                tempG.remove_edges_from(removedEdges)
                tempG.remove_nodes_from(newVertices)

                if (not nx.is_connected(tempG)):
                    # reset
                    tempG = G.copy()
                else:
                    Gscore = score(G, s, t)
                    tempGscore = score(tempG, s, t)
                    if tempGscore > Gscore or probCheck(Gscore, tempGscore, shortestPathLength, p, T):
                        G = tempG.copy()
                        removedVertices.remove(vertex)
                        removedVertices.append(randVertex)
                        removedEdges = list(filter(lambda edge: edge[0] != randVertex and edge[1] != randVertex, removedEdges))
                        
                    else:
                        # reset
                        tempG = G.copy()
            elif action == 'ae':
                added = False
                for _ in range(100):
                    edge = random.choice(list(G.edges))
                    tempG.remove_edge(edge[0], edge[1])
                    if (not nx.is_connected(tempG)):
                        tempG.add_edge(edge[0], edge[1], weight=originalGraph.edges[edge[0], edge[1]]['weight'])
                    else: 
                        Gscore = score(G, s, t)
                        tempGscore = score(tempG, s, t)
                        if tempGscore > Gscore or probCheck(Gscore, tempGscore, shortestPathLength, p, T):
                            G.remove_edge(edge[0], edge[1])
                            removedEdges.append(edge)
                            added = True
                            break
                        tempG.add_edge(edge[0], edge[1], weight=originalGraph.edges[edge[0], edge[1]]['weight'])
            elif action == 'av':
                added = False
                for _ in range(100):
                    newVertex = random.choice(list(G.nodes))
                    if newVertex == s or newVertex == t:
                        continue
                    tempG.remove_node(newVertex)
                    if (not nx.is_connected(tempG)):
                        tempG = G.copy()
                    else:
                        Gscore = score(G, s, t)
                        tempGscore = score(tempG, s, t)
                        if tempGscore > Gscore or probCheck(Gscore, tempGscore, shortestPathLength, p, T):
                            G.remove_node(newVertex)
                            removedVertices.append(newVertex)
                            removedEdges = list(filter(lambda edge: edge[0] != newVertex and edge[1] != newVertex, removedEdges))
                            added=True
                            break
                        tempG = G.copy()
            elif action == 're':
                edge = random.choice(removedEdges)
                tempG.add_edge(edge[0], edge[1], weight=originalGraph.edges[edge[0], edge[1]]['weight'])
                if (not nx.is_connected(tempG)):
                    tempG.remove_edge(edge[0], edge[1])
                else: 
                    Gscore = score(G, s, t)
                    tempGscore = score(tempG, s, t)
                    if tempGscore > Gscore or probCheck(Gscore, tempGscore, shortestPathLength, p, T):
                        G.add_edge(edge[0], edge[1], weight=originalGraph.edges[edge[0], edge[1]]['weight'])
                        removedEdges.remove(edge)
                    else:
                        tempG.remove_edge(edge[0], edge[1])
            elif action == 'rv':
                vertex = random.choice(removedVertices)
                baseVertexList = removedVertices.copy()
                baseVertexList.remove(vertex)
                tempG = originalGraph.copy()
                tempG.remove_edges_from(removedEdges)
                tempG.remove_nodes_from(baseVertexList)
                if (not nx.is_connected(tempG)):
                    tempG = G.copy()
                else: 
                    Gscore = score(G, s, t)
                    tempGscore = score(tempG, s, t)
                    if tempGscore > Gscore or probCheck(Gscore, tempGscore, shortestPathLength, p, T):
                        G = tempG.copy()
                        removedVertices.remove(vertex)
                    else:
                        tempG = G.copy()

            currScore = score(G, s, t)
            if currScore > shortestPathLength:
                print("----- setting new high score -----", currScore, "<- new, prev->", shortestPathLength)
                bestSeenEdges = removedEdges.copy()
                bestSeenVertices = removedVertices.copy()
                shortestPathLength = currScore

    print(bestSeenEdges)
    print(bestSeenVertices)
    G = originalGraph.copy()
    G.remove_edges_from(bestSeenEdges)
    G.remove_nodes_from(bestSeenVertices)
    print(nx.is_connected(G))
    return bestSeenVertices, bestSeenEdges

                        
                
                



def score(G, s, t):
    return nx.dijkstra_path_length(G, s, t)


def probCheck(Gscore, tempGscore, maxScore, p, t):
    base = math.pow(p, t)
    threshold = math.pow(base, (Gscore - tempGscore) / maxScore)
    if (random.random() < threshold):
        return True
    return False



            
# e, v, ae, av, re, rv
def pickAction(numEdges, numVertices, k, c):
    return random.choices(['e', 'v', 'ae', 'av', 're', 'rv'], weights=[numEdges, numVertices, (1 if numEdges < k else 0), (1 if numVertices < c else 0), (1 if numEdges > 0 else 0), (1 if numVertices > 0 else 0)])[0]
    








# Here's an example of how to run your solver.

# Usage: python3 test.py test.in prevresults.out

if __name__ == '__main__':
    assert len(sys.argv) == 3
    path = sys.argv[1]
    outPath = sys.argv[2]
    G = read_input_file(path)
    prevC, prevK = get_config_from_output(outPath)
    c, k = solve(G, prevC, prevK)
    write_output_file(G, c, k, 'custom_outputs/small-1.out')
    assert is_valid_solution(G, c, k)
    print("Shortest Path Difference: {}".format(calculate_score(G, c, k)))
    # write_output_file(G, c, k, 'custom_outputs/small-1.out')


# For testing a folder of inputs to create a folder of outputs, you can use glob (need to import it)
# if __name__ == '__main__':
#     counter = -1
#     inputs = glob.glob('inputs/large/*.in')
#     for input_path in inputs:
#         counter += 1
#         if (counter % 50 == 0):
#             print("Running file #", counter)
#         output_path = 'outputs/large/' + basename(normpath(input_path))[:-3] + '.out'
#         G = read_input_file(input_path)
#         c, k = solve(G)
#         try:
#             assert is_valid_solution(G, c, k)
#             distance = calculate_score(G, c, k)
#         except Exception as exception:
#             c, k = baselineSolve(G)
#         distance = calculate_score(G, c, k)
#         write_output_file(G, c, k, output_path)
    
#     # inputs = glob.glob('inputs/small/*.in')
#     # for input_path in inputs:
#     #     counter += 1
#     #     if (counter % 50 == 0):
#     #         print("Running file #", counter+1)
#     #     output_path = 'outputs/small/' + basename(normpath(input_path))[:-3] + '.out'
#     #     G = read_input_file(input_path)
#     #     c, k = solve(G)
#     #     try:
#     #         assert is_valid_solution(G, c, k)
#     #         distance = calculate_score(G, c, k)
#     #     except Exception as exception:
#     #         c, k = baselineSolve(G)
#     #     write_output_file(G, c, k, output_path)

#     # inputs = glob.glob('inputs/medium/*.in')
#     # for input_path in inputs:
#     #     counter += 1
#     #     if (counter % 50 == 0):
#     #         print("Running file #", counter)
#     #     output_path = 'outputs/medium/' + basename(normpath(input_path))[:-3] + '.out'
#     #     G = read_input_file(input_path)
#     #     c, k = solve(G)
#     #     try:
#     #         assert is_valid_solution(G, c, k)
#     #         distance = calculate_score(G, c, k)
#     #     except Exception as exception:
#     #         c, k = baselineSolve(G)
#     #     write_output_file(G, c, k, output_path)


