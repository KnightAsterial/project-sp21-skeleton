import networkx as nx
from parse import read_input_file, write_output_file, read_output_file
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

def solve(G):
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

    removedVertices = []
    removedEdges = []
    
    for _ in range(k):
        bestEdge = (-1, -1)
        bestPathLength = nx.dijkstra_path_length(G, s, t)
        bestPathNodes = nx.dijkstra_path(G, s, t)
        bestPathEdges = [(bestPathNodes[i], bestPathNodes[i+1]) for i in range(len(bestPathNodes) - 1)]
        for i in range(len(bestPathEdges)):
            currEdge = bestPathEdges[i]
            tempG = G.copy()
            tempG.remove_edge(currEdge[0], currEdge[1])
            if not nx.is_connected(tempG):
                continue
            pathLen = nx.dijkstra_path_length(G, s, t)
            if pathLen >= bestPathLength:
                bestPathLength = pathLen
                bestEdge = currEdge
        if bestEdge != (-1, -1):
            removedEdges.append(bestEdge)
            G.remove_edge(bestEdge[0], bestEdge[1])

    for _ in range(c):
        bestNode = -1
        bestPathLength = nx.dijkstra_path_length(G, s, t)
        for node in G.nodes:
            if node == s or node == t:
                continue
            tempG = G.copy()
            tempG.remove_node(node)
            if not nx.is_connected(tempG):
                continue
            pathLen = nx.dijkstra_path_length(G, s, t)
            if pathLen >= bestPathLength:
                bestPathLength = pathLen
                bestNode = node
        if bestNode != -1:
            removedVertices.append(bestNode)
            G.remove_node(bestNode)
    
    return removedVertices, removedEdges

    





# Here's an example of how to run your solver.

# Usage: python3 test.py test.in prevresults.out

# if __name__ == '__main__':
#     assert len(sys.argv) == 2
#     path = sys.argv[1]
#     G = read_input_file(path)
#     c, k = solve(G)
#     assert is_valid_solution(G, c, k)
#     print("Shortest Path Difference: {}".format(calculate_score(G, c, k)))
#     write_output_file(G, c, k, 'custom_outputs/small-1.out')


# For testing a folder of inputs to create a folder of outputs, you can use glob (need to import it)
if __name__ == '__main__':
    counter = -1
    
    inputs = glob.glob('inputs/small/*.in')
    for input_path in inputs:
        counter += 1
        if (counter % 50 == 0):
            print("Running file #", counter+1)
        output_path = 'outputs/small/' + basename(normpath(input_path))[:-3] + '.out'
        G = read_input_file(input_path)
        c, k = solve(G)
        assert is_valid_solution(G, c, k)
        
        writeSolution = False
        distance = calculate_score(G, c, k)
        # print(output_path, "score:", distance)
        try:
            prevBestScore = read_output_file(G, output_path)
            if distance > prevBestScore:
                writeSolution = True
            # else:
                # print("Current solution is worse. Not overwriting.")
        except Exception as e:
            print(e)
            print("Previous solution didn't exist.")
            writeSolution = True
        if writeSolution:
            print(output_path, "score:", distance)
            print("Writing...")
            write_output_file(G, c, k, output_path)


    inputs = glob.glob('inputs/medium/*.in')
    for input_path in inputs:
        counter += 1
        if (counter % 50 == 0):
            print("Running file #", counter+1)
        output_path = 'outputs/medium/' + basename(normpath(input_path))[:-3] + '.out'
        G = read_input_file(input_path)
        c, k = solve(G)
        assert is_valid_solution(G, c, k)
        
        writeSolution = False
        distance = calculate_score(G, c, k)
        # print(output_path, "score:", distance)
        try:
            prevBestScore = read_output_file(G, output_path)
            if distance > prevBestScore:
                writeSolution = True
            # else:
                # print("Current solution is worse. Not overwriting.")
        except Exception as e:
            print(e)
            print("Previous solution didn't exist.")
            writeSolution = True
        if writeSolution:
            print(output_path, "score:", distance)
            print("Writing...")
            write_output_file(G, c, k, output_path)
    


    inputs = glob.glob('inputs/large/*.in')
    for input_path in inputs:
        counter += 1
        if (counter % 50 == 0):
            print("Running file #", counter+1)
        output_path = 'outputs/large/' + basename(normpath(input_path))[:-3] + '.out'
        G = read_input_file(input_path)
        c, k = solve(G)
        assert is_valid_solution(G, c, k)
        
        writeSolution = False
        distance = calculate_score(G, c, k)
        # print(output_path, "score:", distance)
        try:
            prevBestScore = read_output_file(G, output_path)
            if distance > prevBestScore:
                writeSolution = True
            # else:
                # print("Current solution is worse. Not overwriting.")
        except Exception as e:
            print(e)
            print("Previous solution didn't exist.")
            writeSolution = True
        if writeSolution:
            print(output_path, "score:", distance)
            print("Writing...")
            write_output_file(G, c, k, output_path)


