import networkx as nx
from parse import read_input_file, write_output_file
from utils import is_valid_solution, calculate_score
import sys
from os.path import basename, normpath
import glob

from collections import defaultdict
from copy import deepcopy

from mip import *

def solve(G):
    H = G
    G = H.copy()

    numNodes = G.number_of_nodes();
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
    
    removedEdges = []
    removedVertices = []

    # ILP reduction
        # 0) Make G directed (for every edge, make two directed edges)
        # 1) VARIABLES
        #   X_(u,v) = {0,1} if edge (u,v) is in shortest path
        #   Y_(u,v) = {0,1} if edge (u,v) is removed from graph
        #   Z_v = {0,1} if vertex j is removed
        
        # 2) Constaints
        #   For every vertex v \ {s,t}: sum of X_(u,v) - sum of X_(v,w) = 0
        #   For vertex s                sum of X_(u,s) - sum of X_(s,w) = -1
        #   For vertex t                sum of X_(u,t) - sum of X_(t,w) = 1
        #   For every edge (u,v):   Y_(u,v) = Y_(v,u)
        #                           X_(u,v) + Y_(u,v) + Z_u + Z_v <= 1
        #                       
        #                           sum of Y_(u,v) <= 2k
        #                           sum of Z_v <= c
        # 3) Objective function
        #   minimize sum of w_(u,v) * X_(u,v) - 1000000*(sum of Y_(u,v) + sum of Z_v)
    
    shortestPathLength = nx.shortest_path_length(G, source=s, target=t, weight='weight')

    # create the model
    m = Model(sense=MAXIMIZE)

    #define variables
    X = {}
    Y = {}
    Z = [m.add_var(var_type=BINARY) for i in range(numNodes)]
    
    for edge in G.edges:
        u = edge[0]
        v = edge[1]
        X[(u,v)] = m.add_var(var_type=BINARY)
        X[(v,u)] = m.add_var(var_type=BINARY)
        
        Y[(u,v)] = m.add_var(var_type=BINARY)
        Y[(v,u)] = m.add_var(var_type=BINARY)

    for v in range(s+1, t):
        m += xsum(X[(u,v)] for u in G.neighbors(v)) - xsum(X[(v,w)] for w in G.neighbors(v)) == 0   # For every vertex v \ {s,t}: sum of X_(u,v) - sum of X_(v,w) = 0
    m += xsum(X[(u,s)] for u in G.neighbors(s)) - xsum(X[(s,u)] for w in G.neighbors(s)) == -1         # For vertex s                sum of X_(u,s) - sum of X_(s,w) = -1
    m += xsum(X[(u,t)] for u in G.neighbors(t)) - xsum(X[(t,w)] for w in G.neighbors(t)) == 1          # For vertex t                sum of X_(u,t) - sum of X_(t,w) = 1



    # For every edge (u,v):   Y_(u,v) = Y_(v,u)
    #                         X_(u,v) + Y_(u,v) + Z_u + Z_v <= 1
    for edge in G.edges:
        u = edge[0]
        v = edge[1]

        m += Y[(u,v)] - Y[(v,u)] == 0
        m += X[(u,v)] + Y[(u,v)] + Z[u] + Z[v] <= 1

    # sum of Y_(u,v) <= 2k
    # sum of Z_v <= c
    m += xsum(y for y in Y.values()) <= 2*k
    m += xsum(z for z in Z) <= c

    m.objective = xsum(G.edges[u,v]['weight'] * X[(u,v)] for u,v in X.keys()) - shortestPathLength

    m.max_gap = 0.05
    status = m.optimize(max_seconds=300)
    if status == OptimizationStatus.OPTIMAL:
        print('optimal solution cost {} found'.format(m.objective_value))
    elif status == OptimizationStatus.FEASIBLE:
        print('sol.cost {} found, best possible: {}'.format(m.objective_value, m.objective_bound))
    elif status == OptimizationStatus.NO_SOLUTION_FOUND:
        print('no feasible solution found, lower bound is: {}'.format(m.objective_bound))
    if status == OptimizationStatus.OPTIMAL or status == OptimizationStatus.FEASIBLE:
        for edge in G.edges:
            u = edge[0]
            v = edge[1]
            if Y[(u,v)].x >= 0.99:
                removedEdges.append(edge)
        for v in G.nodes:
            if Z[v].x >= 0.99:
                removedVertices.append(v)
    else:
        return baselineSolve(G)

    print('OriginalShortestPath:', shortestPathLength)
    print(removedVertices)
    print(removedEdges)
    return removedVertices, removedEdges



def baselineSolve(G):
    """
    Args:
        G: networkx.Graph
    Returns:
        c: list of cities to remove
        k: list of edges to remove
    """

    #baseline alg:
        # 1) Run Dijkstra's to find shortest path
        # 2) Delete the smallest (edge? or vertex?) that you can delete from the graph
        # 3) Repeat steps 1-2 until you cannot delete things anymore b/c:
        #        a) it would disconnect s-t
        #        b) it would violate a constraint

    #kinda fancy algorithm
        # 1) Run Dijkstras to find shortest path, add path to PATH_LIST
        # 2) Find minimum set of REMOVE_EDGES such at at least every path in PATH_LIST has at least one path from REMOVE_EDGES
        # 3) repeat until |REMOVE_EDGES| == k
        # 4) pick vertex that shortens size of REMOVE_EDGES
        # 5) keep repeating






    H = G
    G = H.copy()
    numNodes = G.number_of_nodes();

    s = 0
    t = numNodes - 1

    #correctly identify the size of the problem
    k = 15
    c = 1
    if (numNodes > 30):
        k = 50
        c = 3
    if (numNodes > 50):
        k = 100
        c = 5
    
    removedEdges = []
    removedVertices = []

    canContinue = True


    while len(removedVertices) < c and canContinue:
        
        #removes k edges from G
        while len(removedEdges) < k and canContinue:
            canContinue = removeShortestEdge(G, removedEdges, s ,t)
            if (not canContinue):
                break

        vertexToRemove, canRemoveVertex = pickRemoveVertex(G, removedEdges, s, t)
        if (not canRemoveVertex):
            canContinue = False
            break
        G.remove_node(vertexToRemove)
        removedEdges = list(filter(lambda edge: edge[0] != vertexToRemove and edge[1] != vertexToRemove, removedEdges))
        removedVertices.append(vertexToRemove)


    return removedVertices, removedEdges


#removes the shortest edge from the shortest path in G
def removeShortestEdge(G, removedEdges, s, t):
    
    nodePath = nx.dijkstra_path(G, s, t)
    edgePath = [(nodePath[i], nodePath[i+1]) for i in range(len(nodePath) - 1)]
    
    shortestEdge = (nodePath[0], nodePath[1])
    shortestEdgeWeight = G.edges[nodePath[0], nodePath[1]]['weight']

    #iterates through all edges in the path, and finds the shortest edge
    foundShortest = False

    while not foundShortest:
        for i in range(len(edgePath)):
            weight = G.edges[edgePath[i][0], edgePath[i][1]]['weight']
            if weight < shortestEdgeWeight:
                    shortestEdgeWeight = weight
                    shortestEdge = edgePath[i]

        G.remove_edge(shortestEdge[0], shortestEdge[1])
        if nx.is_connected(G):
            foundShortest = True
        else:
            G.add_edge(shortestEdge[0], shortestEdge[1], weight=weight)
            edgePath.remove(shortestEdge)
            if len(edgePath) == 0:
                return False
            shortestEdge = edgePath[0]
            shortestEdgeWeight = G.edges[shortestEdge[0], shortestEdge[1]]['weight']

    #remove the shortestEdge
    removedEdges.append(shortestEdge)
    return True


#we need to add connected check on THIS!!
def pickRemoveVertex(G, removeEdges, s, t):
    counter = defaultdict(lambda: 0)
    for edge in removeEdges:
        if edge[0] != s and edge[0] != t:
            counter[edge[0]] += 1
        if edge[1] != s and edge[1] != t:
            counter[edge[1]] += 1

    while len(counter) > 0:
        maxCount = -1
        maxVertex = 0
        for k, v in counter.items():
            if (v > maxCount):
                maxVertex = k
                maxCount = v

        GCopy = G.copy()
        GCopy.remove_node(maxVertex)

        if nx.is_connected(GCopy):
            return maxVertex, True
        else:
            del counter[maxVertex]

    return 0, False




# Here's an example of how to run your solver.

# Usage: python3 solver.py test.in

if __name__ == '__main__':
    assert len(sys.argv) == 2
    path = sys.argv[1]
    G = read_input_file(path)
    c, k = solve(G)
    assert is_valid_solution(G, c, k)
    print("Shortest Path Difference: {}".format(calculate_score(G, c, k)))
    write_output_file(G, c, k, 'custom_outputs/small-1.out')


# For testing a folder of inputs to create a folder of outputs, you can use glob (need to import it)
# if __name__ == '__main__':
#     counter = 0
#     inputs = glob.glob('inputs/large/*.in')
#     for input_path in inputs:
#         counter += 1
#         if (counter % 50 == 0):
#             print("Running file #", counter)
#         output_path = 'outputs/large/' + basename(normpath(input_path))[:-3] + '.out'
#         G = read_input_file(input_path)
#         c, k = solve(G)
#         assert is_valid_solution(G, c, k)
#         distance = calculate_score(G, c, k)
#         write_output_file(G, c, k, output_path)
    
#     inputs = glob.glob('inputs/medium/*.in')
#     for input_path in inputs:
#         counter += 1
#         if (counter % 50 == 0):
#             print("Running file #", counter)
#         output_path = 'outputs/medium/' + basename(normpath(input_path))[:-3] + '.out'
#         G = read_input_file(input_path)
#         c, k = solve(G)
#         assert is_valid_solution(G, c, k)
#         distance = calculate_score(G, c, k)
#         write_output_file(G, c, k, output_path)

#     inputs = glob.glob('inputs/small/*.in')
#     for input_path in inputs:
#         counter += 1
#         if (counter % 50 == 0):
#             print("Running file #", counter)
#         output_path = 'outputs/small/' + basename(normpath(input_path))[:-3] + '.out'
#         G = read_input_file(input_path)
#         c, k = solve(G)
#         assert is_valid_solution(G, c, k)
#         distance = calculate_score(G, c, k)
#         write_output_file(G, c, k, output_path)
