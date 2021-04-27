import networkx as nx
from parse import read_input_file, write_output_file
from utils import is_valid_solution, calculate_score
import sys
from os.path import basename, normpath
import glob

from collections import defaultdict
from copy import deepcopy


def solve(G):
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

    G = deepcopy(G)
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


    while len(removedVertices) <= c and canContinue:
        
        #removes k edges from G
        while len(removedEdges) <= k and canContinue:
            canContinue = removeShortestEdge(G, removedEdges, s ,t)
            if (not canContinue):
                break

        vertexToRemove = pickRemoveVertex(removedEdges, s, t)
        G.remove_node(vertexToRemove)
        removedEdges = list(filter(lambda edge: edge[0] != vertexToRemove and edge[1] != vertexToRemove, removedEdges))
        removedVertices.append(vertexToRemove)



    #https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.shortest_paths.weighted.dijkstra_path.html#networkx.algorithms.shortest_paths.weighted.dijkstra_path
    print(removedVertices)
    print(removedEdges)
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
        if nx.has_path(G, s, t):
            foundShortest = True
        else:
            print("we have found an edge we need to keep.")
            G.add_edge(shortestEdge[0], shortestEdge[1], weight=weight)
            edgePath.remove(shortestEdge)
            if len(edgePath) == 0:
                return False
            shortestEdge = edgePath[0]
            shortestEdgeWeight = G.edges[shortestEdge[0], shortestEdge[1]]['weight']

    #remove the shortestEdge
    removedEdges.append(shortestEdge)
    return True


def pickRemoveVertex(removeEdges, s, t):
    counter = defaultdict(lambda: 0)
    for edge in removeEdges:
        if edge[0] != s and edge[0] != t:
            counter[edge[0]] += 1
        if edge[1] != s and edge[1] != t:
            counter[edge[1]] += 1
    maxCount = -1
    maxVertex = 0
    for k, v in counter.items():
        if (v > maxCount):
            maxVertex = k
            maxCount = v
    return maxVertex



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
#     inputs = glob.glob('inputs/*')
#     for input_path in inputs:
#         output_path = 'outputs/' + basename(normpath(input_path))[:-3] + '.out'
#         G = read_input_file(input_path)
#         c, k = solve(G)
#         assert is_valid_solution(G, c, k)
#         distance = calculate_score(G, c, k)
#         write_output_file(G, c, k, output_path)
