import networkx as nx
from parse import read_input_file, write_output_file
from utils import is_valid_solution, calculate_score
import sys
from os.path import basename, normpath
import glob


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

    s = 0
    t = G.number_of_nodes() - 1
    k = 15
    c = 1
    
    removedEdges = []
    removedVertices = []

    while (len(removedEdges) )
    
    path = nx.dijkstra_path(G, s, t)




    #https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.shortest_paths.weighted.dijkstra_path.html#networkx.algorithms.shortest_paths.weighted.dijkstra_path
    return [20], [20]
    pass

#path is the list of vertices in the shortest path in G
def removeSmallestEdge(path):

    listOfEdges = []
    [0,1,2,3,4]
    (0,1), (1, 2), (2, 3), (3, 4)
    
    min_weight = G.edges[path[0], path[1]]["weight"]
    for index in range(len(path) - 1):
        e = G.edges[i, i+ 1]
        if e['weight']
        listOfEdges.append(e)

    G.edges[1, 2]['weight']

    #while (we have not removed anything):

        # e = the minimum edge in the listofEdges

        # G.remove_edge(e)

        # if nx.has_path(G, s, t):
        #   return
        # else:
        #   G.add_edge(e)
        #   delete e from path
    return


# Here's an example of how to run your solver.

# Usage: python3 solver.py test.in

# if __name__ == '__main__':
#     assert len(sys.argv) == 2
#     path = sys.argv[1]
#     G = read_input_file(path)
#     c, k = solve(G)
#     assert is_valid_solution(G, c, k)
#     print("Shortest Path Difference: {}".format(calculate_score(G, c, k)))
#     write_output_file(G, c, k, 'outputs/small-1.out')


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
