import numpy as np
import networkx as nx
from parse import write_input_file
import sys

input_path = "custom_inputs/"

def generate_random_complete_graph(num_vertices):
    G = nx.complete_graph(num_vertices)
    for (u,v) in G.edges():
        G.edges[u,v]['weight'] = np.around( (np.random.rand()*99) + 1, decimals=3)
    write_input_file(G, input_path + str(num_vertices) + ".in")


# Usage: python3 input_generator.py 30

if __name__ == '__main__':
    assert len(sys.argv) == 2
    num_vertices = int(sys.argv[1])
    generate_random_complete_graph(num_vertices)
    print("Generated input of size: ", num_vertices)