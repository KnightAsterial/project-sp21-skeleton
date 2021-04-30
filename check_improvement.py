import networkx as nx
from parse import read_input_file, write_output_file, read_output_file
from utils import is_valid_solution, calculate_score
import sys
from os.path import basename, normpath
import glob

# Here's an example of how to run your solver.

# Usage: python3 check_improvement.py test.in test.out

if __name__ == '__main__':
    assert len(sys.argv) == 3
    path = sys.argv[1]
    outPath = sys.argv[2]
    G = read_input_file(path)
    diff = read_output_file(G, outPath)
    print("Shortest Path Difference: {}".format(diff))


