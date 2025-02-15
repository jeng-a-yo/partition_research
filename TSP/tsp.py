import random
import numpy as np
import pandas as pd
import time
from functools import wraps


class TSP:
    def __init__(self, n, weight_range=(1, 100)):
        """
        Initialize the TSP problem with n cities.
        
        Args:
            n (int): Number of cities.
            weight_range (tuple): Range of edge weights (default: 1 to 100).
        """
        self.n = n
        self.graph = self.generate_graph(n, weight_range)
    
    @staticmethod
    def generate_graph(n, weight_range):
        """
        Generate a random weighted adjacency matrix for the TSP graph.

        Args:
            n (int): Number of nodes (cities).
            weight_range (tuple): Range of weights (min, max).

        Returns:
            np.array: Symmetric adjacency matrix.
        """
        rng = np.random.default_rng()
        graph = rng.integers(weight_range[0], weight_range[1] + 1, size=(n, n))
        np.fill_diagonal(graph, 0)
        return (graph + graph.T) // 2
    

    def show_graph(self):
        """
        Display the TSP graph as a well-formatted table.
        """
        df = pd.DataFrame(self.graph, index=[f"C{i}" for i in range(self.n)], columns=[f"C{i}" for i in range(self.n)])
        print("TSP Distance Matrix:")
        print(df)


def time_it(func):
    """A decorator to measure the execution time of a function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Start time
        result = func(*args, **kwargs)  # Execute the function
        end_time = time.time()  # End time
        elapsed_time = end_time - start_time  # Calculate elapsed time
        print(f"Function '{func.__name__}' took {elapsed_time:.6f} seconds to execute.")
        return result
    return wrapper