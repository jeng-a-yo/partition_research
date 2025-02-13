import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pprint import pprint
from dimod import ConstrainedQuadraticModel, Binary, cqm_to_bqm
from neal import SimulatedAnnealingSampler


def kronecker_delta(i, j):
    return 1 if i == j else 0

def define_qubo_problem(adj_matrix, n, K):
    model = ConstrainedQuadraticModel()
    
    # Binary variables
    X = np.array([[Binary(f"x_{i}_{k}") for k in range(K)] for i in range(n)])

    # Get the summation of the adj_matrix
    s = np.sum(adj_matrix)

    # Define m
    m = s / 2  # Division should be float-safe

    # Define Q matrix
    Q = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            k_i = np.sum(adj_matrix[i])  # Sum of row i
            k_j = np.sum(adj_matrix[j])  # Sum of row j
            Q[i][j] = (1 / (2 * m)) * (s - (k_i * k_j) / (2 * m)) * kronecker_delta(i, j)

    # Define the modularity function M
    M = (-1) * np.matmul(X.T, Q, X)

    # Set the objective
    model.set_objective(np.trace(M))

    # Constraints:
    for i in range(n):
        model.add_constraint(sum(X[i, k] for k in range(K)) == 1, label=f"one_hot_{i}")
    
    for k in range(K):
        model.add_constraint(sum(X[i, k] for i in range(n)) >= 1, label=f"non_empty_{k}")
    
    # Convert to Binary Quadratic Model (BQM)
    bqm, _ = cqm_to_bqm(model)
    return model, bqm, Q  # Return Q for modularity computation

def solve_qubo(bqm, num_reads=100):
    sampler = SimulatedAnnealingSampler()
    sampleset = sampler.sample(bqm, num_reads=num_reads)
    decode_solution = sampleset.first.sample
    return sampleset, decode_solution

def compute_modularity(decoded_solution, n, K, Q):
    """ Compute modularity from the decoded solution and Q matrix. """
    modularity = 0
    for i in range(n):
        for j in range(n):
            for k in range(K):
                if decoded_solution[f"x_{i}_{k}"] == 1 and decoded_solution[f"x_{j}_{k}"] == 1:
                    modularity += Q[i][j]

    return -modularity  # Since M was negative in define_qubo_problem()

def extract_partitioning(decoded_solution, n, K, Q):  
    """ Extract the partitioning and compute modularity. """
    partitioning = {k: [] for k in range(K)}

    for key, value in decoded_solution.items():
        if value == 1 and key.startswith("x_"):  # Ignore slack variables
            i, k = map(int, key.split("_")[1:])
            partitioning[k].append(i)

    # Compute modularity from decoded solution
    modularity = compute_modularity(decoded_solution, n, K, Q)

    return partitioning, modularity

def adj2part(adj_matrix, n, K):

    model, bqm, Q = define_qubo_problem(adj_matrix, n, K)
    sampleset, decoded_solution = solve_qubo(bqm)

    partitioning, modularity = extract_partitioning(decoded_solution, n, K, Q)
    return partitioning, modularity

if __name__ == "__main__":
    n, K = 10, 3
    adj_matrix = generate_graph(n)

    print("Adjacency Matrix:")
    pprint(adj_matrix)
    adj2part(adj_matrix)
