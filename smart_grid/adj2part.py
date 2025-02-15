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
    
    # Binary variables: x_{i,k} = 1 if node i is in community k
    X = np.array([[Binary(f"x_{i}_{k}") for k in range(K)] for i in range(n)])

    # Sum of adjacency entries = 2m for an undirected graph
    s = np.sum(adj_matrix)
    m = s / 2.0  # total number of edges (float)

    # Build the modularity matrix B, then scale by 1/(2m)
    Q = np.zeros((n, n))
    for i in range(n):
        k_i = np.sum(adj_matrix[i])  # degree of node i
        for j in range(n):
            k_j = np.sum(adj_matrix[j])  # degree of node j
            # B_{ij} = A_{ij} - (k_i * k_j) / (2m)
            Q[i][j] = adj_matrix[i][j] - (k_i * k_j) / (2.0 * m)

    # Scale by 1/(2m)
    Q /= (2.0 * m)

    # Define the objective:
    # We want to maximize sum_{i,j} Q[i,j] * δ(g_i, g_j).
    # The solver *minimizes* the objective, so we multiply by -1.
    M = np.matmul(np.matmul(X.T, Q), X)
    model.set_objective(-1.0 * np.trace(M))

    # Constraints:
    # 1) Each node i must be in exactly one community
    for i in range(n):
        model.add_constraint(sum(X[i, k] for k in range(K)) == 1, label=f"one_hot_{i}")

    # 2) Each community must have at least one node
    for k in range(K):
        model.add_constraint(sum(X[i, k] for i in range(n)) >= 1, label=f"non_empty_{k}")

    # Convert to a Binary Quadratic Model
    bqm, _ = cqm_to_bqm(model)
    return model, bqm, Q

def solve_qubo(bqm, num_reads=100):
    sampler = SimulatedAnnealingSampler()
    sampleset = sampler.sample(bqm, num_reads=num_reads)
    decode_solution = sampleset.first.sample
    return sampleset, decode_solution

def compute_modularity(decoded_solution, n, K, Q):
    """Compute the modularity from the decoded solution and Q matrix.

    Parameters
    ----------
    decoded_solution : dict
        A dictionary mapping variable name (e.g. "x_0_1") to its assigned binary value (0 or 1).
    n : int
        Number of nodes in the graph.
    K : int
        Number of communities.
    Q : 2D numpy array
        The modularity matrix scaled by 1/(2m). Specifically, Q[i][j] = B[i][j]/(2m),
        where B = A - (k k^T)/(2m).

    Returns
    -------
    float
        The (standard) modularity of the partition given by decoded_solution.
    """

    # Extract each node's community
    # Because each node is constrained to exactly one community,
    # we can simply find the k for which x_{i,k} = 1.
    membership = [-1]*n
    for var, val in decoded_solution.items():
        if val == 1 and var.startswith("x_"):
            i_str, k_str = var.split("_")[1:]  # var like "x_i_k"
            i, k = int(i_str), int(k_str)
            membership[i] = k

    # Sum up Q[i][j] if i, j are in the same community
    modularity = 0.0
    for i in range(n):
        for j in range(n):
            if membership[i] == membership[j]:
                modularity += Q[i][j]

    return modularity


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
    best_modularity = -1
    best_partitioning = []
    k_mod_dict = dict()
    for loop_k in range(1, K+1):
        model, bqm, Q = define_qubo_problem(adj_matrix, n, loop_k)
        sampleset, decoded_solution = solve_qubo(bqm)
        partitioning, modularity = extract_partitioning(decoded_solution, n, loop_k, Q)
        k_mod_dict[loop_k] = modularity
        if modularity > best_modularity:
            best_modularity = modularity
            best_partitioning = partitioning

    return best_modularity, best_partitioning, k_mod_dict


