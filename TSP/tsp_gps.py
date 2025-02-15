from dimod import ConstrainedQuadraticModel, Binary, cqm_to_bqm
import dimod
from neal import SimulatedAnnealingSampler
import numpy as np


def generate_gps_tsp_qubo(graph, N, lambda_penalty=10):
    """
    Generate a QUBO for TSP using the GPS formulation while keeping quadratic constraints.
    """
    model = ConstrainedQuadraticModel()
    
    # Define binary variables
    x = {
        (i, j, r): Binary(f"x_{i}_{j}_{r}")
        for i in range(N)
        for j in range(N) if i != j
        for r in range(3)
    }
    z = {()}
    
    # Objective: Minimize travel distance
    model.set_objective(sum(graph[i][j] * x[i, j, 1] for i in range(N) for j in range(N) if i != j))
    
    # Constraints
    # Constraint 1: For each (i, j), exactly one of the three possibilities (r) must be chosen.
    for i in range(N):
        for j in range(N):
            if i != j:
                model.add_constraint(sum(x[i, j, r] for r in range(3)) == 1, label=f"OneR_{i}_{j}")
    
    # Constraint 2: Each city (including depot) must be reached exactly once.
    for j in range(N):
        model.add_constraint(sum(x[i, j, 1] for i in range(N) if i != j) == 1, label=f"Reach_{j}")
    
    # Constraint 3: Each city (including depot) must be exited exactly once.
    for i in range(N):
        model.add_constraint(sum(x[i, j, 1] for j in range(N) if i != j) == 1, label=f"Exit_{i}")
    
    # Constraint 4: Ordering constraint (linear)
    for i in range(1, N):
        for j in range(1, N):
            if i != j:
                model.add_constraint(x[i, j, 2] + x[j, i, 2] == 1, label=f"Ordering_{i}_{j}")
    
    # Quadratic Constraint: Subtour elimination added as a soft penalty
    for i in range(1, N):
        for j in range(1, N):
            for k in range(1, N):
                if i != j and j != k and k != i:
                    penalty = (
                        x[j, i, 2] * x[k, j, 2] 
                        - x[j, i, 2] * x[k, i, 2] 
                        - x[k, j, 2] * x[k, i, 2] 
                        + x[k, i, 2]
                    )
                    model.set_objective(model.objective + lambda_penalty * penalty)

    # Convert CQM to BQM (now includes quadratic constraints in the objective)
    bqm, _ = cqm_to_bqm(model, lagrange_multiplier=lambda_penalty)

    return model, bqm


def solve_tsp_with_dimod(bqm, num_reads=100):
    """
    Solve the QUBO using a Simulated Annealing sampler from Dimod.
    """
    sampler = SimulatedAnnealingSampler()
    sampleset = sampler.sample(bqm, num_reads=num_reads)
    decoded_solution = sampleset.first.sample

    return sampleset, decoded_solution


def extract_tsp_path(decoded_solution, graph, N):
    """
    Extracts the optimal TSP path from the decoded binary solution and calculates the total cost.
    """
    selected_edges = [
        (int(key.split("_")[1]), int(key.split("_")[2]))
        for key, val in decoded_solution.items()
        if val == 1 and key.startswith("x_") and key.split("_")[3] == "1"
    ]
    
    if not selected_edges:
        return None, None

    path = [0]
    path_dict = {start: end for start, end in selected_edges}
    while len(path) < len(selected_edges) + 1:
        next_node = path_dict[path[-1]]
        path.append(next_node)


    if len(set(path)) != N:
        return None, None
    
    total_cost = sum([graph[path[i]][path[i+1]] for i in range(N)])
    return total_cost, path

def solve_tsp_gps(N, graph, num_reads, lambda_penalty):
    """
    Solve the TSP using the GPS formulation and the given graph.
    """
    # Generate QUBO and solve using SA
    model, bqm = generate_gps_tsp_qubo(graph, N, lambda_penalty)
    sampleset, decoded_solution = solve_tsp_with_dimod(bqm, num_reads=num_reads)
    total_cost, optimal_path = extract_tsp_path(decoded_solution, graph, N)

    return total_cost, optimal_path
  

