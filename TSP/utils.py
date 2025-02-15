from pprint import pprint

from tsp_dp import solve_tsp_dp
from tsp_brute_force import solve_tsp_brute_force
from tsp_gps import solve_tsp_gps
from tsp import TSP, time_it

@time_it
def main():
    n = 7
    N = n + 1  # Number of cities
    weight_range = (1, 20)  # Range for random distances
    num_reads = 100  # Number of SA reads
    lambda_penalty = 1000
    
    # Generate TSP instance
    tsp_instance = TSP(N, weight_range)
    graph = tsp_instance.graph
    tsp_instance.show_graph()
    
    # Solve using alternative methods
    min_cost_dp, optimal_path_dp = solve_tsp_dp(N, graph)
    min_cost_bf, optimal_path_bf = solve_tsp_brute_force(N, graph)
    min_cost_gps, optimal_path_gps = solve_tsp_gps(N, graph, num_reads, lambda_penalty)


    # Print out the results
    print("\nMinimum cost (Dynamic Programming):", min_cost_dp)
    print("Optimal path (Dynamic Programming):", " -> ".join(map(str, optimal_path_dp)))
    print()
    
    print("Minimum cost (Brute Force):", min_cost_bf)
    print("Optimal path (Brute Force):", " -> ".join(map(str, optimal_path_bf)))
    print()
    
    if optimal_path_gps is None:
        print("The solution is not valid")
    else:
        print("Minimum Cost (QUBO Solution):", min_cost_gps)
        print("Optimal Path (QUBO Solution):", " -> ".join(map(str, optimal_path_gps)))
        if min_cost_gps > min_cost_bf:
            print("Not finding optimal path")

# Main execution
if __name__ == "__main__":
    main()