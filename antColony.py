import numpy as np
import matplotlib.pyplot as plt

# Generate distances between cities
def generate_distances(num_cities):
    distances = np.random.randint(3, 51, size=(num_cities, num_cities)) # Random distances between 3 and 50
    np.fill_diagonal(distances, 0)  # Ensure zero distance from a city to itself
    return distances

# Generate distances for 10 cities and 20 cities
distances_10 = generate_distances(10)
distances_20 = generate_distances(20)

# Graph distance between cities
def plot_distance_table(distances):
    num_cities = len(distances)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title('Distance between Cities')
    ax.imshow(distances, cmap='viridis')

    # Add text annotations
    for i in range(num_cities):
        for j in range(num_cities):
            ax.text(j, i, distances[i, j], ha='center', va='center', color='w')

    ax.set_xticks(np.arange(num_cities))
    ax.set_yticks(np.arange(num_cities))
    ax.set_xticklabels(np.arange(1, num_cities + 1))
    ax.set_yticklabels(np.arange(1, num_cities + 1))
    ax.set_xlabel('To City')
    ax.set_ylabel('From City')

    plt.colorbar(ax.imshow(distances, cmap='viridis'), ax=ax, label='Distance')
    plt.tight_layout()
    plt.show()

# Print and plot distances for 10 cities
print("Distances for 10 cities:\n", distances_10)
plot_distance_table(distances_10)

# Print and plot distances for 20 cities
print("\nDistances for 20 cities:\n", distances_20)
plot_distance_table(distances_20)

# Ant colony optimization algorithm
def ant_colony_optimization(distances, num_ants, num_iterations):
    num_cities = len(distances)
    pheromones = np.ones((num_cities, num_cities))  # Initial pheromone levels
    best_path = None
    best_distance = float('inf')

    for iteration in range(num_iterations):
        # Ants construct solutions
        for ant in range(num_ants):
            visited = [0]  # Start at city 0
            while len(visited) < num_cities:
                current_city = visited[-1]
                unvisited_cities = [city for city in range(num_cities) if city not in visited]
                probabilities = [pheromones[current_city, city] ** 2 / distances[current_city, city]
                                 for city in unvisited_cities] #probability = (pheromone level)**2 /  distance
                probabilities /= np.sum(probabilities) # Normalize probabilities
                next_city = np.random.choice(unvisited_cities, p=probabilities)
                visited.append(next_city)

            # Complete the tour
            visited.append(0)  # Return to the start city

            # Update best solution if needed
            distance = sum(distances[visited[i], visited[i+1]] for i in range(num_cities))
            if distance < best_distance:
                best_distance = distance
                best_path = visited.copy()

        # Update pheromones
        pheromones *= 0.9  # Evaporation
        for i in range(num_cities):
            for j in range(num_cities):
                if i != j:
                    pheromones[i, j] += 1 / best_distance  # Deposit pheromone inversely proportional to distance

        if iteration % 10 == 0:
            print(f"Iteration {iteration}: Best distance = {best_distance}")

    return best_path, best_distance

# Graph the best path and best distance
def plot_best_path_and_best_distance(distances, best_path, best_distance):
    num_cities = len(distances)
    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot best path
    ax.set_title('Best Path')
    ax.plot(best_path, 'r.-', markersize=10)

    # Add costs between cities
    for i in range(num_cities):
        start_city = best_path[i]
        end_city = best_path[i + 1]
        cost = int(distances[start_city, end_city])
        ax.text(i + 0.5, (start_city + end_city) / 2, f'{cost}', ha='center', va='center', fontsize=14, fontweight='bold')

    # Set ticks and labels
    ax.set_xticks(np.arange(len(best_path)))
    ax.set_xticklabels([f'{city + 1}' for city in best_path])
    ax.set_ylabel('Cost')
    ax.set_xlabel('Visit Order')
    ax.grid(True)

    # Write best distance
    ax.text(0.5, 0.9, f'Best Distance: {int(best_distance)}', transform=ax.transAxes, fontsize=12,
             horizontalalignment='center', verticalalignment='center', fontweight='bold', color='blue')

    plt.tight_layout()
    plt.show()

# Set of 10 cities
print("\nResults for 10 cities:")
for num_ants in [1, 5, 10, 20]:
    print(f"\nNumber of ants: {num_ants}")
    best_path, best_distance = ant_colony_optimization(distances_10, num_ants, 50)
    print(f"Best distance found: {best_distance}")
    print(f"Best path: {best_path}")
    plot_best_path_and_best_distance(distances_10, best_path, best_distance)

# Set of 20 cities
print("\nResults for 20 cities:")
for num_ants in [1, 5, 10, 20]:
    print(f"\nNumber of ants: {num_ants}")
    best_path, best_distance = ant_colony_optimization(distances_20, num_ants, 50)
    print(f"Best distance found: {best_distance}")
    print(f"Best path: {best_path}")
    plot_best_path_and_best_distance(distances_20, best_path, best_distance)
