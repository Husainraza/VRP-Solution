import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp


def create_data_model():
    """Stores the data for the problem."""
    locations = [
        (0, 0), (1, 3), (4, 3), (5, 5), (7, 8), (10, 5), (11, 3), (14, 1),
        (15, 5), (16, 10), (19, 7), (22, 12), (25, 8), (27, 6), (30, 5),
        (35, 10), (40, 5)
    ]
    # Choose the distance calculation type: 'Manhattan' or 'Euclidean'
    distance_type = 'euclidean'

    def calculate_distance(p1, p2, type='euclidean'):
        if type == 'manhattan':
            return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
        else:  # Euclidean
            return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    distance_matrix = []
    for i in range(len(locations)):
        row = []
        for j in range(len(locations)):
            if i == j:
                row.append(0)
            else:
                row.append(int(calculate_distance(locations[i], locations[j], distance_type)))
        distance_matrix.append(row)

    data = {"distance_matrix": distance_matrix, "num_vehicles": 4, "depot": 0, "locations": locations}
    return data


def print_solution(data, manager, routing, solution):
    """Prints solution on console."""
    print(f"Objective: {solution.ObjectiveValue()}")
    max_route_distance = 0
    for vehicle_id in range(data["num_vehicles"]):
        index = routing.Start(vehicle_id)
        plan_output = f"Route for vehicle {vehicle_id}:\n"
        route_distance = 0
        while not routing.IsEnd(index):
            plan_output += f" {manager.IndexToNode(index)} -> "
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id
            )
        plan_output += f"{manager.IndexToNode(index)}\n"
        plan_output += f"Distance of the route: {route_distance}m\n"
        print(plan_output)
        max_route_distance = max(route_distance, max_route_distance)
    print(f"Maximum of the route distances: {max_route_distance}m")


def plot_routes(data, manager, routing, solution):
    """Plots the routes for each vehicle."""
    plt.figure(figsize=(12, 8))
    colors = sns.color_palette('hsv', n_colors=data['num_vehicles'])
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        coordinates = []
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            coordinates.append(data['locations'][node_index])
            index = solution.Value(routing.NextVar(index))
        coordinates.append(data['locations'][0])  # return to depot
        xs, ys = zip(*coordinates)
        plt.plot(xs, ys, marker='o', color=colors[vehicle_id], label=f'Vehicle {vehicle_id}')
    plt.title('Vehicle Routing Problem Solution')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    """Entry point of the program."""
    # Instantiate the data problem.
    data = create_data_model()

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(
        len(data["distance_matrix"]), data["num_vehicles"], data["depot"]
    )

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    # Create and register a transit callback.
    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data["distance_matrix"][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add Distance constraint.
    dimension_name = "Distance"
    routing.AddDimension(
        transit_callback_index,
        0,  # no slack
        3000,  # vehicle maximum travel distance
        True,  # start cumul to zero
        dimension_name,
    )
    distance_dimension = routing.GetDimensionOrDie(dimension_name)
    distance_dimension.SetGlobalSpanCostCoefficient(100)

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    # Print solution on console.
    if solution:
        print_solution(data, manager, routing, solution)
        plot_routes(data, manager, routing, solution)
    else:
        print("No solution found !")


if __name__ == "__main__":
    main()
  
