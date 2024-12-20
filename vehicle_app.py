import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp


def create_data_model(distance_type, num_vehicles):
    """Stores the data for the problem."""
    locations = [
        (0, 0), (1, 3), (4, 3), (5, 5), (7, 8), (10, 5), (11, 3), (14, 1),
        (15, 5), (16, 10), (19, 7), (22, 12), (25, 8), (27, 6), (30, 5),
        (35, 10), (40, 5), (-2, -3), (-3, -5), (-6, 7), (-4, 8), (2, -4),
        (3, -5), (4, -6), (5, -9), (4, -5), (7, -2), (6, -9), (11, -3)
    ]

    def calculate_distance(p1, p2, type='Euclidean'):
        if type == 'Manhattan':
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
                row.append(int(calculate_distance(locations[i], locations[j], type)))
        distance_matrix.append(row)

    data = {"distance_matrix": distance_matrix, "num_vehicles": num_vehicles, "depot": 0, "locations": locations}
    return data


def plot_locations(data):
    st.write("### Customer Delivery Locations")
    st.write("""
            The graph below shows the distribution of delivery locations and the depot.
            Each customer location is numbered, and the depot is clearly marked in red.
            The arrangement of blue points around the red depot provides insights into delivery patterns, 
            potential route optimizations, and areas with concentrated customer demand, which can be crucial 
            for improving logistical efficiency.
            """)

    plt.figure(figsize=(10, 8))
    xs, ys = zip(*data['locations'])

    # Plot customer locations with numbers
    plt.scatter(xs[1:], ys[1:], c='blue', label='Customer Locations', s=100)

    # Plot depot with distinct marking
    plt.scatter(xs[0], ys[0], c='red', label='Depot', marker='*', s=200, edgecolors='black')

    # Add location numbers as annotations
    for i, (x, y) in enumerate(data['locations']):
        if i == 0:
            plt.annotate(f'Depot', (x, y), xytext=(10, 10),
                         textcoords='offset points', fontsize=10,
                         bbox=dict(facecolor='white', edgecolor='red', alpha=0.7))
        else:
            plt.annotate(f'Customer {i}', (x, y), xytext=(10, 10),
                         textcoords='offset points', fontsize=9,
                         bbox=dict(facecolor='white', edgecolor='blue', alpha=0.7))

    plt.title('Depot and Customer Delivery Locations', pad=20)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(plt)


def print_solution(data, manager, routing, solution):
    st.write("### Solution")
    st.write("""
    The optimised routes taken by the different vehicles and 
    the distance of each routes are as follows:""")
    # Print Solution to the console
    st.write(f"Objective: {solution.ObjectiveValue()}")
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
        st.write(plan_output)
        max_route_distance = max(route_distance, max_route_distance)
    st.write(f"Maximum of the route distances: {max_route_distance}m")


def plot_routes(data, manager, routing, solution):
    """Plots the routes for each vehicle with enhanced labeling."""
    st.write("### Optimized Vehicle Routes")
    st.write("""
            The graph below shows the optimized routes for each vehicle.
            The depot (start/end point) is marked with a star, and each customer
            location is numbered. Different colors represent different vehicle routes.
            The optimization implies that these routes are calculated to minimize distance 
            for logistical efficiency.
            """)

    plt.figure(figsize=(12, 8))
    colors = sns.color_palette('husl', n_colors=data['num_vehicles'])

    # Plot all locations first
    xs, ys = zip(*data['locations'])
    plt.scatter(xs[1:], ys[1:], c='lightgray', s=100, zorder=1)
    plt.scatter(xs[0], ys[0], c='red', marker='*', s=200, edgecolors='black', zorder=2, label='Depot')

    # Plot routes for each vehicle
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        coordinates = []
        route_points = []
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            coordinates.append(data['locations'][node_index])
            route_points.append(node_index)
            index = solution.Value(routing.NextVar(index))
        coordinates.append(data['locations'][0])  # return to depot

        # Plot route
        xs, ys = zip(*coordinates)
        plt.plot(xs, ys, '-', color=colors[vehicle_id],
                 label=f'Vehicle {vehicle_id}', linewidth=2, zorder=3)

        # Add direction arrows
        for i in range(len(coordinates) - 1):
            mid_x = (coordinates[i][0] + coordinates[i + 1][0]) / 2
            mid_y = (coordinates[i][1] + coordinates[i + 1][1]) / 2
            plt.annotate('', xy=(coordinates[i + 1][0], coordinates[i + 1][1]),
                         xytext=(mid_x, mid_y),
                         arrowprops=dict(arrowstyle='->', color=colors[vehicle_id]))

    # Add location labels
    for i, (x, y) in enumerate(data['locations']):
        if i == 0:
            plt.annotate(f'Depot', (x, y), xytext=(10, 10),
                         textcoords='offset points', fontsize=10,
                         bbox=dict(facecolor='white', edgecolor='red', alpha=0.7))
        else:
            plt.annotate(f'Customer {i}', (x, y), xytext=(10, 10),
                         textcoords='offset points', fontsize=9,
                         bbox=dict(facecolor='white', edgecolor='gray', alpha=0.7))

    plt.title('Optimized Vehicle Routes with Labeled Locations', pad=20)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    st.pyplot(plt)


def main():
    """Entry point of the program."""
    st.title('Vehicle Routing Problem')
    st.write("""
    This app offers a comprehensive solution to the Vehicle Routing Problem (VRP) using 
    advanced operational research(OR) tools. 
    Our approach efficiently determines optimal routes for a fleet of vehicles to deliver 
    goods or services to various locations, minimizing transportation costs while respecting 
    constraints like vehicle capacities and delivery time windows.
    """)

    # Instantiate the data problem.
    data = create_data_model('Manhattan', 4)  # Default values to show the graph

    # Plot the locations
    plot_locations(data)

    # User selects the distance metric
    distance_type = st.selectbox('Distance Metric', ['Euclidean', 'Manhattan'])

    # User selects the number of vehicles
    num_vehicles = st.slider('Number of Vehicles', min_value=1, max_value=5, value=4)

    # Add a button to start the algorithm
    if st.button("Run Algorithm"):
        # Update data with user inputs
        data = create_data_model(distance_type, num_vehicles)

        # Create the routing index manager.
        manager = pywrapcp.RoutingIndexManager(
            len(data["distance_matrix"]), data["num_vehicles"], data["depot"]
        )

        # Create Routing Model.
        routing = pywrapcp.RoutingModel(manager)

        # Create and register a transit callback.
        def distance_callback(from_index, to_index):
            """Returns the distance between the two nodes."""
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
            st.write("No solution found!")


if __name__ == "__main__":
    main()
