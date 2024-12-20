import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp


def create_data_model(distance_type, num_vehicles, vehicle_capacity):
    """Stores the data for the problem."""
    locations = [
        (0, 0),  # Depot
        (1, 3), (4, 3), (5, 5), (7, 8), (10, 5),
        (11, 3), (14, 1), (15, 5), (16, 10), (19, 7),
        (22, 12), (25, 8), (27, 6), (30, 5), (35, 10),
        (40, 5), (-2, -3), (-3, -5), (-6, 7), (-4, 8),
        (2, -4), (3, -5), (4, -6), (5, -9), (4, -5),
        (7, -2), (6, -9), (11, -3)
    ]

    # Generate random demands for each customer (depot has no demand)
    np.random.seed(42)  # For reproducibility
    demands = [0]  # Depot has no demand
    for _ in range(len(locations) - 1):
        demands.append(np.random.randint(1, 21))  # Random demand between 1 and 20

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

    data = {
        "distance_matrix": distance_matrix,
        "num_vehicles": num_vehicles,
        "depot": 0,
        "locations": locations,
        "demands": demands,
        "vehicle_capacity": vehicle_capacity
    }
    return data


def plot_locations(data):
    st.write("### Customer Delivery Locations and Demands")
    st.write("""
            The graph below shows the distribution of delivery locations and their demands.
            Each customer location is numbered, and the depot is marked in red.
            The size of each point represents the demand quantity at that location.
            """)

    plt.figure(figsize=(10, 8))
    xs, ys = zip(*data['locations'])

    # Plot customer locations with sizes based on demands
    demands_normalized = np.array(data['demands'][1:]) * 10  # Scale demands for visualization
    plt.scatter(xs[1:], ys[1:], c='blue', s=demands_normalized, alpha=0.6,
                label='Customer Locations')

    # Plot depot
    plt.scatter(xs[0], ys[0], c='red', marker='*', s=200,
                edgecolors='black', label='Depot')

    # Add location numbers and demand annotations
    for i, (x, y) in enumerate(data['locations']):
        if i == 0:
            plt.annotate(f'Depot', (x, y), xytext=(10, 10),
                         textcoords='offset points', fontsize=10,
                         bbox=dict(facecolor='white', edgecolor='red', alpha=0.7))
        else:
            plt.annotate(f'Customer {i}\nDemand: {data["demands"][i]}',
                         (x, y), xytext=(10, 10),
                         textcoords='offset points', fontsize=9,
                         bbox=dict(facecolor='white', edgecolor='blue', alpha=0.7))

    plt.title('Depot, Customer Locations, and Demands', pad=20)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(plt)


def print_solution(data, manager, routing, solution):
    st.write("### Solution")
    st.write("""
    The optimized routes for each vehicle, including distances and loads:""")

    total_distance = 0
    total_load = 0

    for vehicle_id in range(data["num_vehicles"]):
        index = routing.Start(vehicle_id)
        plan_output = f"Route for vehicle {vehicle_id}:\n"
        route_distance = 0
        route_load = 0

        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route_load += data["demands"][node_index]
            plan_output += f" {node_index} Load({route_load}) -> "

            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id
            )

        plan_output += f"{manager.IndexToNode(index)}\n"
        plan_output += f"Distance of the route: {route_distance}m\n"
        plan_output += f"Load of the route: {route_load}\n"

        st.text(plan_output)
        total_distance += route_distance
        total_load += route_load

    st.write(f"Total distance of all routes: {total_distance}m")
    st.write(f"Total load delivered: {total_load}")


def plot_routes(data, manager, routing, solution):
    st.write("### Optimized Vehicle Routes")
    st.write("""
            The graph below shows the optimized routes for each vehicle.
            - The depot is marked with a star
            - Customer locations are shown with circles sized by demand
            - Different colors represent different vehicle routes
            - Arrows show the direction of travel
            """)

    plt.figure(figsize=(12, 8))
    colors = sns.color_palette('husl', n_colors=data['num_vehicles'])

    # Plot all locations
    xs, ys = zip(*data['locations'])
    demands_normalized = np.array(data['demands']) * 10
    plt.scatter(xs[1:], ys[1:], c='lightgray', s=demands_normalized[1:],
                alpha=0.5, zorder=1)
    plt.scatter(xs[0], ys[0], c='red', marker='*', s=200,
                edgecolors='black', zorder=2, label='Depot')

    # Plot routes for each vehicle
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        coordinates = []
        route_load = 0
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            coordinates.append(data['locations'][node_index])
            route_load += data['demands'][node_index]
            index = solution.Value(routing.NextVar(index))
        coordinates.append(data['locations'][0])  # Return to depot

        # Plot route
        xs, ys = zip(*coordinates)
        plt.plot(xs, ys, '-', color=colors[vehicle_id],
                 label=f'Vehicle {vehicle_id} (Load: {route_load})',
                 linewidth=2, zorder=3)

        # Add direction arrows
        for i in range(len(coordinates) - 1):
            mid_x = (coordinates[i][0] + coordinates[i + 1][0]) / 2
            mid_y = (coordinates[i][1] + coordinates[i + 1][1]) / 2
            plt.annotate('', xy=(coordinates[i + 1][0], coordinates[i + 1][1]),
                         xytext=(mid_x, mid_y),
                         arrowprops=dict(arrowstyle='->', color=colors[vehicle_id]))

    # Add location labels with demands
    for i, (x, y) in enumerate(data['locations']):
        if i == 0:
            plt.annotate(f'Depot', (x, y), xytext=(10, 10),
                         textcoords='offset points', fontsize=10,
                         bbox=dict(facecolor='white', edgecolor='red', alpha=0.7))
        else:
            plt.annotate(f'Customer {i}\nDemand: {data["demands"][i]}',
                         (x, y), xytext=(10, 10),
                         textcoords='offset points', fontsize=9,
                         bbox=dict(facecolor='white', edgecolor='gray', alpha=0.7))

    plt.title('Optimized Vehicle Routes with Demands', pad=20)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    st.pyplot(plt)


def main():
    st.title('Vehicle Routing Problem with Capacity Constraints')
    st.write("""
    This app solves the Vehicle Routing Problem (VRP) with capacity constraints using 
    OR-Tools. Each customer has a specific demand, and vehicles have limited capacity.
    The solution minimizes the total distance while ensuring all demands are met and 
    vehicle capacities are not exceeded.
    """)
    # Instantiate the data problem and plot
    data = create_data_model('Manhattan', 3, vehicle_capacity=100)
    plot_locations(data)

    # User inputs
    col1, col2, col3 = st.columns(3)

    with col1:
        distance_type = st.selectbox('Distance Metric', ['Euclidean', 'Manhattan'])

    with col2:
        num_vehicles = st.slider('Number of Vehicles',
                                 min_value=1, max_value=10, value=3)

    with col3:
        vehicle_capacity = st.number_input('Vehicle Capacity',
                                           min_value=10, max_value=200,
                                           value=100, step=10)

    # Create user data
    data = create_data_model(distance_type, num_vehicles, vehicle_capacity)

    # Add a button to start the algorithm
    if st.button("Optimize Routes"):
        # Create the routing index manager
        manager = pywrapcp.RoutingIndexManager(
            len(data["distance_matrix"]), data["num_vehicles"], data["depot"]
        )

        # Create Routing Model
        routing = pywrapcp.RoutingModel(manager)

        # Create and register a transit callback
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return data["distance_matrix"][from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Add Capacity constraint
        def demand_callback(from_index):
            from_node = manager.IndexToNode(from_index)
            return data["demands"][from_node]

        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,  # null capacity slack
            [data["vehicle_capacity"]] * data["num_vehicles"],  # vehicle maximum capacities
            True,  # start cumul to zero
            'Capacity')

        # Add Distance constraint
        dimension_name = 'Distance'
        routing.AddDimension(
            transit_callback_index,
            0,  # no slack
            3000,  # vehicle maximum travel distance
            True,  # start cumul to zero
            dimension_name)
        distance_dimension = routing.GetDimensionOrDie(dimension_name)
        distance_dimension.SetGlobalSpanCostCoefficient(100)

        # Setting first solution heuristic
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
        search_parameters.time_limit.FromSeconds(30)

        # Solve the problem
        solution = routing.SolveWithParameters(search_parameters)

        # Print solution
        if solution:
            print_solution(data, manager, routing, solution)
            plot_routes(data, manager, routing, solution)
        else:
            st.error("""No solution found! 
                    Try increasing the number of vehicles or vehicle capacity.""")


if __name__ == "__main__":
    main()
