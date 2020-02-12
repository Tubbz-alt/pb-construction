import heapq
import math
import random
import time
from collections import defaultdict
from itertools import product, combinations

import numpy as np

from extrusion.utils import get_pairs, get_midpoint, SUPPORT_THETA, get_undirected, compute_element_distance
from pddlstream.utils import get_connected_components
from pybullet_tools.utils import get_distance, elapsed_time, BLACK, wait_for_user, BLUE, RED, get_pitch, INF
from extrusion.stiffness import plan_stiffness

INITIAL_NODE = None
SCALE = 1e3 # millimeters
INVALID = 1e1 # meters (just make larger than sum of path)

STATUS = """
ROUTING_NOT_SOLVED: Problem not solved yet.
ROUTING_SUCCESS: Problem solved successfully.
ROUTING_FAIL: No solution found to the problem.
ROUTING_FAIL_TIMEOUT: Time limit reached before finding a solution.
ROUTING_INVALID: Model, model parameters, or flags are not valid.
""".strip().split('\n')

##################################################

def print_solution(manager, routing, solution):
    """Prints solution on console."""
    max_route_distance = 0
    vehicle_id = 0
    #for vehicle_id in range(data['num_vehicles']):
    index = routing.Start(vehicle_id)
    plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
    route_distance = 0
    while not routing.IsEnd(index):
        plan_output += ' {} -> '.format(manager.IndexToNode(index))
        previous_index = index
        index = solution.Value(routing.NextVar(index))
        route_distance += routing.GetArcCostForVehicle(
            previous_index, index, vehicle_id)
    plan_output += '{}\n'.format(manager.IndexToNode(index))
    plan_output += 'Distance of the route: {}m\n'.format(route_distance)
    print(plan_output)
    max_route_distance = max(route_distance, max_route_distance)
    print('Maximum of the route distances: {}m'.format(max_route_distance))


def solve_tsp(elements, ground_nodes, node_points, initial_point, max_time=30, verbose=False):
    # https://developers.google.com/optimization/routing/tsp
    # https://developers.google.com/optimization/reference/constraint_solver/routing/RoutingModel
    # http://www.math.uwaterloo.ca/tsp/concorde.html
    # AddDisjunction
    # TODO: pick up and delivery
    # TODO: time window for skipping elements
    # TODO: Minimum Spanning Tree (MST) bias
    # TODO: time window constraint to ensure connected
    # TODO: reuse by simply swapping out the first vertex
    from ortools.constraint_solver import routing_enums_pb2, pywrapcp
    from extrusion.visualization import draw_ordered, draw_model
    from extrusion.heuristics import compute_distance_from_node
    assert initial_point is not None
    start_time = time.time()

    extrusion_edges = set()
    point_from_vertex = {INITIAL_NODE: initial_point}
    frame_nodes = set()

    keys_from_node = defaultdict(set)
    for element in elements:
        mid = (element, element)
        point_from_vertex[mid] = get_midpoint(node_points, element)
        for node in element:
            end = (element, node)
            point_from_vertex[end] = node_points[node]
            frame_nodes.add(end)
            keys_from_node[node].add(end)
            for direction in {(end, mid), (mid, end)}:
                # Add edges from anything that is roughly the correct cost
                # TODO: compute supporters from Dijkstra
                start, end = direction
                #delta = point_from_vertex[end] - point_from_vertex[start]
                #pitch = get_pitch(delta)
                #if -SUPPORT_THETA <= pitch:
                extrusion_edges.add(direction)
    # TODO: always keep edges on the minimum cost tree

    for node in keys_from_node:
        for edge in product(keys_from_node[node], repeat=2):
            extrusion_edges.add(edge)

    transit_edges = {pair for pair in product(frame_nodes, repeat=2)}
    transit_edges.update({(INITIAL_NODE, key) for node in ground_nodes for key in keys_from_node[node]}) # initial -> ground
    transit_edges.update({(key, INITIAL_NODE) for key in frame_nodes}) # any -> initial

    key_from_index = list({k for pair in extrusion_edges | transit_edges for k in pair})
    edge_weights = {pair: INVALID for pair in product(key_from_index, repeat=2)}
    for k1, k2 in transit_edges:
        p1, p2 = point_from_vertex[k1], point_from_vertex[k2]
        edge_weights[k1, k2] = get_distance(p1, p2)
    edge_weights.update({e: 0. for e in extrusion_edges}) # frame edges are free

    index_from_key = dict(map(reversed, enumerate(key_from_index)))
    num_vehicles, depot = 1, index_from_key[INITIAL_NODE]
    print('Initial vertex: {}'.format(depot))
    manager = pywrapcp.RoutingIndexManager(len(key_from_index), num_vehicles, depot)
                                           #[depot], [depot])
    solver = pywrapcp.RoutingModel(manager)

    cost_from_index = {}
    for (k1, k2), weight in edge_weights.items():
        i1, i2 = index_from_key[k1], index_from_key[k2]
        cost_from_index[i1, i2] = int(math.ceil(SCALE * weight))
    solver.SetArcCostEvaluatorOfAllVehicles(solver.RegisterTransitCallback(
        lambda i1, i2: cost_from_index[manager.IndexToNode(i1), manager.IndexToNode(i2)]))  # from -> to

    # sequence = plan_stiffness(None, None, node_points, ground_nodes, elements,
    #                           initial_position=initial_point, stiffness=False, max_backtrack=INF)

    node_from_vertex = compute_distance_from_node(elements, node_points, ground_nodes)
    cost_from_edge = {}
    for edge in elements:
        n1, n2 = edge
        if node_from_vertex[n1].cost <= node_from_vertex[n2].cost:
            cost_from_edge[n1, n2] = node_from_vertex[n1].cost
        else:
            cost_from_edge[n2, n1] = node_from_vertex[n2].cost
    sequence = sorted(cost_from_edge, key=lambda e: cost_from_edge[e])

    initial_order = []
    #initial_order = [INITIAL_NODE] # Start and end automatically included
    for directed in sequence:
        node1, node2 = directed
        element = get_undirected(elements, directed)
        initial_order.extend([
            (element, node1),
            (element, element),
            (element, node2),
        ])
    #initial_order.append(INITIAL_NODE)
    initial_route = [index_from_key[key] for key in initial_order]
    #index = initial_route.index(0)
    #initial_route = initial_route[index:] + initial_route[:index] + [0]

    initial_solution = solver.ReadAssignmentFromRoutes([initial_route], ignore_inactive_indices=True)
    assert initial_solution is not None
    #print_solution(manager, solver, initial_solution)
    #print(solver.GetAllDimensionNames())
    #print(solver.ComputeLowerBound())

    total_distance = compute_element_distance(node_points, elements)
    total = initial_solution.ObjectiveValue() / SCALE
    invalid = int(total / INVALID)
    print('Initial solution | Vertices: {} | Edges: {} | Structure: {:.3f} | Invalid: {} | Cost: {:.3f} | Duration: {:.3f}s'.format(
        len(key_from_index), len(edge_weights), total_distance, invalid, total, elapsed_time(start_time)))

    # def time_callback(from_index, to_index):
    #     """Returns the travel time between the two nodes."""
    #     # Convert from routing variable Index to time matrix NodeIndex.
    #     from_node = manager.IndexToNode(from_index)
    #     to_node = manager.IndexToNode(to_index)
    #     return 1
    #     #return data['time_matrix'][from_node][to_node]
    #
    # transit_callback_index = solver.RegisterTransitCallback(time_callback)
    # step = 'Time'
    # solver.AddDimension(
    #     transit_callback_index,
    #     30,  # allow waiting time
    #     30,  # maximum time per vehicle
    #     False,  # Don't force start cumul to zero.
    #     step)
    #
    # time_dimension = solver.GetDimensionOrDie(step)
    # for node, index in index_from_key.items():
    #     time_dimension.CumulVar(manager.NodeToIndex(index))

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    #search_parameters.solution_limit = 1
    search_parameters.time_limit.seconds = int(max_time)
    search_parameters.log_search = verbose
    #search_parameters.first_solution_strategy = (
    #    routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC) # AUTOMATIC | PATH_CHEAPEST_ARC
    #search_parameters.local_search_metaheuristic = (
    #    routing_enums_pb2.LocalSearchMetaheuristic.AUTOMATIC) # AUTOMATIC | GUIDED_LOCAL_SEARCH
    #solution = solver.SolveWithParameters(search_parameters)
    solution = solver.SolveFromAssignmentWithParameters(initial_solution, search_parameters)
    #print_solution(manager, solver, solution)

    print('Status: {}'.format(solver.status()))
    print(STATUS[solver.status()])
    if not solution:
        print('Failure! Duration: {:.3f}s'.format(elapsed_time(start_time)))
        return None

    total = solution.ObjectiveValue() / SCALE
    invalid = int(total / INVALID)
    print('Initial solution | Vertices: {} | Edges: {} | Structure: {:.3f} | Invalid: {} | Cost: {:.3f} | Duration: {:.3f}s'.format(
        len(key_from_index), len(edge_weights), total_distance, invalid, total, elapsed_time(start_time)))
    index = solver.Start(0)
    order = []
    while not solver.IsEnd(index):
        order.append(key_from_index[manager.IndexToNode(index)])
        #previous_index = index
        index = solution.Value(solver.NextVar(index))
        #route_distance += solver.GetArcCostForVehicle(previous_index, index, 0)
    order.append(key_from_index[manager.IndexToNode(index)])

    #start = order.index(INITIAL_NODE)
    #print(start, index_from_key[INITIAL_NODE])
    #order = order[start:] + order[:start] + [INITIAL_NODE]
    #print(order)

    # TODO: visualize by weight
    tour_pairs = get_pairs(order) + get_pairs(list(reversed(order)))

    draw_model(extrusion_edges - set(tour_pairs), point_from_vertex, ground_nodes, color=BLACK)
    draw_ordered(tour_pairs, point_from_vertex)
    wait_for_user()

    return order

##################################################

def compute_spanning_tree(edge_weights):
    tree = set()
    connected = set()
    if not edge_weights:
        return tree
    adjacent = defaultdict(set)
    for e in edge_weights:
        for v in e:
            adjacent[v].add(e)
    root, _ = random.choice(list(edge_weights))
    queue = []
    for edge in adjacent[root]:
        heapq.heappush(queue, (edge_weights[edge], edge))
    while queue:
        weight, edge = heapq.heappop(queue)
        if set(edge) <= connected:
            continue
        vertex = edge[0 if edge[0] not in connected else 1]
        tree.add(edge)
        connected.update(edge)
        for edge in adjacent[vertex]:
            heapq.heappush(queue, (edge_weights[edge], edge))
    return tree


def embed_graph(elements, node_points, initial_position=None, num=1):
    point_from_vertex = dict(enumerate(node_points))

    if initial_position is not None:
        point_from_vertex[INITIAL_NODE] = initial_position

    frame_edges = set()
    for element in elements:
        n1, n2 = element
        path = [n1]
        for t in np.linspace(0, 1, num=2+num, endpoint=True)[1:-1]:
            n3 = len(point_from_vertex)
            point_from_vertex[n3] = t*node_points[n1] + (1-t)*node_points[n2]
            path.append(n3)
        path.append(n2)
        for directed in get_pairs(path) + get_pairs(list(reversed(path))):
            frame_edges.add(directed)

    return point_from_vertex, frame_edges


def compute_euclidean_tree(node_points, ground_nodes, elements, initial_position=None):
    # remove printed elements from the tree
    # TODO: directed version
    start_time = time.time()
    point_from_vertex, edges = embed_graph(elements, node_points, ground_nodes, initial_position)
    edge_weights = {(n1, n2): get_distance(point_from_vertex[n1], point_from_vertex[n2]) for n1, n2 in edges}
    components = get_connected_components(point_from_vertex, edge_weights)
    tree = compute_spanning_tree(edge_weights)
    from extrusion.visualization import draw_model
    draw_model(tree, point_from_vertex, ground_nodes, color=BLUE)
    draw_model(edges - tree, point_from_vertex, ground_nodes, color=RED)

    weight = sum(edge_weights[e] for e in tree)
    print(len(components), len(point_from_vertex), len(tree), weight, elapsed_time(start_time))
    wait_for_user()
    return weight

##################################################

def compute_component_mst(node_points, ground_nodes, unprinted, initial_position=None):
    # Weighted A*
    start_time = time.time()
    point_from_vertex = dict(enumerate(node_points))
    vertices = {v for e in unprinted for v in e}
    components = get_connected_components(vertices, unprinted)

    entry_nodes = set(ground_nodes)
    if initial_position is not None:
        point_from_vertex[INITIAL_NODE] = initial_position
        components.append([INITIAL_NODE])
        entry_nodes.add(INITIAL_NODE)

    edge_weights = {}
    for c1, c2 in combinations(range(len(components)), r=2):
        # TODO: directed edges from all points to entry nodes
        nodes1 = set(components[c1])
        nodes2 = set(components[c2])
        if (c1 == [INITIAL_NODE]) or (c2 == [INITIAL_NODE]):
            nodes1 &= entry_nodes
            nodes2 &= entry_nodes
        edge_weights[c1, c2] = min(get_distance(point_from_vertex[n1], point_from_vertex[n2])
                                   for n1, n2 in product(nodes1, nodes2))

    tree = compute_spanning_tree(edge_weights)
    weight = sum(edge_weights[e] for e in tree)
    print('Elements: {} | Components: {} | Tree: {} | Weight: {}: Time: {:.3f} '.format(
        len(unprinted), len(components), tree, weight, elapsed_time(start_time)))
    return weight