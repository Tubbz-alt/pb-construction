import heapq

import numpy as np

from extrusion.equilibrium import compute_all_reactions, compute_node_reactions
from extrusion.parsing import load_extrusion
from extrusion.utils import get_extructed_ids, force_from_reaction, create_stiffness_checker, torque_from_reaction
from pddlstream.utils import adjacent_from_edges
from pybullet_tools.utils import get_distance, INF

DISTANCE_HEURISTICS = [
    'z',
    'dijkstra',
    # 'fixed-dijkstra',
]
TOPOLOGICAL_HEURISTICS = [
    'length',
    'degree',
]
STIFFNESS_HEURISTICS = [
    'stiffness',  # Performs poorly with respect to stiffness
    'fixed-stiffness',
    'relative-stiffness',
    'load',
    'forces',
    'fixed-forces',
]
HEURISTICS = ['none'] + DISTANCE_HEURISTICS + STIFFNESS_HEURISTICS

##################################################

def compute_distance_from_node(elements, node_points, ground_nodes):
    #incoming_supporters, _ = neighbors_from_orders(get_supported_orders(
    #    element_from_id.values(), node_points))
    neighbors = adjacent_from_edges(elements)
    edge_costs = {edge: get_distance(node_points[edge[0]], node_points[edge[1]])
                  for edge in elements}
    edge_costs.update({edge[::-1]: distance for edge, distance in edge_costs.items()})

    cost_from_node = {}
    queue = []
    for node in ground_nodes:
        cost = 0
        cost_from_node[node] = cost
        heapq.heappush(queue, (cost, node))
    while queue:
        cost1, node1 = heapq.heappop(queue)
        if cost_from_node[node1] < cost1:
            continue
        for node2 in neighbors[node1]:
            cost2 = cost1 + edge_costs[node1, node2]
            if cost2 < cost_from_node.get(node2, INF):
                cost_from_node[node2] = cost2
                heapq.heappush(queue, (cost2, node2))
    return cost_from_node


def score_stiffness(extrusion_path, element_from_id, elements, checker=None):
    if not elements:
        return 0
    if checker is None:
        checker = create_stiffness_checker(extrusion_path)
    # TODO: analyze fixities projections in the xy plane

    # Lower is better
    extruded_ids = get_extructed_ids(element_from_id, elements)
    checker.solve(exist_element_ids=extruded_ids, if_cond_num=True)
    success, nodal_displacement, fixities_reaction, _ = checker.get_solved_results()
    if not success:
        return INF
    #operation = np.max
    operation = np.sum # equivalently average
    # TODO: LInf or L1 norm applied on forces
    # TODO: looking for a colored path through the space

    # trans unit: meter, rot unit: rad
    trans_tol, rot_tol = checker.get_nodal_deformation_tol()
    max_trans, max_rot, _, _ = checker.get_max_nodal_deformation()
    relative_trans = max_trans / trans_tol # lower is better
    relative_rot = max_rot / rot_tol # lower is better
    # More quickly approximate deformations by modifying the matrix operations incrementally

    reaction_forces = np.array([force_from_reaction(d) for d in fixities_reaction.values()])
    reaction_moments = np.array([torque_from_reaction(d) for d in fixities_reaction.values()])
    heuristic = 'compliance'
    scores = {
        # Yijiang was surprised that fixities_translation worked
        'fixities_translation': np.linalg.norm(reaction_forces, axis=1), # Bad when forward
        'fixities_rotation': np.linalg.norm(reaction_moments, axis=1), # Bad when forward
        'nodal_translation': np.linalg.norm(list(nodal_displacement.values()), axis=1),
        'compliance': [checker.get_compliance()],
        'deformation': [relative_trans, relative_rot],
    }
    # TODO: remove pairs of elements
    # TODO: clustering
    return operation(scores[heuristic])
    #return relative_trans
    #return max(relative_trans, relative_rot)
    #return relative_trans + relative_rot # arithmetic mean
    #return relative_trans * relative_rot # geometric mean
    #return 2*relative_trans * relative_rot / (relative_trans + relative_rot) # harmonic mean

##################################################

def get_heuristic_fn(extrusion_path, heuristic, forward, checker=None):
    # TODO: penalize disconnected
    element_from_id, node_points, ground_nodes = load_extrusion(extrusion_path)
    elements = frozenset(element_from_id.values())
    distance_from_node = compute_distance_from_node(elements, node_points, ground_nodes)
    sign = +1 if forward else -1
    # TODO: round values for more tie-breaking opportunities

    stiffness_cache = {}
    if heuristic in ('fixed-stiffness', 'relative-stiffness'):
        stiffness_cache.update({element: score_stiffness(extrusion_path, element_from_id, elements - {element},
                                                         checker=checker) for element in elements})
    reaction_cache = {}

    def fn(printed, element, conf):
        # Queue minimizes the statistic
        structure = printed | {element} if forward else printed - {element}
        structure_ids = get_extructed_ids(element_from_id, structure)
        # TODO: build away from the robot

        normalizer = 1
        #normalizer = len(structure)
        #normalizer = compute_element_distance(node_points, elements)

        operator = sum # sum | max
        fn = force_from_reaction  # force_from_reaction | torque_from_reaction

        if heuristic == 'none':
            return 0
        elif heuristic == 'degree':
            # TODO: online/offline and ground
            # TODO: other graph statistics
            #printed_nodes = {n for e in printed for n in e}
            #n1, n2 = element
            #node = n1 if n2 in printed_nodes else n2
            #if node in ground_nodes:
            #    return 0
            raise NotImplementedError()
        elif heuristic == 'length':
            # Equivalent to mass if uniform density
            n1, n2 = element
            return get_distance(node_points[n2], node_points[n1])
        elif heuristic == 'z':
            # TODO: tiebreak by angle or x
            z = np.average([node_points[n][2] for n in element])
            return sign*z
        elif heuristic == 'dijkstra':
            # min, max, node not in set
            # TODO: recompute online (but all at once)
            # TODO: sum of all element path distances
            normalizer = np.average([distance_from_node[node] for node in element])
            return sign * normalizer
        elif heuristic == 'load':
            nodal_loads = checker.get_nodal_loads(existing_ids=structure_ids, dof_flattened=False) # get_self_weight_loads
            return operator(np.linalg.norm(force_from_reaction(reaction)) for reaction in nodal_loads.values())
        elif heuristic == 'fixed-forces':
            #printed = elements # disable to use most up-to-date
            # TODO: relative to the load introduced
            if printed not in reaction_cache:
                reaction_cache[printed] = compute_all_reactions(extrusion_path, elements, checker=checker)
            force = operator(np.linalg.norm(fn(reaction)) for reaction in reaction_cache[printed].reactions[element])
            return force / normalizer
        elif heuristic == 'forces':
            reactions_from_nodes = compute_node_reactions(extrusion_path, structure, checker=checker)
            #torque = sum(np.linalg.norm(np.sum([torque_from_reaction(reaction) for reaction in reactions], axis=0))
            #            for reactions in reactions_from_nodes.values())
            #return torque / normalizer
            total = operator(np.linalg.norm(fn(reaction)) for reactions in reactions_from_nodes.values()
                            for reaction in reactions)
            return total / normalizer
            #return max(sum(np.linalg.norm(fn(reaction)) for reaction in reactions)
            #               for reactions in reactions_from_nodes.values())
        elif heuristic == 'stiffness':
            # TODO: add different variations
            # TODO: normalize by initial stiffness, length, or degree
            # Most unstable or least unstable first
            # Gets faster with fewer elements
            #old_stiffness = score_stiffness(extrusion_path, element_from_id, printed, checker=checker)
            stiffness = score_stiffness(extrusion_path, element_from_id, structure, checker=checker) # lower is better
            return stiffness / normalizer
            #return stiffness / old_stiffness
        elif heuristic == 'fixed-stiffness':
            # TODO: invert the sign for regression/progression?
            # TODO: sort FastDownward by the (fixed) action cost
            return stiffness_cache[element] / normalizer
        elif heuristic == 'relative-stiffness':
            stiffness = score_stiffness(extrusion_path, element_from_id, structure, checker=checker) # lower is better
            if normalizer == 0:
                return 0
            return stiffness / normalizer
            #return stiffness / stiffness_cache[element]
        raise ValueError(heuristic)
    return fn
