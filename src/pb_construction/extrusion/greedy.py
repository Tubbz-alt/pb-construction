from __future__ import print_function

import heapq
import random
import time

from collections import namedtuple, OrderedDict

import numpy as np

from pybullet_planning import INF
from pybullet_planning import elapsed_time, remove_all_debug, wait_for_user, has_gui, LockRenderer, \
    reset_simulation, disconnect, set_renderer, randomize, connect, ClientSaver, get_distance

from pb_construction.extrusion.parsing import load_extrusion
from pb_construction.extrusion.visualization import draw_element, draw_model, draw_ordered
from pb_construction.extrusion.stream import get_print_gen_fn
from pb_construction.extrusion.utils import check_connected, torque_from_reaction, force_from_reaction, compute_element_distance, \
    test_stiffness, create_stiffness_checker, get_id_from_element, load_world, get_supported_orders, get_extructed_ids, \
    nodes_from_elements, PrintTrajectory
from pb_construction.extrusion.equilibrium import compute_node_reactions, compute_all_reactions

from pddlstream.utils import neighbors_from_orders, adjacent_from_edges, implies

#State = namedtuple('State', ['element', 'printed', 'plan'])
Node = namedtuple('Node', ['action', 'state'])

def retrace_trajectories(visited, current_state):
    command, prev_state = visited[current_state]
    if prev_state is None:
        return []
    return retrace_trajectories(visited, prev_state) + [traj for traj in command.trajectories]
    # TODO: search over local stability for each node

def retrace_elements(visited, current_state):
    return [traj.element for traj in retrace_trajectories(visited, current_state)
            if isinstance(traj, PrintTrajectory)]

##################################################

def compute_printed_nodes(ground_nodes, printed):
    return nodes_from_elements(printed) | set(ground_nodes)

def sample_extrusion(print_gen_fn, ground_nodes, printed, element):
    next_nodes =  compute_printed_nodes(ground_nodes, printed)
    # TODO: could always reverse these trajectories
    for node in element:
        if node in next_nodes:
            try:
                command, = next(print_gen_fn(node, element, extruded=printed))
                return command
            except StopIteration:
                pass
    return None

def display_failure(node_points, extruded_elements, element):
    client = connect(use_gui=True)
    with ClientSaver(client):
        obstacles, robot = load_world()
        handles = []
        for e in extruded_elements:
            handles.append(draw_element(node_points, e, color=(0, 1, 0)))
        handles.append(draw_element(node_points, element, color=(1, 0, 0)))
        print('Failure!')
        wait_for_user()
        reset_simulation()
        disconnect()

def draw_action(node_points, printed, element):
    if not has_gui():
        return []
    with LockRenderer():
        remove_all_debug()
        handles = [draw_element(node_points, element, color=(1, 0, 0))]
        handles.extend(draw_element(node_points, e, color=(0, 1, 0)) for e in printed)
    wait_for_user()
    return handles

##################################################

GREEDY_HEURISTICS = [
    'none',
    'z',
    #'dijkstra',
    #'fixed-dijkstra',
    'stiffness', # Performs poorly with respect to stiffness
    'fixed-stiffness',
    'relative-stiffness',
    'length',
    'degree',
    'load',
    'forces',
    'fixed-forces',
]

STIFFNESS_CRITERIA = [
    'fixities_translation',
    'fixities_rotation',
    'nodal_translation',
    'compliance',
    'deformation',
]

# TODO: visualize the branching factor

def get_heuristic_fn(extrusion_path, heuristic, forward, checker=None, stiffness_criteria='compliance'):
    # TODO: penalize disconnected
    element_from_id, node_points, ground_nodes = load_extrusion(extrusion_path)
    elements = frozenset(element_from_id.values())
    distance_from_node = compute_distance_from_node(elements, node_points, ground_nodes)
    sign = +1 if forward else -1

    stiffness_cache = {}
    if heuristic in ('fixed-stiffness', 'relative-stiffness'):
        stiffness_cache.update({element: score_stiffness(extrusion_path, element_from_id, elements - {element},
                                                         checker=checker) for element in elements})

    reaction_cache = {}

    def fn(printed, element, conf=None):
        # Queue minimizes the statistic
        structure = printed | {element} if forward else printed - {element}
        structure_ids = get_extructed_ids(element_from_id, structure)
        # TODO: build away from the robot

        distance = 1
        #distance = len(structure)
        #distance = compute_element_distance(node_points, elements)

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
            # TODO: round values for more tie-breaking opportunities
            z = get_z(node_points, element)
            return sign*z
        elif heuristic == 'load':
            nodal_loads = checker.get_nodal_loads(existing_ids=structure_ids, dof_flattened=False) # get_self_weight_loads
            return operator(np.linalg.norm(force_from_reaction(reaction)) for reaction in nodal_loads.values())
        elif heuristic == 'fixed-forces':
            #printed = elements # disable to use most up-to-date
            # TODO: relative to the load introduced
            if printed not in reaction_cache:
                reaction_cache[printed] = compute_all_reactions(extrusion_path, elements, checker=checker)
            force = operator(np.linalg.norm(fn(reaction)) for reaction in reaction_cache[printed].reactions[element])
            return force / distance
        elif heuristic == 'forces':
            reactions_from_nodes = compute_node_reactions(extrusion_path, structure, checker=checker)
            #torque = sum(np.linalg.norm(np.sum([torque_from_reaction(reaction) for reaction in reactions], axis=0))
            #            for reactions in reactions_from_nodes.values())
            #return torque / distance
            total = operator(np.linalg.norm(fn(reaction)) for reactions in reactions_from_nodes.values()
                            for reaction in reactions)
            return total / distance
            #return max(sum(np.linalg.norm(fn(reaction)) for reaction in reactions)
            #               for reactions in reactions_from_nodes.values())
        elif heuristic == 'stiffness':
            # TODO: add different variations
            # TODO: normalize by initial stiffness, length, or degree
            # Most unstable or least unstable first
            # Gets faster with fewer elements
            structure = printed | {element} if forward else printed - {element}
            return score_stiffness(extrusion_path, element_from_id, structure, checker=checker) # lower is better
        elif heuristic == 'fixed-stiffness':
            # TODO: invert the sign for regression/progression?
            # TODO: sort FastDownward by the (fixed) action cost
            return stiffness_cache[element] / distance
        elif heuristic == 'relative-stiffness':
            stiffness = score_stiffness(extrusion_path, element_from_id, structure, checker=checker) # lower is better
            if distance == 0:
                return 0
            return stiffness / distance
            #return stiffness / stiffness_cache[element]
        elif heuristic == 'dijkstra':
            # min, max, node not in set
            # TODO: recompute online (but all at once)
            # TODO: sum of all element path distances
            distance = np.average([distance_from_node[node] for node in element])
            return sign * distance
        raise ValueError(heuristic)
    return fn

def get_z(node_points, element):
    # TODO: tiebreak by angle or x
    return np.average([node_points[n][2] for n in element])

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

def score_stiffness(extrusion_path, element_from_id, elements, checker=None, stiffness_criteria='compliance'):
    if not elements:
        return 0
    if checker is None:
        checker = create_stiffness_checker(extrusion_path)
    # TODO: analyze fixities projections in the xy plane

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
        'compliance': [-checker.get_compliance()], # negated because higher is better
        'deformation': [relative_trans, relative_rot],
    }
    return operation(scores[heuristic])
    #return relative_trans
    #return max(relative_trans, relative_rot)
    #return relative_trans + relative_rot # arithmetic mean
    #return relative_trans * relative_rot # geometric mean
    #return 2*relative_trans * relative_rot / (relative_trans + relative_rot) # harmonic mean

##################################################

def export_log_data(extrusion_file_path, log_data, overwrite=True):
    import os
    import datetime
    import json

    with open(extrusion_file_path, 'r') as f:
        shape_data = json.loads(f.read())
    
    if 'model_name' in shape_data:
        file_name = shape_data['model_name']
    else:
        file_name = extrusion_file_path.split('.json')[-2].split(os.sep)[-1]

    result_file_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'extrusion_log')
    if not os.path.exists(result_file_dir):
        os.makedirs(result_file_dir) 
    
    data = OrderedDict()
    data['assembly_type'] = 'extrusion'
    data['file_name'] = file_name
    data['write_time'] = str(datetime.datetime.now())
    data.update(log_data)

    file_name_tag = log_data['search_method'] + '-' + log_data['heuristic']
    if log_data['heuristic'] in ['stiffness', 'fixed-stiffness']:
        file_name_tag += '-' + log_data['stiffness_criteria']
    plan_path = os.path.join(result_file_dir, '{}_log_{}{}.json'.format(file_name, 
        file_name_tag,  '_'+data['write_time'] if not overwrite else ''))
    with open(plan_path, 'w') as f:
        # json.dump(data, f, indent=2, sort_keys=True)
        json.dump(data, f)

##################################################

def add_successors(queue, elements, node_points, ground_nodes, heuristic_fn, printed, conf, visualize=False):
    """successor generation function used in progression & deadend algorithm
    
    Parameters
    ----------
    queue : [type]
        [description]
    elements : [type]
        [description]
    node_points : [type]
        [description]
    ground_nodes : [type]
        [description]
    heuristic_fn : [type]
        [description]
    printed : [type]
        [description]
    conf : [type]
        [description]
    visualize : bool, optional
        [description], by default False
    """
    remaining = elements - printed
    num_remaining = len(remaining) - 1
    assert 0 <= num_remaining
    nodes = compute_printed_nodes(ground_nodes, printed)
    bias_from_element = {}
    for element in randomize(remaining):
        if any(n in nodes for n in element):
            bias = heuristic_fn(printed, element, conf)
            priority = (num_remaining, bias, random.random())
            visits = 0
            heapq.heappush(queue, (visits, priority, printed, element, conf))
            bias_from_element[element] = bias

    if visualize and has_gui():
        handles = []
        with LockRenderer():
            remove_all_debug()
            for element in printed:
                handles.append(draw_element(node_points, element, color=(0, 0, 0)))
            successors = sorted(bias_from_element, key=lambda e: bias_from_element[e])
            handles.extend(draw_ordered(successors, node_points))
        print('Min: {:.3E} | Max: {:.3E}'.format(bias_from_element[successors[0]], bias_from_element[successors[-1]]))
        wait_for_user()

def progression(robot, obstacles, element_bodies, extrusion_path,
                heuristic='z', max_time=INF, max_backtrack=INF, 
                stiffness=True, stiffness_criteria='compliance', **kwargs):

    start_time = time.time()
    element_from_id, node_points, ground_nodes = load_extrusion(extrusion_path)
    checker = create_stiffness_checker(extrusion_path, verbose=False)
    #checker = None
    print_gen_fn = get_print_gen_fn(robot, obstacles, node_points, element_bodies, ground_nodes,
                                    precompute_collisions=False, max_directions=500, **kwargs)
    id_from_element = get_id_from_element(element_from_id)
    elements = frozenset(element_bodies)
    heuristic_fn = get_heuristic_fn(extrusion_path, heuristic, checker=checker, forward=True, stiffness_criteria=stiffness_criteria)

    final_printed = frozenset(element_bodies)
    if not check_connected(ground_nodes, final_printed) or \
            not test_stiffness(extrusion_path, element_from_id, final_printed):
        data = {
            'sequence': None,
            'runtime': elapsed_time(start_time),
        }
        return None, data

    initial_conf = None
    initial_printed = frozenset()
    queue = []
    visited = {initial_printed: Node(None, None)}
    #add_successors(initial_printed)
    add_successors(queue, elements, node_points, ground_nodes, heuristic_fn, initial_printed, initial_conf)

    plan = None
    min_remaining = INF
    num_evaluated = 0
    while queue and (elapsed_time(start_time) < max_time):
        num_evaluated += 1

        # (visits, priority, printed, element, conf)
        _, _, printed, element, _ = heapq.heappop(queue)
        num_remaining = len(elements) - len(printed)
        backtrack = num_remaining - min_remaining
        if max_backtrack <= backtrack:
            continue
        num_evaluated += 1
        if num_remaining < min_remaining:
            min_remaining = num_remaining
        print('Iteration: {} | Best: {} | Printed: {} | Element: {} | Index: {} | Time: {:.3f}'.format(
            num_evaluated, min_remaining, len(printed), element, id_from_element[element], elapsed_time(start_time)))
        next_printed = printed | {element}
        if (next_printed in visited) or not check_connected(ground_nodes, next_printed) or \
                (stiffness and not test_stiffness(extrusion_path, element_from_id, next_printed, checker=checker)):
            continue
        command = sample_extrusion(print_gen_fn, ground_nodes, printed, element)
        if command is None:
            continue
        visited[next_printed] = Node(command, printed)
        if elements <= next_printed:
            min_remaining = 0
            plan = retrace_trajectories(visited, next_printed)
            break
        #add_successors(next_printed)
        add_successors(queue, elements, node_points, ground_nodes, heuristic_fn, next_printed, initial_conf)

    sequence = None
    if plan is not None:
        sequence = [traj.directed_element for traj in plan if isinstance(traj, PrintTrajectory)]
    data = {
        'sequence': sequence,
        'runtime': elapsed_time(start_time),
        'num_evaluated': num_evaluated,
        'num_remaining': min_remaining,
        'num_elements': len(elements)
    }
    return plan, data

##################################################

def regression(robot, obstacles, element_bodies, extrusion_path,
               heuristic='z', max_time=INF, max_backtrack=INF, stiffness=True, 
               stiffness_criteria='compliance', log=False, **kwargs):
    # Focused has the benefit of reusing prior work
    # Greedy has the benefit of conditioning on previous choices
    # TODO: persistent search to reuse
    # TODO: max branching factor

    start_time = time.time()
    element_from_id, node_points, ground_nodes = load_extrusion(extrusion_path)
    id_from_element = get_id_from_element(element_from_id)
    checker = create_stiffness_checker(extrusion_path, verbose=False)
    #checker = None
    print_gen_fn = get_print_gen_fn(robot, obstacles, node_points, element_bodies, ground_nodes,
                                    precompute_collisions=False, max_attempts=500, **kwargs)
    heuristic_fn = get_heuristic_fn(extrusion_path, heuristic, checker=checker, forward=False)
    # TODO: compute the heuristic function once and fix

    queue = []
    visited = {}
    def add_successors(printed):
        for element in sorted(printed, key=lambda e: -1.0 * get_z(node_points, e)):
            num_remaining = len(printed) - 1
            assert 0 <= num_remaining
            bias = heuristic_fn(printed, element)
            priority = (num_remaining, bias, random.random())
            heapq.heappush(queue, (priority, printed, element))

    initial_printed = frozenset(element_bodies)
    if not check_connected(ground_nodes, initial_printed) or \
            not test_stiffness(extrusion_path, element_from_id, initial_printed, checker=checker):
        data = {
            'sequence': None,
            'runtime': elapsed_time(start_time),
        }
        return None, data
    visited[initial_printed] = Node(None, None)
    add_successors(initial_printed)

    if log:
        log_data = OrderedDict()
        log_data['search_method'] = 'regression'
        log_data['heuristic'] = heuristic
        log_data['stiffness_criteria'] = stiffness_criteria
        log_data['max_time'] = max_time
        log_data['max_backtrack'] = max_backtrack if abs(max_backtrack) != INF else 'inf'
        log_data['total_e_num'] = len(initial_printed)
        log_data['search_log'] = []
        if 'fixed' in heuristic:
            log_data['precomputed_heuristic'] = {'criteria' : stiffness_criteria}
            cache = {}
            for e in initial_printed:
                cache[id_from_element[e]] = heuristic_fn(initial_printed, e)
                print('E#{} : {} - score {}'.format(id_from_element[e], e, cache[id_from_element[e]]))
            # sorted_ids = sorted(cache, key=lambda e_id: cache[e_id])
            # log_data['precomputed_heuristic']['cache'] = OrderedDict({e_id : cache[e_id] for e_id in sorted_ids})
            log_data['precomputed_heuristic']['cache'] = cache
    
    # TODO: lazy hill climbing using the true stiffness heuristic
    # Choose the first move that improves the score
    if has_gui():
        sequence = sorted(initial_printed, key=lambda e: heuristic_fn(initial_printed, e), reverse=True)
        remove_all_debug()
        draw_ordered(sequence, node_points)
        wait_for_user()
    # TODO: fixed branching factor
    # TODO: be more careful when near the end
    # TODO: max time spent evaluating successors (less expensive when few left)
    # TODO: tree rollouts
    # TODO: best-first search with a minimizing path distance cost
    # TODO: immediately select if becomes more stable
    # TODO: focus branching factor on most stable regions

    plan = None
    min_remaining = INF
    num_evaluated = 0
    while queue and (elapsed_time(start_time) < max_time):
        priority, printed, element = heapq.heappop(queue)
        num_remaining = len(printed)
        backtrack = num_remaining - min_remaining
        if max_backtrack <= backtrack:
            continue
        num_evaluated += 1

        print('Iteration: {} | Best: {} | Printed: {} | Element: {} | Index: {} | Time: {:.3f}'.format(
            num_evaluated, min_remaining, len(printed), element, id_from_element[element], elapsed_time(start_time)))
        next_printed = printed - {element}
        #draw_action(node_points, next_printed, element)
        #if 3 < backtrack + 1:
        #    remove_all_debug()
        #    set_renderer(enable=True)
        #    draw_model(next_printed, node_points, ground_nodes)
        #    wait_for_user()

        if (next_printed in visited) or not check_connected(ground_nodes, next_printed) or \
                not implies(stiffness, test_stiffness(extrusion_path, element_from_id, next_printed, checker=checker)):
            continue
        # TODO: could do this eagerly to inspect the full branching factor
        command = sample_extrusion(print_gen_fn, ground_nodes, next_printed, element)
        if command is None:
            continue

        if num_remaining < min_remaining:
            min_remaining = num_remaining
            print('New best: {}'.format(num_remaining))
            #if has_gui():
            #    # TODO: change link transparency
            #    remove_all_debug()
            #    draw_model(next_printed, node_points, ground_nodes)
            #    wait_for_duration(0.5)

        visited[next_printed] = Node(command, printed) # TODO: be careful when multiple trajs

        queue_log_cnt = 200
        if log:
            cur_data = {}
            cur_data['iter'] = num_evaluated - 1
            cur_data['min_remain'] = min_remaining
            cur_data['backtrack'] = backtrack+1 if abs(backtrack) != INF else 0
            cur_data['chosen_id'] = id_from_element[element]
            cur_data['total_q_len'] = len(queue)
            cur_data['queue'] = []
            cur_data['queue'].append((id_from_element[element], priority))
            for candidate in heapq.nsmallest(queue_log_cnt, queue):
                cur_data['queue'].append((id_from_element[candidate[2]], candidate[0]))
            log_data['search_log'].append(cur_data)

        if not next_printed:
            min_remaining = 0
            plan = list(reversed(retrace_trajectories(visited, next_printed)))
            break
        add_successors(next_printed)

    if log:
        export_log_data(extrusion_path, log_data)

    # TODO: store maximum stiffness violations (for speed purposes)
    sequence = None
    if plan is not None:
        sequence = [traj.directed_element for traj in plan if isinstance(traj, PrintTrajectory)]
    data = {
        'sequence': sequence,
        'runtime': elapsed_time(start_time),
        'num_evaluated': num_evaluated,
        'num_remaining': min_remaining,
        'num_elements': len(element_bodies)
    }
    return plan, data

GREEDY_ALGORITHMS = [
    progression.__name__,
    regression.__name__,
]