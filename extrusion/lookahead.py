from __future__ import print_function

import heapq
import time
import numpy as np
from collections import defaultdict

from extrusion.validator import compute_plan_deformation
from extrusion.greedy import Node, retrace_trajectories, add_successors, compute_printed_nodes, \
    recover_directed_sequence, sample_extrusion, recover_sequence
from extrusion.heuristics import get_heuristic_fn
from extrusion.parsing import load_extrusion
from extrusion.stream import get_print_gen_fn
from extrusion.utils import check_connected, test_stiffness, \
    create_stiffness_checker, get_id_from_element, PrintTrajectory, JOINT_WEIGHTS
from extrusion.visualization import color_structure
from extrusion.motion import compute_motion
# https://github.com/yijiangh/conmech/blob/master/src/bindings/pyconmech/pyconmech.cpp
from pybullet_tools.utils import INF, has_gui, elapsed_time, LockRenderer, randomize, \
    wait_for_user, get_movable_joints, get_joint_positions, get_distance_fn

def retrace_elements(visited, current_state):
    return [traj.element for traj in retrace_trajectories(visited, current_state)
            if isinstance(traj, PrintTrajectory)]

##################################################

def get_sample_traj(elements, print_gen_fn, max_extrusions=INF):
    gen_from_element = {element: print_gen_fn(node1=None, element=element, extruded=[], trajectories=[])
                        for element in elements}
    trajs_from_element = defaultdict(list)

    def enumerate_extrusions(printed, element):
        for traj in trajs_from_element[element]:
            yield traj
        if max_extrusions <= len(trajs_from_element[element]):
            return
        with LockRenderer():
            #generator = gen_from_element[element]
            generator = print_gen_fn(node1=None, element=element, extruded=printed,
                                     trajectories=trajs_from_element[element])
            for traj, in generator: # TODO: islice for the num to sample
                trajs_from_element[element].append(traj)
                yield traj
            #for _ in range(100):
            #    traj, = next(print_gen_fn(None, element, extruded=[]), (None,))


    def sample_traj(printed, element, num=1):
        # TODO: other num conditions: max time, min collisions, etc
        assert 1 <= num
        safe_trajectories = []
        for traj in enumerate_extrusions(printed, element):
            # TODO: lazy collision checking
            if not (traj.colliding & printed):
                safe_trajectories.append(traj)
            if num <= len(safe_trajectories):
                break
        return safe_trajectories

    return sample_traj, trajs_from_element

def topological_sort(robot, obstacles, element_bodies, extrusion_path):
    # TODO: take fewest collision samples and attempt to topological sort
    # Repeat if a cycle is detected
    raise NotImplementedError()

##################################################

def lookahead(robot, obstacles, element_bodies, extrusion_path,
              num_ee=1, num_arm=0, max_directions=500, max_attempts=1,
              plan_all=False, use_conficts=False, use_replan=False, heuristic='z', max_time=INF, # max_backtrack=INF,
              revisit=False, ee_only=False, collisions=True, stiffness=True, motions=True, **kwargs):
    if ee_only:
        num_ee, num_arm = max(num_arm, num_ee), 0
    if not use_conficts:
        num_arm = 1
    start_time = time.time()
    joints = get_movable_joints(robot)
    initial_conf = get_joint_positions(robot, joints)
    element_from_id, node_points, ground_nodes = load_extrusion(extrusion_path)
    checker = create_stiffness_checker(extrusion_path, verbose=False)
    #checker = None

    #print_gen_fn = get_print_gen_fn(robot, obstacles, node_points, element_bodies, ground_nodes,
    #                                precompute_collisions=False, supports=False, bidirectional=False, ee_only=ee_only,
    #                                max_directions=500, max_attempts=1, collisions=collisions, **kwargs)
    full_print_gen_fn = get_print_gen_fn(robot, obstacles, node_points, element_bodies, ground_nodes,
                                         precompute_collisions=True, supports=False, bidirectional=True, ee_only=ee_only,
                                         max_directions=max_directions, max_attempts=max_attempts, collisions=collisions, **kwargs)
    # TODO: could just check kinematics instead of collision
    ee_print_gen_fn = get_print_gen_fn(robot, obstacles, node_points, element_bodies, ground_nodes,
                                        precompute_collisions=True, supports=False, bidirectional=True, ee_only=True,
                                        max_directions=max_directions, max_attempts=max_attempts, collisions=collisions, **kwargs)
    id_from_element = get_id_from_element(element_from_id)
    all_elements = frozenset(element_bodies)
    heuristic_fn = get_heuristic_fn(extrusion_path, heuristic, checker=checker, forward=True)
    distance_fn = get_distance_fn(robot, joints, weights=JOINT_WEIGHTS)
    # TODO: 2-step lookahead based on neighbors or spatial proximity

    full_sample_traj, full_trajs_from_element = get_sample_traj(all_elements, full_print_gen_fn)
    ee_sample_traj, ee_trajs_from_element = get_sample_traj(all_elements, ee_print_gen_fn)
    if ee_only:
        full_sample_traj = ee_sample_traj
    #ee_sample_traj, ee_trajs_from_element = full_sample_traj, full_trajs_from_element

    #heuristic_trajs_from_element = full_trajs_from_element if (num_ee == 0) else ee_trajs_from_element
    heuristic_trajs_from_element = full_trajs_from_element if (num_arm != 0) else ee_trajs_from_element

    #########################

    def sample_remaining(printed, sample_fn, num=1, **kwargs):
        if num == 0:
            return True
        nodes = compute_printed_nodes(ground_nodes, printed)
        if plan_all:
            elements = all_elements - printed
        else:
            elements = [element for element in all_elements - printed if any(n in nodes for n in element)]
        return all(sample_fn(printed, element, num, **kwargs) for element in randomize(elements))

    def conflict_fn(printed, element, conf):
        # TODO: could add element if desired
        #return np.random.random()
        #return 0 # Dead-end detection without stability performs reasonably well
        order = retrace_elements(visited, printed)
        printed = frozenset(order[:-1]) # Remove last element (to ensure at least one traj)
        if use_replan:
            remaining = list(all_elements - printed)
            requires_replan = [all(element in traj.colliding for traj in ee_trajs_from_element[e2]
                                  if not (traj.colliding & printed)) for e2 in remaining if e2 != element]
            return len(requires_replan)
        else:
            safe_trajectories = [traj for traj in heuristic_trajs_from_element[element] if not (traj.colliding & printed)]
            assert safe_trajectories
            best_traj = max(safe_trajectories, key=lambda traj: len(traj.colliding))
            num_colliding = len(best_traj.colliding)
            return -num_colliding
        #distance = distance_fn(conf, best_traj.start_conf)
        # TODO: ee distance vs conf distance
        # TODO: l0 distance based on whether we remain at the same node
        # TODO: minimize instability while printing (dynamic programming)
        #return (-num_colliding, distance)

    if use_conficts:
        priority_fn = lambda *args: (conflict_fn(*args), heuristic_fn(*args))
    else:
        priority_fn = heuristic_fn

    #########################

    initial_printed = frozenset()
    queue = []
    visited = {initial_printed: Node(None, None)}
    if check_connected(ground_nodes, all_elements) and test_stiffness(extrusion_path, element_from_id, all_elements) and \
            sample_remaining(initial_printed, ee_sample_traj, num=num_ee) and sample_remaining(initial_printed, full_sample_traj, num=num_arm):
        add_successors(queue, all_elements, node_points, ground_nodes, priority_fn, initial_printed, initial_conf)

    plan = None
    min_remaining = INF
    num_evaluated = worst_backtrack = num_deadends = 0
    while queue and (elapsed_time(start_time) < max_time):
        num_evaluated += 1
        visits, priority, printed, element, current_conf = heapq.heappop(queue)
        num_remaining = len(all_elements) - len(printed)
        backtrack = num_remaining - min_remaining
        #if max_backtrack < backtrack: # backtrack_bound
        #    continue
        worst_backtrack = max(worst_backtrack, backtrack)
        num_evaluated += 1
        if num_remaining < min_remaining:
            min_remaining = num_remaining
        print('Iteration: {} | Best: {} | Backtrack: {} | Deadends: {} | Printed: {} | Element: {} | Index: {} | Time: {:.3f}'.format(
            num_evaluated, min_remaining, worst_backtrack, num_deadends, len(printed), element, id_from_element[element], elapsed_time(start_time)))
        printed_nodes = compute_printed_nodes(ground_nodes, printed)
        if has_gui():
            color_structure(element_bodies, printed, element)

        next_printed = printed | {element}
        if (next_printed in visited) or not check_connected(ground_nodes, next_printed) or \
                (stiffness and not test_stiffness(extrusion_path, element_from_id, next_printed, checker=checker)):
            # Hard dead-end
            #num_deadends += 1
            continue

        # TODO: the directionality actually matters for the printing orientation
        if not sample_remaining(next_printed, ee_sample_traj, num=num_ee):
            # Soft dead-end
            num_deadends += 1
            #wait_for_user()
            continue

        #command = sample_extrusion(print_gen_fn, ground_nodes, printed, element)
        command = next(iter(full_sample_traj(printed, element)), None)
        if command is None:
            # Soft dead-end
            #num_deadends += 1
            continue
        print_traj = command.print_trajectory
        assert any(n in printed_nodes for n in print_traj.element)
        if print_traj.n1 not in printed_nodes:
            command = command.reverse()

        if not sample_remaining(next_printed, full_sample_traj, num=num_arm):
            # Soft dead-end
            num_deadends += 1
            continue

        start_conf = end_conf = None
        if not ee_only:
            start_conf, end_conf = command.start_conf, command.end_conf
        if (start_conf is not None) and motions:
            motion_traj = compute_motion(robot, obstacles, element_bodies,
                                         printed, current_conf, start_conf, collisions=collisions)
            if motion_traj is None:
                continue
            command.trajectories.insert(0, motion_traj)

        visited[next_printed] = Node(command, printed)
        if all_elements <= next_printed:
            # TODO: anytime mode
            min_remaining = 0
            plan = retrace_trajectories(visited, next_printed)
            break
        add_successors(queue, all_elements, node_points, ground_nodes, priority_fn, next_printed, end_conf)
        if revisit:
            heapq.heappush(queue, (visits + 1, priority, printed, element, current_conf))

    max_translation, max_rotation = compute_plan_deformation(extrusion_path, recover_sequence(plan))
    data = {
        'sequence': recover_directed_sequence(plan),
        'runtime': elapsed_time(start_time),
        'num_elements': len(all_elements),
        'num_evaluated': num_evaluated,
        'min_remaining': min_remaining,
        'max_backtrack': worst_backtrack,
        'max_translation': max_translation,
        'max_rotation': max_rotation,
    }
    return plan, data
