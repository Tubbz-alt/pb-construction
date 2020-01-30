from __future__ import print_function

from pddlstream.utils import incoming_from_edges
from pybullet_tools.utils import connect, ClientSaver, wait_for_user, INF, has_gui, remove_all_debug, \
    get_movable_joints, get_joint_positions
from extrusion.logger import export_log_data, RECORD_BT, RECORD_CONSTRAINT_VIOLATION, RECORD_QUEUE, OVERWRITE, VISUALIZE_ACTION, CHECK_BACKTRACK, QUEUE_COUNT, PAUSE_UPON_BT, MAX_STATES_STORED
from extrusion.motion import compute_motion, compute_motions
from extrusion.stiffness import TRANS_TOL, ROT_TOL, create_stiffness_checker, test_stiffness
from extrusion.utils import check_connected, get_id_from_element, load_world, PrintTrajectory, \
    compute_printed_nodes, compute_printable_elements, get_memory_in_kb, check_memory
from extrusion.stream import get_print_gen_fn, STEP_SIZE, APPROACH_DISTANCE, MAX_DIRECTIONS, MAX_ATTEMPTS
from extrusion.visualization import draw_element, color_structure
from extrusion.parsing import load_extrusion
from pybullet_tools.utils import elapsed_time, \
    LockRenderer, reset_simulation, disconnect, randomize
from extrusion.heuristics import get_heuristic_fn, score_stiffness
from extrusion.validator import compute_plan_deformation
from collections import namedtuple
import heapq
import random
import time
from termcolor import cprint

# https://github.com/ContinuumIO/anaconda-issues/issues/905
import os
# os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = 'T'
env = 'FOR_DISABLE_CONSOLE_CTRL_HANDLER'
if env not in os.environ:
    os.environ[env] = '1'

#State = namedtuple('State', ['element', 'printed', 'plan'])
Node = namedtuple('Node', ['action', 'state'])


##################################################


def retrace_trajectories(visited, current_state, horizon=INF, reverse=False):
    command, prev_state = visited[current_state]
    if (prev_state is None) or (horizon == 0):
        return []
    prior_trajectories = retrace_trajectories(
        visited, prev_state, horizon=horizon-1, reverse=reverse)
    current_trajectories = [traj for traj in command.trajectories]
    if reverse:
        return current_trajectories + prior_trajectories
    return prior_trajectories + current_trajectories
    # TODO: search over local stability for each node


def recover_sequence(plan):
    if plan is None:
        return plan
    return [traj.element for traj in plan if isinstance(traj, PrintTrajectory)]


def recover_directed_sequence(plan):
    if plan is None:
        return plan
    return [traj.directed_element for traj in plan if isinstance(traj, PrintTrajectory)]

##################################################


def sample_extrusion(print_gen_fn, ground_nodes, printed, element):
    printed_nodes = compute_printed_nodes(ground_nodes, printed)
    for node in element:
        # TODO: sample between different orientations if both are feasible
        if node in printed_nodes:
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
        handles.extend(draw_element(node_points, e, color=(0, 1, 0))
                       for e in printed)
    wait_for_user()
    return handles

##################################################


def add_successors(queue, all_elements, node_points, ground_nodes, heuristic_fn, printed, conf,
                   partial_orders=[], visualize=False):
    incoming_from_element = incoming_from_edges(partial_orders)
    remaining = all_elements - printed
    num_remaining = len(remaining) - 1
    #assert 0 <= num_remaining
    bias_from_element = {}
    for element in randomize(compute_printable_elements(all_elements, ground_nodes, printed)):
        if not (incoming_from_element[element] <= printed):
            continue
        bias = heuristic_fn(printed, element, conf)
        priority = (num_remaining, bias, random.random())
        visits = 0
        heapq.heappush(queue, (visits, priority, printed, element, conf))
        bias_from_element[element] = bias

    # if visualize and has_gui():
    #     handles = []
    #     with LockRenderer():
    #         remove_all_debug()
    #         for element in printed:
    #             handles.append(draw_element(node_points, element, color=(0, 0, 0)))
    #         successors = sorted(bias_from_element, key=lambda e: bias_from_element[e])
    #         handles.extend(draw_ordered(successors, node_points))
    #     print('Min: {:.3E} | Max: {:.3E}'.format(bias_from_element[successors[0]], bias_from_element[successors[-1]]))
    #     wait_for_user()


def progression(robot, obstacles, element_bodies, extrusion_path, partial_orders=[],
                heuristic='z', max_time=INF, backtrack_limit=INF,
                stiffness=True, motions=True, collisions=True, lazy=True, **kwargs):

    start_time = time.time()
    joints = get_movable_joints(robot)
    initial_conf = get_joint_positions(robot, joints)
    element_from_id, node_points, ground_nodes = load_extrusion(extrusion_path)
    checker = create_stiffness_checker(extrusion_path, verbose=False)
    #checker = None
    print_gen_fn = get_print_gen_fn(robot, obstacles, node_points, element_bodies, ground_nodes,
                                    supports=False, precompute_collisions=False,
                                    max_directions=MAX_DIRECTIONS, max_attempts=MAX_ATTEMPTS,
                                    collisions=collisions, **kwargs)
    id_from_element = get_id_from_element(element_from_id)
    all_elements = frozenset(element_bodies)
    heuristic_fn = get_heuristic_fn(
        extrusion_path, heuristic, checker=checker, forward=True)

    initial_printed = frozenset()
    queue = []
    visited = {initial_printed: Node(None, None)}
    if check_connected(ground_nodes, all_elements) and \
            test_stiffness(extrusion_path, element_from_id, all_elements):
        add_successors(queue, all_elements, node_points, ground_nodes, heuristic_fn, initial_printed, initial_conf, partial_orders=partial_orders)
    else:
        cprint('Grounded nodes are not connected to any of the elements or the whole structure is not stiff!', 'red')

    plan = None
    min_remaining = len(all_elements)
    num_evaluated = max_backtrack = num_deadends = stiffness_failures = extrusion_failures= transit_failures = 0

    #############################################
    locker = LockRenderer()
    bt_data = []  # backtrack history
    cons_data = []  # constraint violation history
    queue_data = []  # queue candidates history

    def snapshot_state(data_list=None, reason='', queue_log_cnt=0):
        """a lot of global parameters are used

        """
        cur_data = {}
        cur_data['num_evaluated'] = num_evaluated # iter
        cur_data['reason'] = reason
        cur_data['elapsed_time'] = elapsed_time(start_time)
        cur_data['min_remaining'] = min_remaining
        cur_data['max_backtrack'] = max_backtrack

        cur_data['extrusion_failures'] = extrusion_failures
        cur_data['stiffness_failures'] = stiffness_failures
        cur_data['transit_failures'] = transit_failures

        cur_data['backtrack'] = backtrack
        cur_data['total_q_len'] = len(queue)

        cur_data['chosen_element'] = element
        record_plan = retrace_trajectories(visited, printed)
        planned_elements = recover_directed_sequence(record_plan)
        cur_data['planned_elements'] = planned_elements
        cur_data['queue'] = []

        print('++++++++++++')
        if queue_log_cnt > 0:
            top_candidates = [(visits, priority, printed, element)] + list(heapq.nsmallest(queue_log_cnt, queue))
            for candidate in top_candidates:
                # * for progression
                temporal_chosen_element = candidate[3]
                temp_visits, temp_priority = candidate[0], candidate[1]
                temporal_structure = printed | {temporal_chosen_element}
                if len(temporal_structure) == len(printed):
                    continue

                stiffness_score = score_stiffness(
                    extrusion_path, element_from_id, temporal_structure, checker=checker)
                temp_command = sample_extrusion(
                    print_gen_fn, ground_nodes, printed, temporal_chosen_element)
                extrusion_feasible = 0 if temp_command is None else 1
                # lower is better
                print('cand: {}, compl: {}, feas: {}'.format(
                    temporal_chosen_element, stiffness_score, extrusion_feasible))
                cur_data['queue'].append(
                    (list(temporal_chosen_element), stiffness_score, extrusion_feasible, temp_visits, temp_priority))
        print('++++++++++++')

        if data_list is not None and len(data_list) < MAX_STATES_STORED:
            data_list.append(cur_data)

        if CHECK_BACKTRACK:
            draw_action(node_points, next_printed, element)
            # color_structure(element_bodies, next_printed, element)

            # TODO: can take picture here
            locker.restore()
            cprint('{} detected, press Enter to continue!'.format(reason), 'red')
            wait_for_user()
            locker = LockRenderer()
        return cur_data
    # end snapshot
    #############################################

    try:
        while queue:
            if elapsed_time(start_time) > max_time and check_memory(): #max_memory):
                if elapsed_time(start_time) < max_time:
                    cprint('memory leak: {} | {} '.format(check_memory(), get_memory_in_kb()))
                raise TimeoutError
            visits, priority, printed, element, current_conf = heapq.heappop(queue)
            num_remaining = len(all_elements) - len(printed)
            num_evaluated += 1
            print('-'*5)
            if num_remaining < min_remaining:
                min_remaining = num_remaining
                cprint('New best: {}/{}'.format(num_remaining,
                                                len(all_elements)), 'green')

            cprint('Eval Iter: {} | Best: {}/{} | Backtrack: {} | Printed: {} | Element: {} | E-Id: {} | Time: {:.3f}'.format(
                num_evaluated, min_remaining, len(all_elements), max_backtrack, len(printed), element, id_from_element[element], elapsed_time(start_time)))
            next_printed = printed | {element}

            backtrack = num_remaining - min_remaining
            if backtrack > max_backtrack:
                max_backtrack = backtrack
                # * (optional) visualization for diagnosis
                if RECORD_BT:
                    cprint('max backtrack increased to {}'.format(
                        max_backtrack), 'cyan')
                    snapshot_state(bt_data, reason='Backtrack')
                    if PAUSE_UPON_BT: wait_for_user()

            if backtrack_limit < backtrack:
                cprint('backtrack {} exceeds limit {}, exit.'.format(
                    backtrack, backtrack_limit), 'red')
                raise KeyboardInterrupt
                # break  # continue

            if RECORD_QUEUE:
                snapshot_state(
                    queue_data, reason='queue_history', queue_log_cnt=QUEUE_COUNT)

            # * constraint checking
            # ! connectivity and avoid checking duplicate states
            if (next_printed in visited) or not check_connected(ground_nodes, next_printed):
                continue

            # ! stiffness constraint
            if stiffness and not test_stiffness(extrusion_path, element_from_id, next_printed, checker=checker):
                cprint('&&& stiffness not passed.', 'red')
                stiffness_failures += 1
                # * (optional) visualization for diagnosis
                if RECORD_CONSTRAINT_VIOLATION:
                    snapshot_state(cons_data, reason='stiffness_violation')
                continue

            # ! manipulation constraint
            command = sample_extrusion(
                print_gen_fn, ground_nodes, printed, element)
            if command is None:
                extrusion_failures += 1
                continue
            # ! transition motion constraint
            if motions and not lazy:
                motion_traj = compute_motion(robot, obstacles, element_bodies, printed,
                                             command.end_conf, current_conf, collisions=collisions,
                                             max_time=max_time - elapsed_time(start_time))
                if motion_traj is None:
                    cprint('>>> transition motion not passed.', 'red')
                    transit_failures += 1
                    if RECORD_CONSTRAINT_VIOLATION:
                        snapshot_state(cons_data, reason='transit_failure')
                    continue
                command.trajectories.insert(0, motion_traj)

            visited[next_printed] = Node(command, printed)
            if all_elements <= next_printed:
                min_remaining = 0
                plan = retrace_trajectories(visited, next_printed)
                if motions and not lazy:
                    # transit back to the initial conf
                    motion_traj = compute_motion(robot, obstacles, element_bodies, frozenset(),
                                                 initial_conf, plan[0].start_conf, collisions=collisions,
                                                 max_time=max_time - elapsed_time(start_time))
                    if motion_traj is None:
                        plan = None
                    else:
                        plan.append(motion_traj)
                if motions and lazy:
                    # laziness: try to find all transition plan after a plan has been found
                    plan = compute_motions(robot, obstacles, element_bodies, initial_conf, plan,
                                           collisions=collisions, max_time=max_time - elapsed_time(start_time))
                if plan is not None:
                    break
                else:
                    # backtrack
                    transit_failures += 1
            add_successors(queue, all_elements, node_points, ground_nodes, heuristic_fn, next_printed, command.end_conf, partial_orders=partial_orders, visualize=VISUALIZE_ACTION)

    except (KeyboardInterrupt, TimeoutError):
        # log data
        cur_data = {}
        cur_data['algorithm'] = 'progression'
        cur_data['heuristic'] = heuristic
        when_stop_data = snapshot_state(reason='external stop')

        cur_data['when_stopped'] = when_stop_data
        cur_data['backtrack_history'] = bt_data
        cur_data['constraint_violation_history'] = cons_data
        cur_data['queue_history'] = queue_data

        export_log_data(extrusion_path, cur_data, overwrite=OVERWRITE, **kwargs)

        cprint('search terminated by user interruption or timeout.', 'red')
        if has_gui():
            color_structure(element_bodies, printed, element)
            locker.restore()
            wait_for_user()
        # assert False, 'search terminated.'

    if RECORD_QUEUE | RECORD_CONSTRAINT_VIOLATION | RECORD_BT:
        # log data even if a plan has been found
        cur_data = {}
        cur_data['algorithm'] = 'progression'
        cur_data['heuristic'] = heuristic
        when_stop_data = snapshot_state(reason='plan found')

        cur_data['when_stopped'] = when_stop_data
        cur_data['backtrack_history'] = bt_data
        cur_data['constraint_violation_history'] = cons_data
        cur_data['queue_history'] = queue_data

        export_log_data(extrusion_path, cur_data, overwrite=OVERWRITE, **kwargs)

    max_translation, max_rotation, max_compliance = compute_plan_deformation(extrusion_path, recover_sequence(plan))
    data = {
        'sequence': recover_directed_sequence(plan),
        'runtime': elapsed_time(start_time),
        'memory': get_memory_in_kb(), # May need to update instead
        'num_evaluated': num_evaluated,
        'min_remaining': min_remaining,
        'max_backtrack': max_backtrack,
        #
        'max_translation': max_translation,
        'max_rotation': max_rotation,
        'max_compliance': max_compliance,
        #
        'stiffness_failures': stiffness_failures,
        'extrusion_failures': extrusion_failures,
        'transit_failures': transit_failures,
        #
        'backtrack_history': bt_data,
        'constraint_violation_history': cons_data,
        'queue_history': queue_data,
    }

    if not data['sequence'] and has_gui():
        color_structure(element_bodies, printed, element)
        locker.restore()
        cprint('No plan found.', 'red')
        wait_for_user()

    return plan, data
