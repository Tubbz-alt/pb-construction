import numpy as np

from examples.pybullet.utils.pybullet_tools.utils import get_movable_joints, get_joint_positions, multiply, invert, \
    set_joint_positions, inverse_kinematics, get_link_pose, get_distance, point_from_pose, wrap_angle, get_sample_fn, \
    link_from_name, get_pose, get_collision_fn
from extrusion.extrusion_utils import get_grasp_pose, TOOL_NAME, get_disabled_collisions, get_node_neighbors, \
    sample_direction, check_trajectory_collision, PrintTrajectory, retrace_supporters, get_supported_orders
#from extrusion.run import USE_IKFAST, get_supported_orders, retrace_supporters, SELF_COLLISIONS, USE_CONMECH
from pddlstream.language.stream import WildOutput
from pddlstream.utils import neighbors_from_orders, irange, user_input

try:
    from utils.ikfast.kuka_kr6_r900.ik import sample_tool_ik
except ImportError as e:
    print('\x1b[6;30;43m' + '{}, Using pybullet ik fn instead'.format(e) + '\x1b[0m')
    USE_IKFAST = False
    user_input("Press Enter to continue...")
else:
    USE_IKFAST = True

USE_CONMECH = True
try:
    import conmech_py
except ImportError as e:
    print('\x1b[6;30;43m' + '{}, Not using conmech'.format(e) + '\x1b[0m')
    USE_CONMECH = False

SELF_COLLISIONS = True

def optimize_angle(robot, link, element_pose, translation, direction, reverse, initial_angles,
                   collision_fn, max_error=1e-2):
    movable_joints = get_movable_joints(robot)
    best_error, best_angle, best_conf = max_error, None, None
    initial_conf = get_joint_positions(robot, movable_joints)
    for i, angle in enumerate(initial_angles):
        grasp_pose = get_grasp_pose(translation, direction, angle, reverse)
        # Pose_{world,EE} = Pose_{world,element} * Pose_{element,EE}
        #                 = Pose_{world,element} * (Pose_{EE,element})^{-1}
        target_pose = multiply(element_pose, invert(grasp_pose))
        set_joint_positions(robot, movable_joints, initial_conf)

        if USE_IKFAST:
            # TODO: randomly sample the initial configuration
            bias_conf = initial_conf
            #bias_conf = None # Randomizes solutions
            conf = sample_tool_ik(robot, target_pose, nearby_conf=bias_conf)
        else:
            # note that the conf get assigned inside this ik fn right away!
            conf = inverse_kinematics(robot, link, target_pose)
        if conf is None: # TODO(caelan): this is suspect
            conf = get_joint_positions(robot, movable_joints)
        #if pairwise_collision(robot, robot):

        if not collision_fn(conf):
            link_pose = get_link_pose(robot, link)
            error = get_distance(point_from_pose(target_pose), point_from_pose(link_pose))
            if error < best_error:  # TODO: error a function of direction as well
                best_error, best_angle, best_conf = error, angle, conf
            # wait_for_interrupt()
    #print(best_error, translation, direction, best_angle)
    if best_conf is not None:
        set_joint_positions(robot, movable_joints, best_conf)
        #wait_for_interrupt()
    return best_angle, best_conf

##################################################

def compute_direction_path(robot, length, reverse, element_body, direction, collision_fn):
    """
    :param robot:
    :param length: element's length
    :param reverse: True if element end id tuple needs to be reversed
    :param element_body: the considered element's pybullet body
    :param direction: a sampled Pose (v \in unit sphere)
    :param collision_fn: collision checker (pybullet_tools.utils.get_collision_fn)
    note that all the static objs + elements in the support set of the considered element
    are accounted in the collision fn
    :return: feasible PrintTrajectory if found, None otherwise
    """
    step_size = 0.0025 # 0.005
    #angle_step_size = np.pi / 128
    angle_step_size = np.math.radians(0.25)
    angle_deltas = [-angle_step_size, 0, angle_step_size]
    #num_initial = 12
    num_initial = 1

    steps = np.append(np.arange(-length / 2, length / 2, step_size), [length / 2])
    #print('Length: {} | Steps: {}'.format(length, len(steps)))

    #initial_angles = [wrap_angle(angle) for angle in np.linspace(0, 2*np.pi, num_initial, endpoint=False)]
    initial_angles = [wrap_angle(angle) for angle in np.random.uniform(0, 2*np.pi, num_initial)]
    movable_joints = get_movable_joints(robot)

    if not USE_IKFAST:
        # randomly sample and set joint conf for the pybullet ik fn
        sample_fn = get_sample_fn(robot, movable_joints)
        set_joint_positions(robot, movable_joints, sample_fn())
    link = link_from_name(robot, TOOL_NAME)
    element_pose = get_pose(element_body)
    current_angle, current_conf = optimize_angle(robot, link, element_pose,
                                                 steps[0], direction, reverse, initial_angles, collision_fn)
    if current_conf is None:
        return None
    # TODO: constrain maximum conf displacement
    # TODO: alternating minimization for just position and also orientation
    trajectory = [current_conf]
    for translation in steps[1:]:
        #set_joint_positions(robot, movable_joints, current_conf)
        initial_angles = [wrap_angle(current_angle + delta) for delta in angle_deltas]
        current_angle, current_conf = optimize_angle(
            robot, link, element_pose, translation, direction, reverse, initial_angles, collision_fn)
        if current_conf is None:
            return None
        trajectory.append(current_conf)
    return trajectory

##################################################

def get_print_gen_fn(robot, fixed_obstacles, node_points, element_bodies, ground_nodes):
    max_attempts = 300 # 150 | 300
    max_trajectories = 10
    check_collisions = True
    # 50 doesn't seem to be enough

    movable_joints = get_movable_joints(robot)
    disabled_collisions = get_disabled_collisions(robot)
    #element_neighbors = get_element_neighbors(element_bodies)
    node_neighbors = get_node_neighbors(element_bodies)
    incoming_supporters, _ = neighbors_from_orders(get_supported_orders(element_bodies, node_points))
    # TODO: print on full sphere and just check for collisions with the printed element
    # TODO: can slide a component of the element down
    # TODO: prioritize choices that don't collide with too many edges

    def gen_fn(node1, element): # fluents=[]):
        reverse = (node1 != element[0])
        element_body = element_bodies[element]
        n1, n2 = reversed(element) if reverse else element
        delta = node_points[n2] - node_points[n1]
        # if delta[2] < 0:
        #    continue
        length = np.linalg.norm(delta)  # 5cm

        #supporters = {e for e in node_neighbors[n1] if element_supports(e, n1, node_points)}
        supporters = []
        retrace_supporters(element, incoming_supporters, supporters)
        elements_order = [e for e in element_bodies if (e != element) and (e not in supporters)]
        bodies_order = [element_bodies[e] for e in elements_order]
        obstacles = fixed_obstacles + [element_bodies[e] for e in supporters]
        collision_fn = get_collision_fn(robot, movable_joints, obstacles,
                                        attachments=[], self_collisions=SELF_COLLISIONS,
                                        disabled_collisions=disabled_collisions,
                                        custom_limits={}) # TODO: get_custom_limits
        trajectories = []
        for num in irange(max_trajectories):
            for attempt in range(max_attempts):
                path = compute_direction_path(robot, length, reverse, element_body, sample_direction(), collision_fn)
                if path is None:
                    continue
                if check_collisions:
                    collisions = check_trajectory_collision(robot, path, bodies_order)
                    colliding = {e for k, e in enumerate(elements_order) if (element != e) and collisions[k]}
                else:
                    colliding = set()
                if (node_neighbors[n1] <= colliding) and not any(n in ground_nodes for n in element):
                    continue
                print_traj = PrintTrajectory(robot, movable_joints, path, element, reverse, colliding)
                trajectories.append(print_traj)
                # TODO: need to prune dominated trajectories
                if print_traj not in trajectories:
                    continue
                print('{}) {}->{} ({}) | {} | {} | {}'.format(
                    num, n1, n2, len(supporters), attempt, len(trajectories),
                    sorted(len(t.colliding) for t in trajectories)))
                yield (print_traj,)
                if not colliding:
                    return
            else:
                print('{}) {}->{} ({}) | {} | Max attempts exceeded!'.format(
                    num, len(supporters), n1, n2, max_attempts))
                user_input('Continue?')
                return
    return gen_fn

##################################################

def get_wild_print_gen_fn(robot, obstacles, node_points, element_bodies, ground_nodes, collisions=True):
    gen_fn = get_print_gen_fn(robot, obstacles, node_points, element_bodies, ground_nodes)
    def wild_gen_fn(node1, element):
        for t, in gen_fn(node1, element):
            outputs = [(t,)]
            facts = [('Collision', t, e2) for e2 in t.colliding] if collisions else []
            yield WildOutput(outputs, facts)
    return wild_gen_fn


def test_stiffness(fluents=[]):
    assert all(fact[0] == 'printed' for fact in fluents)
    if not USE_CONMECH:
       return True
    # https://github.com/yijiangh/conmech
    # TODO: to use the non-skeleton focused algorithm, need to remove the negative axiom upon success
    import conmech_py
    elements = {fact[1] for fact in fluents}
    #print(elements)
    return True