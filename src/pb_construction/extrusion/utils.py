from __future__ import print_function

import os
import numpy as np

from collections import defaultdict, deque, namedtuple

from pyconmech import StiffnessChecker

from pybullet_planning import get_link_pose, BodySaver, set_point, set_joint_positions, \
    Point, load_model, HideOutput, load_pybullet, link_from_name, has_link, joint_from_name, angle_between, get_aabb, get_distance

import pb_construction

# URDF_PATH = 'models/kuka_kr6_r900/urdf/kuka_kr6_r900_extrusion.urdf'
URDF_PATH = 'models/kuka_kr6_r900/urdf/kuka_kr6_r900_extrusion_mit_3-412.urdf'
SRDF_PATH = 'models/kuka_kr6_r900/srdf/kuka_kr6_r900_extrusion_mit_3-412.srdf'
COLLISION_OBJ_DIR = 'somepath'
COLLISION_FILE_PATTERN = '*.obj'

def get_robot_data():
    from compas.robots import RobotModel
    from compas_fab.robots import Robot as RobotClass
    from compas_fab.robots import RobotSemantics

    urdf_filepath = pb_construction.get_data(URDF_PATH)
    srdf_filepath = pb_construction.get_data(SRDF_PATH)

    model = RobotModel.from_urdf_file(urdf_filepath)
    semantics = RobotSemantics.from_srdf_file(srdf_filepath, model)
    robot = RobotClass(model, semantics=semantics)

    base_link_name = robot.get_base_link_name(group='manipulator_ee')
    ee_link_name = robot.get_end_effector_link_name(group='manipulator_ee')
    ik_joint_names = robot.get_configurable_joint_names(group='manipulator_ee')
    disabled_link_names = semantics.get_disabled_collisions()

    return base_link_name, ee_link_name, ik_joint_names, disabled_link_names

BASE_LINK_NAME, EE_LINK_NAME, IK_JOINT_NAMES, DISABLED_LINK_NAMES = get_robot_data()
TOOL_ROOT = 'eef_base_link' # robot_tool0 # TODO: call be derived from SRDF as well
JOINT_WEIGHTS = [0.3078557810844393, 0.443600199302506, 0.23544367607317915,
                 0.03637161028426032, 0.04644626184081511, 0.015054267683041092]

# EXTRA_DISABLED_LINK_NAMES = [
#     ('robot_base_link', 'workspace_objects'),
#     ('robot_link_1', 'workspace_objects'),
#     ('robot_link_3', 'material_feeder_material_feeder'),
# ]

CUSTOM_LIMITS = {
    'robot_joint_a1': (-np.pi/2, np.pi/2),
}
SUPPORT_THETA = np.math.radians(10)  # Support polygon

USE_FLOOR = False


RESOLUTION = 0.005
JOINT_WEIGHTS = np.array([0.3078557810844393, 0.443600199302506, 0.23544367607317915,
                          0.03637161028426032, 0.04644626184081511, 0.015054267683041092])

##################################################

def load_world(use_floor=USE_FLOOR, parse_collision_objects=False):
    root_directory = os.path.dirname(os.path.abspath(__file__))
    obstacles = []
    with HideOutput():
        # robot = load_pybullet(os.path.join(root_directory, KUKA_PATH), fixed_base=True)
        robot = load_pybullet(pb_construction.get_data(URDF_PATH), fixed_base=True)
        lower, _ = get_aabb(robot)
        if use_floor:
            floor = load_model('models/short_floor.urdf')
            obstacles.append(floor)
            set_point(floor, Point(z=lower[2]))
        else:
            floor = None # TODO: make this an empty list of obstacles
        if parse_collision_objects:
            import glob
            from pybullet_planning import create_obj
            obj_file_names = glob.glob(os.path.join(pb_construction.get_data(COLLISION_OBJ_DIR), COLLISION_FILE_PATTERN))
            if obj_file_names:
                collision_objs = [create_obj(os.path.join(pb_construction.get_data(COLLISION_OBJ_DIR), file)) for file in obj_file_names]
                obstacles.extend(collision_objs)
    return obstacles, robot


def prune_dominated(trajectories):
    start_len = len(trajectories)
    for traj1 in list(trajectories):
        if any((traj1 != traj2) and (traj2.colliding <= traj1.colliding)
               for traj2 in trajectories):
            trajectories.remove(traj1)
    return len(trajectories) == start_len

##################################################

def get_node_neighbors(elements):
    node_neighbors = defaultdict(set)
    for e in elements:
        n1, n2 = e
        node_neighbors[n1].add(e)
        node_neighbors[n2].add(e)
    return node_neighbors

def nodes_from_elements(elements):
    # TODO: always include ground nodes
    return {n for e in elements for n in e}

def get_element_neighbors(elements):
    node_neighbors = get_node_neighbors(elements)
    element_neighbors = defaultdict(set)
    for e in elements:
        n1, n2 = e
        element_neighbors[e].update(node_neighbors[n1])
        element_neighbors[e].update(node_neighbors[n2])
        element_neighbors[e].remove(e)
    return element_neighbors

##################################################

def get_disabled_collisions(robot):
    return {tuple(link_from_name(robot, link)
                  for link in pair if has_link(robot, link))
                  for pair in DISABLED_LINK_NAMES}

def get_custom_limits(robot):
    return {joint_from_name(robot, joint): limits
            for joint, limits in CUSTOM_LIMITS.items()}

##################################################

class Trajectory(object):
    def __init__(self, robot, joints, path):
        self.robot = robot
        self.joints = joints
        self.path = path
        self.path_from_link = {}
    def get_link_path(self, link_name=EE_LINK_NAME):
        link = link_from_name(self.robot, link_name)
        if link not in self.path_from_link:
            with BodySaver(self.robot):
                self.path_from_link[link] = []
                for conf in self.path:
                    set_joint_positions(self.robot, self.joints, conf)
                    self.path_from_link[link].append(get_link_pose(self.robot, link))
        return self.path_from_link[link]
    def reverse(self):
        raise NotImplementedError()
    def iterate(self):
        for conf in self.path[1:]:
            set_joint_positions(self.robot, self.joints, conf)
            yield

class MotionTrajectory(Trajectory):
    def __init__(self, robot, joints, path, attachments=[]):
        super(MotionTrajectory, self).__init__(robot, joints, path)
        self.attachments = attachments
    def reverse(self):
        return self.__class__(self.robot, self.joints, self.path[::-1], self.attachments)
    def __repr__(self):
        return 'm(#J {},#pth {})'.format(len(self.joints), len(self.path))

class PrintTrajectory(Trajectory):
    def __init__(self, robot, joints, path, tool_path, element, is_reverse=False):
        super(PrintTrajectory, self).__init__(robot, joints, path)
        self.tool_path = tool_path
        self.is_reverse = is_reverse
        #assert len(self.path) == len(self.tool_path)
        self.element = element
        self.n1, self.n2 = reversed(element) if self.is_reverse else element
    @property
    def directed_element(self):
        return (self.n1, self.n2)
    def get_link_path(self, link_name=EE_LINK_NAME):
        if link_name == EE_LINK_NAME:
            return self.tool_path
        return super(PrintTrajectory, self).get_link_path(link_name)
    def reverse(self):
        return self.__class__(self.robot, self.joints, self.path[::-1],
                              self.tool_path[::-1], self.element, not self.is_reverse)
    def __repr__(self):
        return 'n{}->n{}'.format(self.n1, self.n2)

class Command(object):
    def __init__(self, trajectories=[], colliding=set()):
        self.trajectories = list(trajectories)
        self.colliding = set(colliding)
    @property
    def print_trajectory(self):
        for traj in self.trajectories:
            if isinstance(traj, PrintTrajectory):
                return traj
        return None
    @property
    def start_conf(self):
        return self.trajectories[0].path[0]
    @property
    def end_conf(self):
        return self.trajectories[-1].path[-1]
    def reverse(self):
        return self.__class__([traj.reverse() for traj in reversed(self.trajectories)],
                              colliding=self.colliding)
    def iterate(self):
        for trajectory in self.trajectories:
            for output in trajectory.iterate():
                yield output
    def __repr__(self):
        return 'c[{}]'.format(','.join(map(repr, self.trajectories)))

##################################################

def is_start_node(n1, e, node_points):
    return not element_supports(e, n1, node_points)

def doubly_printable(e, node_points):
    return all(is_start_node(n, e, node_points) for n in e)

def get_other_node(node1, element):
    assert node1 in element
    return element[node1 == element[0]]

def is_ground(element, ground_nodes):
    return any(n in ground_nodes for n in element)

def compute_element_distance(node_points, elements):
    return sum(get_distance(node_points[n1], node_points[n2]) for n1, n2 in elements)

##################################################

def get_supported_orders(elements, node_points):
    node_neighbors = get_node_neighbors(elements)
    orders = set()
    for node in node_neighbors:
        supporters = {e for e in node_neighbors[node] if element_supports(e, node, node_points)}
        printers = {e for e in node_neighbors[node] if is_start_node(node, e, node_points)
                    and not doubly_printable(e, node_points)}
        orders.update((e1, e2) for e1 in supporters for e2 in printers)
    return orders

def element_supports(e, n1, node_points): # A property of nodes
    # TODO: support polygon (ZMP heuristic)
    # TODO: recursively apply as well
    # TODO: end-effector force
    # TODO: allow just a subset to support
    # TODO: construct using only upwards
    n2 = get_other_node(n1, e)
    delta = node_points[n2] - node_points[n1]
    theta = angle_between(delta, [0, 0, -1])
    return theta < (np.pi / 2 - SUPPORT_THETA)

def retrace_supporters(element, incoming_edges, supporters):
    for element2 in incoming_edges[element]:
        if element2 not in supporters:
            retrace_supporters(element2, incoming_edges, supporters=supporters)
            supporters.append(element2)

##################################################

def downsample_nodes(elements, node_points, ground_nodes, num=None):
    if num is None:
        return elements, ground_nodes
    node_order = list(range(len(node_points)))
    # np.random.shuffle(node_order)
    node_order = sorted(node_order, key=lambda n: node_points[n][2])
    elements = sorted(elements, key=lambda e: min(node_points[n][2] for n in e))

    if num is not None:
        node_order = node_order[:num]
    ground_nodes = [n for n in ground_nodes if n in node_order]
    elements = [element for element in elements
                if all(n in node_order for n in element)]
    return elements, ground_nodes

def check_connected(ground_nodes, printed_elements):
    if not printed_elements:
        return True
    node_neighbors = get_node_neighbors(printed_elements)
    queue = deque(ground_nodes)
    visited_nodes = set(ground_nodes)
    visited_elements = set()
    while queue:
        node1 = queue.popleft()
        for element in node_neighbors[node1]:
            visited_elements.add(element)
            node2 = get_other_node(node1, element)
            if node2 not in visited_nodes:
                queue.append(node2)
                visited_nodes.add(node2)
    return printed_elements <= visited_elements

def get_connected_structures(elements):
    from pddlstream.utils import get_connected_components
    edges = {(e1, e2) for e1, neighbors in get_element_neighbors(elements).items()
             for e2 in neighbors}
    return get_connected_components(elements, edges)

##################################################

TRANS_TOL = 0.0015
ROT_TOL = 5 * np.pi / 180
# TRANS_TOL = 0.005
# ROT_TOL = 10 * np.pi / 180

def create_stiffness_checker(extrusion_path, verbose=False):
    # TODO: the stiffness checker likely has a memory leak
    # https://github.com/yijiangh/conmech/blob/master/src/bindings/pyconmech/pyconmech.cpp
    if not os.path.exists(extrusion_path):
        raise FileNotFoundError(extrusion_path)
    with HideOutput():
        checker = StiffnessChecker(json_file_path=extrusion_path, verbose=verbose)
    #checker.set_output_json(True)
    #checker.set_output_json_path(file_path=os.getcwd(), file_name="stiffness-results.json")

    checker.set_self_weight_load(include_self_weight=True)
    # checker.set_loads(include_self_weight=True, gravity_direction=[0, 0, -100])

    #checker.set_nodal_displacement_tol(transl_tol=0.005, rot_tol=10 * np.pi / 180)
    #checker.set_nodal_displacement_tol(transl_tol=0.003, rot_tol=5 * np.pi / 180)
    # checker.set_nodal_displacement_tol(transl_tol=1e-3, rot_tol=3 * (np.pi / 360))
    checker.set_nodal_displacement_tol(trans_tol=TRANS_TOL, rot_tol=ROT_TOL)
    #checker.set_loads(point_loads=None, include_self_weight=False, uniform_distributed_load={})

    return checker

def get_id_from_element(element_from_id):
    return {e: i for i, e in element_from_id.items()}

def get_extructed_ids(element_from_id, directed_elements):
    id_from_element = get_id_from_element(element_from_id)
    extruded_ids = []
    for directed in directed_elements:
        is_reverse = directed not in id_from_element
        assert (directed in id_from_element) != is_reverse
        element = directed[::-1] if is_reverse else directed
        extruded_ids.append(id_from_element[element])
    return sorted(extruded_ids)

Deformation = namedtuple('Deformation', ['success', 'displacements', 'fixities', 'reactions']) # TODO: get_max_nodal_deformation
Displacement = namedtuple('Displacement', ['dx', 'dy', 'dz', 'theta_x', 'theta_y', 'theta_z'])
Reaction = namedtuple('Reaction', ['fx', 'fy', 'fz', 'mx', 'my', 'mz'])

def force_from_reaction(reaction):
    return reaction[:3]

def torque_from_reaction(reaction):
    return reaction[3:]

##################################################

def evaluate_stiffness(extrusion_path, element_from_id, elements, checker=None, verbose=True):
    # TODO: check each component individually
    if not elements:
        return Deformation(True, {}, {}, {})
    #return True
    if checker is None:
        checker = create_stiffness_checker(extrusion_path, verbose=False)
    # TODO: load element_from_id
    extruded_ids = get_extructed_ids(element_from_id, elements)
    #print(checker.get_element_local2global_rot_matrices())
    #print(checker.get_element_stiffness_matrices(in_local_coordinate=False))

    #nodal_loads = checker.get_nodal_loads(existing_ids=[], dof_flattened=False) # per node
    #weight_loads = checker.get_self_weight_loads(existing_ids=[], dof_flattened=False) # get_nodal_loads = get_self_weight_loads?
    #for node in sorted(nodal_load):
    #    print(node, nodal_loads[node] - weight_loads[node])

    is_stiff = checker.solve(exist_element_ids=extruded_ids, if_cond_num=True)
    #print("has stored results: {0}".format(checker.has_stored_result()))
    success, nodal_displacement, fixities_reaction, element_reaction = checker.get_solved_results()
    assert is_stiff == success # TODO: this sometimes isn't true
    displacements = {i: Displacement(*d) for i, d in nodal_displacement.items()}
    fixities = {i: Reaction(*d) for i, d in fixities_reaction.items()}
    reactions = {i: (Reaction(*d[0]), Reaction(*d[1])) for i, d in element_reaction.items()}

    #translation = np.max(np.linalg.norm([d[:3] for d in displacements.values()], axis=1))
    #rotation = np.max(np.linalg.norm([d[3:] for d in displacements.values()], axis=1))

    #print("nodal displacement (m/rad):\n{0}".format(nodal_displacement)) # nodes x 7
    # TODO: investigate if nodal displacement can be used to select an ordering
    #print("fixities reaction (kN, kN-m):\n{0}".format(fixities_reaction)) # ground x 7
    #print("element reaction (kN, kN-m):\n{0}".format(element_reaction)) # elements x 13
    trans_tol, rot_tol = checker.get_nodal_deformation_tol()
    max_trans, max_rot, max_trans_vid, max_rot_vid = checker.get_max_nodal_deformation()
    # The inverse of stiffness is flexibility or compliance
    if verbose:
        print('Stiff: {} | Compliance: {:.5f}'.format(is_stiff, checker.get_compliance()))
        print('Max translation deformation: {0:.5f} / {1:.5} = {2:.5}, at node #{3}'.format(
            max_trans, trans_tol, max_trans / trans_tol, max_trans_vid))
        print('Max rotation deformation: {0:.5f} / {1:.5} = {2:.5}, at node #{3}'.format(
            max_rot, rot_tol, max_rot / rot_tol, max_rot_vid))
    #disc = 10
    #exagg_ratio = 1.0
    #time_step = 1.0
    #orig_beam_shape = checker.get_original_shape(disc=disc, draw_full_shape=False)
    #beam_disp = checker.get_deformed_shape(exagg_ratio=exagg_ratio, disc=disc)
    return Deformation(is_stiff, displacements, fixities, reactions)

def test_stiffness(extrusion_path, element_from_id, elements, **kwargs):
    return evaluate_stiffness(extrusion_path, element_from_id, elements, **kwargs).success

##################################################
# copy from PDDLStream
# https://github.com/caelan/pddlstream/blob/18b303e19bbab9f8e0016fbb2656f461067e1e94/pddlstream/utils.py

def incoming_from_edges(edges):
    incoming_vertices = defaultdict(set)
    for v1, v2 in edges:
        incoming_vertices[v2].add(v1)
    return incoming_vertices

def outgoing_from_edges(edges):
    outgoing_vertices = defaultdict(set)
    for v1, v2 in edges:
        outgoing_vertices[v1].add(v2)
    return outgoing_vertices

def neighbors_from_orders(orders):
    return incoming_from_edges(orders), \
           outgoing_from_edges(orders)