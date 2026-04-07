#!/usr/bin/env python3
from typing import Optional, Tuple, List, Dict
from argparse import ArgumentParser
from math import inf, sqrt, atan2, pi
from time import sleep, time
import queue
import json

import numpy as np
import rospy
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Twist, Point32, PoseStamped, Pose, Vector3, Quaternion, Point
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import LaserScan, PointCloud, ChannelFloat32
from visualization_msgs.msg import MarkerArray, Marker
from tf.transformations import euler_from_quaternion

# AABB format: (x_min, x_max, y_min, y_max)
OBS_TYPE = Tuple[float, float, float, float]
# Position format: {"x": x, "y": y, "theta": theta}
POSITION_TYPE = Dict[str, float]

# don't change this
GOAL_THRESHOLD = 0.1


def angle_to_0_to_2pi(angle: float) -> float:
    while angle < 0:
        angle += 2 * pi
    while angle > 2 * pi:
        angle -= 2 * pi
    return angle


class PIDController:
    """
    Generates control action taking into account instantaneous error (proportional action),
    accumulated error (integral action) and rate of change of error (derivative action).
    """

    def __init__(self, kP, kI, kD, kS, u_min, u_max):
        assert u_min < u_max, "u_min should be less than u_max"
        self.kP = kP
        self.kI = kI
        self.kD = kD
        self.kS = kS
        self.err_int = 0
        self.err_dif = 0
        self.err_prev = 0
        self.err_hist = []
        self.t_prev = 0
        self.u_min = u_min
        self.u_max = u_max

    def control(self, err, t):
        dt = t - self.t_prev
        self.err_hist.append(err)
        self.err_int += err
        if len(self.err_hist) > self.kS:
            self.err_int -= self.err_hist.pop(0)
        self.err_dif = err - self.err_prev
        u = (self.kP * err) + (self.kI * self.err_int * dt) + (self.kD * self.err_dif / dt)
        self.err_prev = err
        self.t_prev = t
        return max(self.u_min, min(u, self.u_max))


class Node:
    def __init__(self, position: POSITION_TYPE, parent: "Node"):
        self.position = position
        self.neighbors = []
        self.parent = parent

    def distance_to(self, other_node: "Node") -> float:
        return np.linalg.norm(self.position - other_node.position)

    def to_dict(self) -> Dict:
        return {"x": self.position[0], "y": self.position[1]}

    def __str__(self) -> str:
        return (
            f"Node<pos: {round(self.position[0], 4)}, {round(self.position[1], 4)}, #neighbors: {len(self.neighbors)}>"
        )


class RrtPlanner:

    def __init__(self, obstacles: List[OBS_TYPE], map_aabb: Tuple):
        self.obstacles = obstacles
        self.map_aabb = map_aabb
        self.graph_publisher = rospy.Publisher("/rrt_graph", MarkerArray, queue_size=10)
        self.plan_visualization_pub = rospy.Publisher("/waypoints", MarkerArray, queue_size=10)
        self.delta = 0.1
        self.obstacle_padding = 0.15
        self.goal_threshold = GOAL_THRESHOLD

    def visualize_plan(self, path: List[Dict]):
        marker_array = MarkerArray()
        for i, waypoint in enumerate(path):
            marker = Marker()
            marker.header.frame_id = "odom"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "waypoints"
            marker.id = i
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            marker.pose.position = Point(waypoint["x"], waypoint["y"], 0.0)
            marker.pose.orientation = Quaternion(0, 0, 0, 1)
            marker.scale = Vector3(0.075, 0.075, 0.1)
            marker.color = ColorRGBA(0.0, 0.0, 1.0, 0.5)
            marker_array.markers.append(marker)
        self.plan_visualization_pub.publish(marker_array)

    def visualize_graph(self, graph: List[Node]):
        marker_array = MarkerArray()
        for i, node in enumerate(graph):
            marker = Marker()
            marker.header.frame_id = "odom"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "waypoints"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.scale = Vector3(0.05, 0.05, 0.05)
            marker.pose.position = Point(node.position[0], node.position[1], 0.01)
            marker.pose.orientation = Quaternion(0, 0, 0, 1)
            marker.color = ColorRGBA(0.0, 1.0, 0.0, 0.5)
            marker_array.markers.append(marker)
        self.graph_publisher.publish(marker_array)

    def _randomly_sample_q(self) -> Node:
        # Choose uniform randomly sampled points within map_aabb
        ######### Your code starts here #########
        x_min, x_max, y_min, y_max = self.map_aabb
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)
        return Node(np.array([x, y]), None)
        ######### Your code ends here #########

    def _nearest_vertex(self, graph: List[Node], q: Node) -> Node:
        # Determine vertex nearest to sampled point
        ######### Your code starts here #########
        nearest = None
        min_dist = float('inf')
        for node in graph:
            d = node.distance_to(q)
            if d < min_dist:
                min_dist = d
                nearest = node
        return nearest
        ######### Your code ends here #########

    def _is_in_collision(self, q_rand: Node):
        x = q_rand.position[0]
        y = q_rand.position[1]
        for obs in self.obstacles:
            x_min, x_max, y_min, y_max = obs
            x_min -= self.obstacle_padding
            y_min -= self.obstacle_padding
            x_max += self.obstacle_padding
            y_max += self.obstacle_padding
            if (x_min < x and x < x_max) and (y_min < y and y < y_max):
                return True
        return False

    def _extend(self, graph: List[Node], q_rand: Node):
        # Check if sampled point is in collision and add to tree if not
        ######### Your code starts here #########
        q_near = self._nearest_vertex(graph, q_rand)

        # Compute direction from q_near toward q_rand
        direction = q_rand.position - q_near.position
        dist = np.linalg.norm(direction)

        if dist == 0:
            return None

        # Step by delta (or less if q_rand is closer)
        step = min(self.delta, dist)
        new_pos = q_near.position + (direction / dist) * step

        q_new = Node(new_pos, q_near)

        if not self._is_in_collision(q_new):
            q_near.neighbors.append(q_new)
            graph.append(q_new)
            return q_new

        return None
        ######### Your code ends here #########

    def generate_plan(self, start: POSITION_TYPE, goal: POSITION_TYPE) -> Tuple[List[POSITION_TYPE], List[Node]]:
        """Public facing API for generating a plan. Returns the plan and the graph.

        Return format:
            plan:
            [
                {"x": start["x"], "y": start["y"]},
                {"x": ...,      "y": ...},
                            ...
                {"x": goal["x"],  "y": goal["y"]},
            ]
            graph:
                [
                    Node<pos: x1, y1, #neighbors: n_1>,
                    ...
                    Node<pos: x_n, y_n, #neighbors: z>,
                ]
        """
        graph = [Node(np.array([start["x"], start["y"]]), None)]
        goal_node = Node(np.array([goal["x"], goal["y"]]), None)
        plan = []

        # Find path from start to goal location through tree
        ######### Your code starts here #########
        max_iterations = 10000
        goal_reached_node = None

        for _ in range(max_iterations):
            q_rand = self._randomly_sample_q()
            q_new = self._extend(graph, q_rand)

            if q_new is not None:
                # Check if q_new is within goal threshold
                if q_new.distance_to(goal_node) <= self.goal_threshold:
                    # Add the actual goal node as the final node
                    goal_final = Node(np.array([goal["x"], goal["y"]]), q_new)
                    graph.append(goal_final)
                    goal_reached_node = goal_final
                    break

        # Backtrack from goal to start to extract path
        if goal_reached_node is not None:
            current = goal_reached_node
            while current is not None:
                plan.append({"x": current.position[0], "y": current.position[1]})
                current = current.parent
            plan.reverse()
        else:
            rospy.logwarn("RRT failed to find a path within max iterations")

        ######### Your code ends here #########
        return plan, graph


class ObstacleFreeWaypointController:
    """
    Controller that navigates a robot through a sequence of waypoints.
    Adapted from lab 5/6/7 style waypoint controller.
    """

    def __init__(self, plan: List[POSITION_TYPE]):
        self.plan = plan
        self.waypoint_index = 0
        self.goal_threshold = GOAL_THRESHOLD

        # PID controllers for angular and linear velocity
        self.angular_pid = PIDController(kP=2.0, kI=0.0, kD=0.1, kS=10, u_min=-1.5, u_max=1.5)
        self.linear_pid  = PIDController(kP=0.5, kI=0.0, kD=0.0, kS=10, u_min=0.0,  u_max=0.3)

        self.cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self.odom_sub    = rospy.Subscriber("/odom", Odometry, self._odom_callback)

        self.position = {"x": 0.0, "y": 0.0, "theta": 0.0}
        self.rate = rospy.Rate(10)

    def _odom_callback(self, msg: Odometry):
        self.position["x"] = msg.pose.pose.position.x
        self.position["y"] = msg.pose.pose.position.y
        quat = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])
        self.position["theta"] = yaw

    def _distance_to_waypoint(self, waypoint: POSITION_TYPE) -> float:
        dx = waypoint["x"] - self.position["x"]
        dy = waypoint["y"] - self.position["y"]
        return sqrt(dx * dx + dy * dy)

    def _angle_to_waypoint(self, waypoint: POSITION_TYPE) -> float:
        dx = waypoint["x"] - self.position["x"]
        dy = waypoint["y"] - self.position["y"]
        return atan2(dy, dx)

    def control_robot(self):
        if self.waypoint_index >= len(self.plan):
            # Stop the robot
            self.cmd_vel_pub.publish(Twist())
            return

        waypoint = self.plan[self.waypoint_index]
        dist = self._distance_to_waypoint(waypoint)

        if dist < self.goal_threshold:
            self.waypoint_index += 1
            rospy.loginfo(f"Reached waypoint {self.waypoint_index}/{len(self.plan)}")
            if self.waypoint_index >= len(self.plan):
                self.cmd_vel_pub.publish(Twist())
                rospy.loginfo("Goal reached!")
            return

        # Compute heading error
        target_angle = self._angle_to_waypoint(waypoint)
        heading_error = target_angle - self.position["theta"]

        # Normalize to [-pi, pi]
        while heading_error > pi:
            heading_error -= 2 * pi
        while heading_error < -pi:
            heading_error += 2 * pi

        t = time()
        angular_vel = self.angular_pid.control(heading_error, t)

        # Only move forward when roughly facing the waypoint
        if abs(heading_error) < 0.5:
            linear_vel = self.linear_pid.control(dist, t)
        else:
            linear_vel = 0.0

        cmd = Twist()
        cmd.linear.x  = linear_vel
        cmd.angular.z = angular_vel
        self.cmd_vel_pub.publish(cmd)
        self.rate.sleep()


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--map_filepath", type=str, required=True)
    args = parser.parse_args()
    with open(args.map_filepath, "r") as f:
        map_ = json.load(f)
        goal_position = map_["goal_position"]
        obstacles = map_["obstacles"]
        map_aabb = map_["map_aabb"]
        start_position = {"x": 0.0, "y": 0.0}

    rospy.init_node("rrt_planner")
    planner = RrtPlanner(obstacles, map_aabb)
    plan, graph = planner.generate_plan(start_position, goal_position)
    planner.visualize_plan(plan)
    planner.visualize_graph(graph)
    controller = ObstacleFreeWaypointController(plan)

    try:
        while not rospy.is_shutdown():
            controller.control_robot()
    except rospy.ROSInterruptException:
        print("Shutting down...")
