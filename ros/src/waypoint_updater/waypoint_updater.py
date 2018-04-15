#!/usr/bin/env python


import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped
from std_msgs.msg import Int32
from styx_msgs.msg import Lane, Waypoint
from scipy.spatial import KDTree

import math
import numpy as np

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.
As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.
Once you have created dbw_node, you will update this node to use the status of traffic lights too.
Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.
TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOP_RATE = 30  # Hz
LOOKAHEAD_WPS = 100  # 200

MIN_DECELERATION_DISTANCE = 30

# TODO use accel_limit? param in the simulator : 1.0
# use lesser acceleration to ensure comfortable ride
MAX_ACCELERATION = 0.75

# TODO use decel_limit? param in the simulator : -5.0
# use 1/10 of limit instead to ensure comfortable ride
MAX_DECELERATION = 0.5

OBSTACLE_DISTANCE_OFFSET = 1


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.logwarn("[WaypointUpdater] Start initialization")

        rospy.Subscriber('/current_pose', PoseStamped, self.set_current_pose)
        rospy.Subscriber('/current_velocity', TwistStamped, self.set_velocity)
        rospy.Subscriber('/base_waypoints', Lane, self.set_base_waypoints)
        rospy.Subscriber('/traffic_waypoint', Int32, self.set_closest_traffic_waypoint)
        rospy.Subscriber('/obstacle_waypoint', Lane, self.set_closest_obstacle_waypoint)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        target_velocity_kmh_param = rospy.get_param('/waypoint_loader/velocity')

        # slow down a bit from max speed
        target_velocity_kmh_param = target_velocity_kmh_param * 0.9

        # init constants
        self.target_velocity_mps_param = (target_velocity_kmh_param * 1000) / 3600

        rospy.logwarn("[WaypointUpdater] Target velocity param: %f. m/s" % self.target_velocity_mps_param)

        # init setter fields
        self.current_pose = None
        self.current_velocity_mps = 0
        self.waypoints = None
        self.stop_waypoint_index = None

        rospy.logwarn("[WaypointUpdater] Initialization successful!")

        self.loop()

    # control functions

    def loop(self):

        rate = rospy.Rate(LOOP_RATE)

        while not rospy.is_shutdown():
            if self.current_pose and self.waypoints:
                self.publish_waypoints()

            rate.sleep()

    def publish_waypoints(self):
        final_lane = self.generate_lane()
        self.final_waypoints_pub.publish(final_lane)

    def generate_lane(self):

        look_ahead_waypoints = []

        closest_waypoint_idx = self.get_closest_waypoint_idx()

        # get closest obstacle index or the last waypoint (end of the track)
        closest_stop_idx = self.get_closest_stop_idx()

        distance_to_obstacle = self.distance(self.waypoints, closest_waypoint_idx, closest_stop_idx)

        if closest_stop_idx > closest_waypoint_idx:
            look_ahead_index = min(closest_waypoint_idx + LOOKAHEAD_WPS, closest_stop_idx)
            look_ahead_waypoints = self.waypoints[closest_waypoint_idx: look_ahead_index]

            if distance_to_obstacle <= MIN_DECELERATION_DISTANCE:
                self.decelerate(look_ahead_waypoints, len(look_ahead_waypoints) - 1)
            else:
                self.accelerate(look_ahead_waypoints, self.target_velocity_mps_param)

        lane = Lane()
        lane.waypoints = look_ahead_waypoints

        return lane

    def accelerate(self, waypoints, target_velocity):

        for i in range(len(waypoints)):
            waypoint_velocity = min(self.current_velocity_mps + (i + 1) * MAX_ACCELERATION, target_velocity)
            self.set_waypoint_velocity(waypoints[i], waypoint_velocity)

    def decelerate(self, waypoints, next_stop_index):

        self.set_waypoint_velocity(waypoints[next_stop_index], 0)

        for index, waypoint in enumerate(waypoints):
            if index < next_stop_index:
                distance = self.distance(waypoints, index, next_stop_index)
                distance = max(distance - OBSTACLE_DISTANCE_OFFSET, 0)

                velocity = math.sqrt(2 * MAX_DECELERATION * distance)
            else:
                velocity = 0

            if velocity < 1.:
                velocity = 0.0

            self.set_waypoint_velocity(waypoint, min(velocity, waypoint.twist.twist.linear.x))

    # listeners

    def set_current_pose(self, msg):
        self.current_pose = msg.pose

    def set_velocity(self, msg):
        self.current_velocity_mps = math.sqrt(msg.twist.linear.x ** 2 + msg.twist.linear.y ** 2)

        current_velocity_mph = self.current_velocity_mps / 0.44704

        rospy.logwarn("[WaypointUpdater] self.current_velocity:%f. mph; %f mps" % (
        current_velocity_mph, self.current_velocity_mps))

    def set_base_waypoints(self, lane):
        self.waypoints = lane.waypoints

        self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in
                             lane.waypoints]
        self.waypoint_tree = KDTree(self.waypoints_2d)

    def set_closest_traffic_waypoint(self, msg):
        self.stop_waypoint_index = msg.data

    def set_closest_obstacle_waypoint(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    # util functions

    def get_closest_waypoint_idx(self):
        x = self.current_pose.position.x
        y = self.current_pose.position.y
        closest_idx = self.waypoint_tree.query([x, y], 1)[1]

        # Is closest point ahead or behind car
        closest_coord = self.waypoints_2d[closest_idx]
        prev_coord = self.waypoints_2d[closest_idx - 1]
        # Hyperplane through closest coords
        cl_vect = np.array(closest_coord)
        prev_vect = np.array(prev_coord)
        pos_vect = np.array([x, y])

        val = np.dot(cl_vect - prev_vect, pos_vect - cl_vect)

        if val > 0:
            closest_idx = (closest_idx + 1) % len(self.waypoints_2d)
        return closest_idx

    def get_closest_stop_idx(self):
        map_waypoints_num = len(self.waypoints)

        if self.stop_waypoint_index is not None \
                and 0 < self.stop_waypoint_index < map_waypoints_num:
            return self.stop_waypoint_index

        return map_waypoints_num - 1

    def distance(self, waypoints, first_waypoint_index, second_waypoint_index):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)
        for i in range(first_waypoint_index, second_waypoint_index + 1):
            dist += dl(waypoints[first_waypoint_index].pose.pose.position, waypoints[i].pose.pose.position)
            first_waypoint_index = i
        return dist

    def set_waypoint_velocity(self, waypoint, velocity):
        waypoint.twist.twist.linear.x = velocity


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
