#!/usr/bin/env python
import os
import numpy as np
import subprocess
# from numba import jit

from msgs import *
import rospy
import rospkg
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from sensor_msgs.msg import Imu, JointState, LaserScan, NavSatFix
from geometry_msgs.msg import PoseStamped, Pose, Twist, Point, PolygonStamped
from move_base_msgs.msg import MoveBaseAction, MoveBaseActionGoal
from std_msgs.msg import Empty

np.random.seed()

base_path = rospkg.RosPack().get_path('jackal_helper')
reset_flag = False

def movebaseBypassInit():
    rospy.init_node("deepnav", anonymous=True)

    rospy.Subscriber("/move_base/goal", MoveBaseActionGoal, receiver('goal'))
    rospy.Subscriber("/move_base/local_costmap/footprint", PolygonStamped, receiver('footprint'))
    rospy.Subscriber("/move_base/local_costmap/costmap", OccupancyGrid, receiver('costmap'))
    rospy.Subscriber("/imu/data", Imu, receiver('imu'))
    rospy.Subscriber("/cmd_vel_ref", Twist, receiver('cmd_vel_ref'))

    rospy.Subscriber("/deepnav/reward", Point, receiver('reward'))
    # 服务
    rospy.Subscriber("/deepnav/reset", Empty, reset_cb)

    return rospy.Publisher('/cmd_vel', Twist, queue_size=1)


def movebaseRefInit():
    launch_file = os.path.join(base_path, '..', 'deepnav/launch/move_base_eband.launch')

    move_base_process = subprocess.Popen([
        'roslaunch',
        launch_file,
    ])


def getState():
    # dim (200, 200), (20,)
    costmap = global_msgs.get('costmap')
    imu = global_msgs.get('imu')
    gps = global_msgs.get('footprint')
    goal = global_msgs.get('goal')
    cmd_vel_ref = global_msgs.get('cmd_vel_ref')

    for var in (costmap, imu, gps, cmd_vel_ref):
        if var is None:
            return None, None, None


    # map (200, 200)
    width = costmap.info.width
    height = costmap.info.height
    resolution = costmap.info.resolution
    costmap = np.array(costmap.data, dtype=np.float32)
    costmap = costmap.reshape((height, width))
    costmap = costmap / 50 - 1

    # imu (10,)
    imu = (imu.orientation.x, imu.orientation.y, imu.orientation.z, imu.orientation.w,
           imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z,
           imu.linear_acceleration.x,imu.linear_acceleration.y, imu.linear_acceleration.z)
    imu = np.array(imu)
    imu[4:] /= 10
    # gps (8,)
    gps = (gps.polygon.points[0].x, gps.polygon.points[0].y, gps.polygon.points[1].x, gps.polygon.points[1].y,
           gps.polygon.points[2].x, gps.polygon.points[2].y, gps.polygon.points[3].x, gps.polygon.points[3].y)
    gps = np.array(gps)
    gps /= 5
    # ref (2,)
    ref = (cmd_vel_ref.linear.x / 0.5, cmd_vel_ref.angular.z / 1.57)
    ref = np.array(ref)

    return costmap, np.concatenate((imu, gps, ref), axis=0), cmd_vel_ref


def getReward():
    reward = -0.02
    r_dist = global_msgs.get('reward')
    if r_dist is None:
        r_dist = 0
    else:
        r_dist = r_dist.x
        global_msgs.store('reward', None)
    reward += r_dist
    return reward

def getDone():
    return 0

def reset_cb(_):
    global reset_flag
    reset_flag = True

def run():
    movebaseRefInit()

    from args import args
    from rl_base.agent import CnnPPO as Agent
    agent = Agent()
    agent.init(state_dim=args.state1d_dim,
               action_dim=args.action_dim,
               net_dim=args.net_dim,
               activation=args.activation,
               optimizer=args.optimizer,
               lr_cri=args.learning_rate_critic,
               lr_act=args.learning_rate_policy,
               device=args.device,
               )
    agent.init_static_param(**args.agent_kwargs)
    agent.save_or_load_agent(cwd='deepnav/scripts/agent', if_save=False)

    movebaseBypassInit()

    timer = rospy.Rate(21)  # reserve time for processing
    actionPub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

    while not rospy.is_shutdown():
        state, state1d, cmd_vel_ref = getState()
        if state is not None:
            a_avg, a_std, a_noise, logprob = agent.select_action(state, state1d)
            action = a_avg
            action = action.numpy()
            print(action)

            # action = (1, 0, 0)

            # output
            if action[2] > 0:
                cmd = cmd_vel_ref
            else:
                cmd = Twist()
                cmd.linear.x = np.clip(action[0], -1, 1) * 0.5
                cmd.angular.z = np.clip(action[1], -1, 1) * 0.5 * np.pi
            actionPub.publish(cmd)

        timer.sleep()


if __name__ == '__main__':
    run()

