from turtle_bot_control.Dircol import dircol
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Path, Odometry
import rospy
import numpy as np
from tf.transformations import euler_from_quaternion


class MPC:
    def __init__(self):

        self.traj_opt = dircol()
        rospy.loginfo("[MPC]: initializing DIRCOL")

        # goal pose
        self.desired_pose = np.array([2, 2, 0])
        self.initial_pose = np.zeros(3)
        # frequency of mpc
        self.mpc_rate = 20
        # setting up inital and final contraints
        self.traj_opt.set_init_final_contraints(
            self.initial_pose, self.desired_pose)
        # setting up mpc function in DIRCOL
        # self.traj_opt.setup_MPC()
        # declaring message type
        self.cmd_vel_msg = Twist()
        self.horizon_viz_msg = Path()
        self.turtle_path_viz_msg = Path()

        # mpc horizon and tracked path msg
        self.horizon_viz_msg.header.frame_id = "odom"
        self.turtle_path_viz_msg.header.frame_id = "odom"

        # publisher to publish messages
        self.cmd_vel_pub = rospy.Publisher("cmd_vel", Twist, queue_size=10)
        self.horizon_viz_pub = rospy.Publisher(
            "mpc_horizon", Path, queue_size=10)
        self.turtle_path_pub = rospy.Publisher(
            "control_horizon", Path, queue_size=10)

        # subcriber to get model state
        self.pose_sub = rospy.Subscriber("/odom", Odometry, self.odom_callback)

    def odom_callback(self, msg: Odometry):
        '''callback odom msg'''

        self.odom_msg = msg
        self.initial_pose[0] = msg.pose.pose.position.x
        self.initial_pose[1] = msg.pose.pose.position.y
        r, p, y = euler_from_quaternion(
            [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,
             msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])

        self.initial_pose[2] = y

        # print(f"[MPC]: {self.initial_pose}")

    def run_mpc(self):

        # setting up initial position of turtle_bot
        self.traj_opt.opti.set_value(
            self.traj_opt.initial_pose, self.initial_pose)
        # calculating error values
        if np.linalg.norm(self.desired_pose - self.initial_pose) >= 1e-2:
            self.traj_opt.run_optimization()
            self.u = self.traj_opt.sol.value(self.traj_opt.u)

            self.mpc = True
        else:
            self.mpc = False
            pass

        # getting message values
        self.get_cmd_vel_msg()
        self.get_horizon_msg()
        self.get_path_msg()
        self.send_command()

    def get_horizon_msg(self):
        '''function to set mpc horizon  into path msg'''

        self.horizon_viz_msg.header.stamp = rospy.Time.now()
        # clearing previous path msg
        self.horizon_viz_msg.poses.clear()

        # setting values
        for i in range(self.traj_opt.N):
            viz_pose = PoseStamped()
            viz_pose.pose.position.x = self.traj_opt.sol.value(
                self.traj_opt.x[0, i])
            viz_pose.pose.position.y = self.traj_opt.sol.value(
                self.traj_opt.x[1, i])
            self.horizon_viz_msg.poses.append(viz_pose)

    def get_path_msg(self):
        '''turtle bot tracked path msg'''

        self.turtle_path_viz_msg.header.stamp = rospy.Time.now()
        viz_pose = PoseStamped()

        viz_pose.pose.position.x = self.initial_pose[0]
        viz_pose.pose.position.y = self.initial_pose[1]
        self.turtle_path_viz_msg.poses.append(viz_pose)

    def get_cmd_vel_msg(self):
        '''cmd_vel msg function'''

        # if mpc is active
        if self.mpc:
            self.cmd_vel_msg.linear.x = self.u[0, 0]
            self.cmd_vel_msg.angular.z = self.u[1, 0]
        else:
            self.cmd_vel_msg.linear.x = 0
            self.cmd_vel_msg.angular.z = 0

    def send_command(self):
        '''function to send message'''

        # pubslishing cmd_vel and path msg
        self.cmd_vel_pub.publish(self.cmd_vel_msg)
        self.horizon_viz_pub.publish(self.horizon_viz_msg)
        self.turtle_path_pub.publish(self.turtle_path_viz_msg)


if __name__ == '__main__':
    test = MPC()
