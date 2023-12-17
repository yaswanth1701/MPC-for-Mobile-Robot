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
        self.desired_pose = np.array([1, 0, 0])
        self.initial_pose = np.zeros(3)
        self.mpc_rate = 50
        # declaring message type
        self.cmd_vel_msg = Twist()
        self.horizon_viz_msg = Path()
        self.control_horizon_viz_msg = Path()
        self.horizon_viz_msg.header.frame_id = "map"
        self.control_horizon_viz_msg.header.frame_id = "map"
        # publisher to publish messages
        self.cmd_vel_pub = rospy.Publisher("cmd_vel", Twist, queue_size=10)
        self.horizon_viz_pub = rospy.Publisher(
            "mpc_horizon", Path, queue_size=10)
        self.control_viz_horizon_pub = rospy.Publisher(
            "control_horizon", Path, queue_size=10)
        # subcriber to get model state
        self.pose_sub = rospy.Subscriber("/odom", Odometry, self.odom_callback)

    def odom_callback(self, msg: Odometry):
        self.odom_msg = msg
        self.initial_pose[0] = msg.pose.pose.position.x
        self.initial_pose[1] = msg.pose.pose.position.y
        r, p, y = euler_from_quaternion(
            [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,
             msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])

        self.initial_pose[2] = y

        # print(f"[MPC]: {self.initial_pose}")

    def run_mpc(self):

        self.traj_opt.run_optimization(self.initial_pose, self.desired_pose)
        self.get_cmd_vel_msg()
        self.get_horizon_msg()
        self.get_control_viz_horizon_msg()
        self.send_command()

    def get_horizon_msg(self):
        self.horizon_viz_msg.header.stamp = rospy.Time.now()
        self.horizon_viz_msg.poses.clear()

        viz_pose = PoseStamped()

        for i in range(self.traj_opt.N):
            viz_pose.pose.position.x = self.traj_opt.sol.value(
                self.traj_opt.x[0, i])
            viz_pose.pose.position.y = self.traj_opt.sol.value(
                self.traj_opt.x[1, i])
            self.horizon_viz_msg.poses.append(viz_pose)

    def get_control_viz_horizon_msg(self):
        self.control_horizon_viz_msg.header.stamp = rospy.Time.now()
        self.control_horizon_viz_msg.poses.clear()

        viz_pose = PoseStamped()

        viz_pose.pose.position = self.odom_msg.pose.pose.position
        self.control_horizon_viz_msg.poses.append(viz_pose)

        viz_pose.pose.position.x = self.traj_opt.sol.value(
            self.traj_opt.x[0, 1])
        viz_pose.pose.position.y = self.traj_opt.sol.value(
            self.traj_opt.x[1, 1])
        self.control_horizon_viz_msg.poses.append(viz_pose)

    def get_cmd_vel_msg(self):
        self.cmd_vel_msg.linear.x = self.traj_opt.sol.value(
            self.traj_opt.u[0, 0])
        self.cmd_vel_msg.angular.z = self.traj_opt.sol.value(
            self.traj_opt.sol.value(self.traj_opt.u[1, 0]))

    def send_command(self):
        self.cmd_vel_pub.publish(self.cmd_vel_msg)
        self.horizon_viz_pub.publish(self.horizon_viz_msg)
        self.control_viz_horizon_pub.publish(self.control_horizon_viz_msg)


if __name__ == '__main__':
    test = MPC()
