from casadi import *
import numpy as np
import matplotlib.pyplot as plt
import time as t


class dircol:
    def __init__(self):
        # calling Casadi opti stack
        self.opti = Opti()
        self.opti.solver('ipopt')
        # delta t
        self.h = 0.1
        # total time
        self.T = 3
        # horizon length
        self.N = int(self.T/self.h)
        # limits for control input
        self.max_vel = 0.22
        self.max_angvel = 2.84
        # declaring opti variables
        self.x = self.opti.variable(3, self.N)
        self.u = self.opti.variable(2, self.N)
        self.initial_pose = self.opti.parameter(3, 1)
        self.final_pose = self.opti.parameter(3, 1)
        self.mpc_param = self.opti.parameter(3, 1)

        self.get_cost()
        self.get_dynamics_contraints()
        self.get_control_constraints()

    def setup_MPC(self):
        # set MPC function
        self.MPC = self.opti.to_function('M', [self.initial_pose], [horzcat(self.u[:, 0], self.x[0:2, :])], [
            'initial_pose'], ['optimal_u'])

        print(self.MPC)

    def set_init_final_contraints(self, initial_pose, final_pose):
        # adding initial constraints
        self.opti.subject_to(self.x[:, 0] == self.initial_pose)
        # setting values for parameters
        self.opti.set_value(self.initial_pose, initial_pose)
        self.opti.set_value(self.final_pose, final_pose)

    def dynamics(self, x, u):
        # dynamics of model
        return vertcat(u[0]*cos(x[2]), u[0]*sin(x[2]), u[1])

    def get_cost(self):
        # minize cost function
        self.opti.minimize(
            30*sumsqr(self.x[0, :] - self.final_pose[0, :]) +
            30*sumsqr(self.x[1, :] - self.final_pose[1, :]) +
            sumsqr(self.x[2, :] - self.final_pose[2, :]) +
            sumsqr(self.u))

    def get_dynamics_contraints(self):
        # setting up Direct collocation constraints (3rd order spline)
        dyna_0 = self.dynamics(self.x[:, 0], self.u[:, 0])
        for i in range(self.N-1):
            dyna_1 = self.dynamics(self.x[:, i+1], self.u[:, i+1])

            x_half = 0.5*(self.x[:, i] + self.x[:, i+1]) + \
                self.h*(dyna_0 - dyna_1)/8
            u_half = 0.5*(self.u[:, i] + self.u[:, i+1])
            dyna_half = self.dynamics(x_half, u_half)

            self.opti.subject_to(dyna_half +
                                 (3/(2*self.h))*(self.x[:, i] - self.x[:, i+1]) + (dyna_0 + dyna_1)/4 == 0)
            dyna_0 = dyna_1

    def get_control_constraints(self):

        # setting control constraint of linear and angular velocity
        self.opti.subject_to(
            self.opti.bounded(-self.max_vel, self.u[0, :], self.max_vel))
        self.opti.subject_to(
            self.opti.bounded(-self.max_angvel, self.u[1, :], self.max_angvel))

    def run_optimization(self):

        # running optimization
        self.sol = self.opti.solve()
        self.get_plot()

    def get_plot(self):
        plt.figure()
        plt.plot(self.sol.value(self.u).T)
        plt.figure()
        plt.plot(self.sol.value(self.x[0, :]), self.sol.value(self.x[1, :]))
        plt.show()


if __name__ == '__main__':
    test = dircol()
    test.set_init_final_contraints(
        np.array([0, 0, 0]), np.array([1, 0, 0]))

    test.setup_MPC()
    x_log = []
    x = [0, 0, 0]
    t_start = t.time()
    for i in range(test.N):
        u = test.MPC(x)
        # print(test.MPC.value(test.x))
        test.opti.set_initial(test.x[0:2, 1:], u[:, 2:])
        # print(u)
    t_end = t.time()
    print(f"time takend {t_end - t_start}")

    # test.run_optimization()
    # test.run_optimization(np.array([0, 0, 0]), np.array([1, 0, 0]))
