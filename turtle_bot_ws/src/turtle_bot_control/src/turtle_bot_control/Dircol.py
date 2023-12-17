from casadi import *
import numpy as np
import matplotlib.pyplot as plt


class dircol:
    def __init__(self):

        self.opti = Opti()
        self.opti.solver('ipopt')
        self.h = 0.02
        self.T = 6
        self.N = int(self.T/self.h)
        self.max_vel = 0.22
        self.max_angvel = 2.84
        self.previous_x = np.zeros((3, self.N))
        self.previous_u = np.zeros((2, self.N))

    def dynamics(self, x, u):
        return vertcat(u[0]*cos(x[2]), u[0]*sin(x[2]), u[1])

    def get_cost(self):
        self.opti.minimize(
            10*sumsqr(self.x[0, :] - self.final_pose[0, :]) +
            10*sumsqr(self.x[1, :] - self.final_pose[1, :]) +
            sumsqr(self.x[2, :] - self.final_pose[2, :]) +
            sumsqr(self.u))

    def get_dynamics_contraints(self):

        for i in range(self.N-1):
            dyna_0 = self.dynamics(self.x[:, i], self.u[:, i])
            dyna_1 = self.dynamics(self.x[:, i+1], self.u[:, i+1])

            x_half = 0.5*(self.x[:, i] + self.x[:, i+1]) + \
                self.h*(dyna_0 - dyna_1)/8

            u_half = 0.5*(self.u[:, i] + self.u[:, i+1])
            dyna_half = self.dynamics(x_half, u_half)

            self.opti.subject_to(dyna_half +
                                 (3/(2*self.h))*(self.x[:, i] - self.x[:, i+1]) + (dyna_0 + dyna_1)/4 == 0)

    def get_control_constraints(self):

        self.opti.subject_to(
            self.opti.bounded(-self.max_vel, self.u[0, :], self.max_vel))
        self.opti.subject_to(
            self.opti.bounded(-self.max_angvel, self.u[1, :], self.max_angvel))

    def run_optimization(self, initial_pose: np.ndarray, final_pose: np.ndarray):

        self.x = self.opti.variable(3, self.N)
        self.u = self.opti.variable(2, self.N)
        self.initial_pose = self.opti.parameter(3, 1)
        self.final_pose = self.opti.parameter(3, 1)

        self.get_cost()
        self.get_dynamics_contraints()
        self.get_control_constraints()

        # add initial and final contraints
        self.opti.subject_to(self.x[:, 0] == initial_pose)
        self.opti.subject_to(self.x[:, self.N-1] == final_pose)
        # setting values for parameters
        self.opti.set_value(self.initial_pose, initial_pose)
        self.opti.set_value(self.final_pose, final_pose)

        # initialization of variables
        self.opti.set_initial(self.x, self.previous_x)
        self.opti.set_initial(self.u, self.previous_u)

        # running optimization
        self.sol = self.opti.solve()
        self.previous_x = np.array(self.sol.value(self.x))
        self.previous_u = np.array(self.sol.value(self.u))

    def get_plot(self):
        plt.figure()
        plt.plot(self.sol.value(self.u).T)
        plt.figure()
        plt.plot(self.sol.value(self.x[0, :]), self.sol.value(self.x[1, :]))
        plt.show()


if __name__ == '__main__':
    test = dircol()
    test.run_optimization(np.array([0, 0, 0]), np.array([1, 0, 0]))
