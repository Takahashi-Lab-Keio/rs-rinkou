import numpy as np
from scipy.linalg import solve_continuous_are


class TwoLinkHumanoid:
    def __init__(self):
        self.dt = 1 / 30.0
        self.m1 = 20.0
        self.m2 = 40.0
        self.l1 = 0.8
        self.l2 = 0.6
        self.a1 = self.l1 / 2
        self.a2 = self.l2 / 2
        self.i1 = self.m1 * self.l1**2 / 12
        self.i2 = self.m2 * self.l2**2 / 12
        self.g = 9.81

        self.M0 = np.array(
            [
                [
                    self.m1 * self.a1**2
                    + self.i1
                    + self.m2 * self.l1**2,
                    self.m2 * self.l1 * self.a2,
                ],
                [
                    self.m2 * self.l1 * self.a2,
                    self.m2 * self.a2**2 + self.i2,
                ],
            ]
        )
        self.G = np.array(
            [
                [
                    self.g
                    * (self.m1 * self.a1 + self.m2 * self.l1),
                    0.0,
                ],
                [0.0, self.g * self.m2 * self.a2],
            ]
        )
        self.S = np.array(
            [
                [1.0, -1.0],
                [0.0, 1.0],
            ]
        )

        gravity_linear = np.linalg.solve(self.M0, self.G)
        torque_linear = np.linalg.solve(self.M0, self.S)
        A = np.array(
            [
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [
                    gravity_linear[0, 0],
                    gravity_linear[0, 1],
                    0.0,
                    0.0,
                ],
                [
                    gravity_linear[1, 0],
                    gravity_linear[1, 1],
                    0.0,
                    0.0,
                ],
            ]
        )
        B = np.array(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [torque_linear[0, 0], torque_linear[0, 1]],
                [torque_linear[1, 0], torque_linear[1, 1]],
            ]
        )
        Q = np.array(
            [
                [400.0, 0.0, 0.0, 0.0],
                [0.0, 800.0, 0.0, 0.0],
                [0.0, 0.0, 20.0, 0.0],
                [0.0, 0.0, 0.0, 20.0],
            ]
        )
        R = np.array(
            [
                [0.2, 0.0],
                [0.0, 0.2],
            ]
        )
        P = solve_continuous_are(A, B, Q, R)
        self.K = np.linalg.solve(R, B.T @ P)
        self.x = np.array(
            [np.deg2rad(3.0), np.deg2rad(-2.0), 0.0, 0.0]
        )

    def dynamics(self, x, u):
        q1, q2, dq1, dq2 = x
        d = self.m2 * self.l1 * self.a2
        M = np.array(
            [
                [self.M0[0, 0], d * np.cos(q1 - q2)],
                [d * np.cos(q1 - q2), self.M0[1, 1]],
            ]
        )
        h = np.array(
            [
                d * np.sin(q1 - q2) * dq2**2,
                -d * np.sin(q1 - q2) * dq1**2,
            ]
        )
        gravity = np.array(
            [
                -self.G[0, 0] * np.sin(q1),
                -self.G[1, 1] * np.sin(q2),
            ]
        )
        ddq = np.linalg.solve(M, self.S @ u - h - gravity)
        return np.array([dq1, dq2, ddq[0], ddq[1]])

    def step(self, reference_angle):
        x_reference = np.array([0.0, reference_angle, 0.0, 0.0])
        gravity_reference = np.array(
            [0.0, -self.G[1, 1] * np.sin(reference_angle)]
        )
        u_reference = np.linalg.solve(self.S, gravity_reference)
        u = u_reference - self.K @ (self.x - x_reference)
        u = np.clip(u, -100.0, 100.0)

        k1 = self.dynamics(self.x, u)
        k2 = self.dynamics(self.x + k1 * self.dt / 2, u)
        k3 = self.dynamics(self.x + k2 * self.dt / 2, u)
        k4 = self.dynamics(self.x + k3 * self.dt, u)
        self.x = self.x + (k1 + 2 * k2 + 2 * k3 + k4) * self.dt / 6

        return self.x.copy(), u.copy()
