import threading

import numpy as np

from rs_rinkou.kf import kf
from rs_rinkou.pose_viewer import RealSensePoseViewer


rs = RealSensePoseViewer()
thread = threading.Thread(target=rs.loop)
thread.start()

C = np.array(
    [
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
    ]
)
Bu = np.array([[0], [0], [0], [0], [0], [0]])
u = np.array([0])
Q = np.array(
    [
        [1.0, 0, 0],
        [0, 1.0, 0],
        [0, 0, 1.0],
    ]
)
R = np.array(
    [
        [0.03**2, 0, 0],
        [0, 0.03**2, 0],
        [0, 0, 0.10**2],
    ]
)

xhat = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
P = np.array(
    [
        [0.03**2, 0, 0, 0, 0, 0],
        [0, 0.03**2, 0, 0, 0, 0],
        [0, 0, 0.10**2, 0, 0, 0],
        [0, 0, 0, 1.0, 0, 0],
        [0, 0, 0, 0, 1.0, 0],
        [0, 0, 0, 0, 0, 1.0],
    ]
)
initialized = False
last_sample_id = 0
dt = 1 / 30.0

while thread.is_alive():
    sample_id, _, xyz = rs.get_sample("right_shoulder")
    if sample_id == last_sample_id:
        rs.draw()
        continue

    last_sample_id = sample_id

    if not initialized:
        if np.all(np.isfinite(xyz)):
            xhat[:3] = xyz
            initialized = True
            rs.set_estimate("right_shoulder", xhat[:3], sample_id)
        rs.draw()
        continue

    A = np.array(
        [
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ]
    )
    B = np.array(
        [
            [dt**2 / 2, 0, 0],
            [0, dt**2 / 2, 0],
            [0, 0, dt**2 / 2],
            [dt, 0, 0],
            [0, dt, 0],
            [0, 0, dt],
        ]
    )

    if np.all(np.isfinite(xyz)):
        xhat, P, G = kf(A, B, Bu, C, Q, R, u, xyz, xhat, P)
    else:
        xhat = A @ xhat
        P = A @ P @ A.T + B @ Q @ B.T

    estimated_xyz = xhat[:3]
    rs.set_estimate("right_shoulder", estimated_xyz, sample_id)
    rs.draw()
