import numpy as np

from rs_rinkou.humanoid import TwoLinkHumanoid
from rs_rinkou.humanoid_support import HumanoidViewer


def reference_angle(t):
    if t < 2.0:
        return 0.0
    if t < 5.0:
        return np.deg2rad(8.0)
    if t < 8.0:
        return np.deg2rad(-8.0)
    return 0.0


humanoid = TwoLinkHumanoid()
viewer = HumanoidViewer()
simulation_time = 12.0

for i in range(int(simulation_time / humanoid.dt)):
    t = i * humanoid.dt
    reference = reference_angle(t)
    x, u = humanoid.step(reference)
    viewer.add(t, reference, x, u)

viewer.show()
