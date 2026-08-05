
import threading

from rs_rinkou.pose_viewer import RealSensePoseViewer


rs = RealSensePoseViewer()
thread = threading.Thread(target=rs.loop)
thread.start()

last_sample_id = 0
dt = 1 / 30.0
while thread.is_alive():
    sample_id, _, xyz = rs.get_sample("right_shoulder")
    if sample_id == last_sample_id:
        rs.draw()
        continue

    last_sample_id = sample_id
    print(xyz)
    estimated_xyz = xyz

    # kalman filter begin


    # kalman filter end

    if estimated_xyz is not None:
        rs.set_estimate("right_shoulder", estimated_xyz, sample_id)
    rs.draw()
