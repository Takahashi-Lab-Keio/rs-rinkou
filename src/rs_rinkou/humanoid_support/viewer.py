import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


class HumanoidViewer:
    def __init__(self):
        self.times = []
        self.references = []
        self.q1_values = []
        self.q2_values = []
        self.ankle_torques = []
        self.hip_torques = []

    def add(self, t, reference_angle, x, u):
        self.times.append(t)
        self.references.append(np.rad2deg(reference_angle))
        self.q1_values.append(np.rad2deg(x[0]))
        self.q2_values.append(np.rad2deg(x[1]))
        self.ankle_torques.append(u[0])
        self.hip_torques.append(u[1])

    def show(self):
        figure = plt.figure(figsize=(11, 6))
        grid = figure.add_gridspec(2, 2, width_ratios=[1, 2])
        robot_axis = figure.add_subplot(grid[:, 0])
        angle_axis = figure.add_subplot(grid[0, 1])
        torque_axis = figure.add_subplot(grid[1, 1])

        robot_axis.plot([-0.4, 0.4], [0.0, 0.0], color="black")
        robot_axis.set_xlim(-1.2, 1.2)
        robot_axis.set_ylim(-0.1, 1.6)
        robot_axis.set_aspect("equal")
        robot_axis.set_title("Two-link humanoid")
        robot_axis.set_xlabel("Horizontal position (m)")
        robot_axis.set_ylabel("Height (m)")
        robot_axis.grid()
        robot_line = robot_axis.plot(
            [],
            [],
            "o-",
            linewidth=4,
            markersize=8,
            label="humanoid",
        )[0]
        reference_line = robot_axis.plot(
            [],
            [],
            "--",
            color="red",
            linewidth=2,
            label="reference",
        )[0]
        time_text = robot_axis.text(-1.1, 1.5, "")
        robot_axis.legend()

        angle_axis.set_xlim(self.times[0], self.times[-1])
        angle_min = min(self.references + self.q1_values + self.q2_values) - 2
        angle_max = max(self.references + self.q1_values + self.q2_values) + 2
        angle_axis.set_ylim(angle_min, angle_max)
        angle_axis.set_ylabel("Angle (deg)")
        angle_axis.grid()
        reference_plot = angle_axis.plot(
            [], [], "--", label="human reference"
        )[0]
        q1_plot = angle_axis.plot([], [], label="leg angle")[0]
        q2_plot = angle_axis.plot([], [], label="torso angle")[0]
        angle_axis.legend()

        torque_axis.set_xlim(self.times[0], self.times[-1])
        torque_min = min(self.ankle_torques + self.hip_torques) - 5
        torque_max = max(self.ankle_torques + self.hip_torques) + 5
        torque_axis.set_ylim(torque_min, torque_max)
        torque_axis.set_xlabel("Time (s)")
        torque_axis.set_ylabel("Torque (Nm)")
        torque_axis.grid()
        ankle_plot = torque_axis.plot([], [], label="ankle torque")[0]
        hip_plot = torque_axis.plot([], [], label="hip torque")[0]
        torque_axis.legend()

        def update(i):
            q1 = np.deg2rad(self.q1_values[i])
            q2 = np.deg2rad(self.q2_values[i])
            reference = np.deg2rad(self.references[i])
            hip_x = 0.8 * np.sin(q1)
            hip_y = 0.8 * np.cos(q1)
            shoulder_x = hip_x + 0.6 * np.sin(q2)
            shoulder_y = hip_y + 0.6 * np.cos(q2)
            reference_x = hip_x + 0.6 * np.sin(reference)
            reference_y = hip_y + 0.6 * np.cos(reference)

            robot_line.set_data(
                [0.0, hip_x, shoulder_x],
                [0.0, hip_y, shoulder_y],
            )
            reference_line.set_data(
                [hip_x, reference_x],
                [hip_y, reference_y],
            )
            time_text.set_text(f"t = {self.times[i]:.1f} s")

            reference_plot.set_data(
                self.times[: i + 1], self.references[: i + 1]
            )
            q1_plot.set_data(self.times[: i + 1], self.q1_values[: i + 1])
            q2_plot.set_data(self.times[: i + 1], self.q2_values[: i + 1])
            ankle_plot.set_data(
                self.times[: i + 1], self.ankle_torques[: i + 1]
            )
            hip_plot.set_data(
                self.times[: i + 1], self.hip_torques[: i + 1]
            )

            return (
                robot_line,
                reference_line,
                time_text,
                reference_plot,
                q1_plot,
                q2_plot,
                ankle_plot,
                hip_plot,
            )

        self.animation = FuncAnimation(
            figure,
            update,
            frames=len(self.times),
            interval=1000 / 30,
            blit=True,
            repeat=True,
        )
        plt.show()
