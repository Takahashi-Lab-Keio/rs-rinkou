import threading
import time

import matplotlib.pyplot as plt
import numpy as np
import pyrealsense2 as rs
from matplotlib.collections import LineCollection
from mmpose.apis import inference_topdown, init_model


DISPLAY_KEYPOINT_NAMES = [
    "nose",
    "right_shoulder",
    "right_hip",
    "right_knee",
    "right_ankle",
]

CONFIG_FILE = "/home/ytnpc2022b/rs-rinkou/.venv/lib/python3.11/site-packages/mmpose/.mim/configs/body_2d_keypoint/rtmpose/coco/rtmpose-t_8xb256-420e_aic-coco-256x192.py"
CHECKPOINT_FILE = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-tiny_simcc-aic-coco_pt-aic-coco_420e-256x192-cfc8f33d_20230126.pth"
SKELETON = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]
KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


class RealSensePoseViewer:
    def __init__(self):
        self.running = False
        self.lock = threading.Lock()
        self.model = init_model(CONFIG_FILE, CHECKPOINT_FILE, device="cuda:0")

        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
        profile = self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)
        self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        self.color_intrinsics = profile.get_stream(
            rs.stream.color
        ).as_video_stream_profile().get_intrinsics()

        self.display_keypoint_indexes = [
            KEYPOINT_NAMES.index(name) for name in DISPLAY_KEYPOINT_NAMES
        ]
        self.xyz = {
            name: np.array([np.nan, np.nan, np.nan])
            for name in DISPLAY_KEYPOINT_NAMES
        }
        self.estimated_xyz = {
            name: np.array([np.nan, np.nan, np.nan])
            for name in DISPLAY_KEYPOINT_NAMES
        }
        self.sample_id = 0
        self.timestamp = 0.0
        self.drawn_sample_id = -1
        self.graph_drawn_sample_id = -1
        self.sample_ids = []
        self.x_history = [[] for _ in DISPLAY_KEYPOINT_NAMES]
        self.y_history = [[] for _ in DISPLAY_KEYPOINT_NAMES]
        self.z_history = [[] for _ in DISPLAY_KEYPOINT_NAMES]
        self.estimate_sample_ids = {
            name: [] for name in DISPLAY_KEYPOINT_NAMES
        }
        self.estimate_x_history = {
            name: [] for name in DISPLAY_KEYPOINT_NAMES
        }
        self.estimate_y_history = {
            name: [] for name in DISPLAY_KEYPOINT_NAMES
        }
        self.estimate_z_history = {
            name: [] for name in DISPLAY_KEYPOINT_NAMES
        }
        self.depth_image = np.zeros((640, 480))
        self.color_image = np.zeros((640, 480, 3), dtype=np.uint8)
        self.visible_keypoints = np.empty((0, 2))
        self.lines = []
        self.points_3d = np.full((len(DISPLAY_KEYPOINT_NAMES), 3), np.nan)
        self._setup_plots()

    def _setup_plots(self):
        plt.ion()
        self.figure, self.axes = plt.subplots(1, 2)
        self.depth_plot = self.axes[0].imshow(
            np.zeros((640, 480)),
            cmap="viridis",
            vmin=0,
            vmax=5000,
            interpolation="nearest",
            animated=True,
        )
        self.color_plot = self.axes[1].imshow(
            np.zeros((640, 480, 3), dtype=np.uint8),
            interpolation="nearest",
            animated=True,
        )
        self.skeleton_plot = LineCollection(
            [], colors="lime", linewidths=2, animated=True
        )
        self.keypoint_plot = self.axes[1].scatter(
            [], [], c="red", s=15, animated=True
        )
        self.axes[1].add_collection(self.skeleton_plot)
        self.axes[0].set_title("Depth")
        self.axes[1].set_title("RGB")
        self.figure.colorbar(self.depth_plot, ax=self.axes[0], label="Depth")
        self.figure.canvas.draw()
        self.background = self.figure.canvas.copy_from_bbox(self.figure.bbox)

        self.history_figure, self.history_axes = plt.subplots(
            3, 1, figsize=(8, 8)
        )
        self.history_axes[0].set_xlim(-99, 0)
        self.history_axes[0].set_ylim(-1, 1)
        self.history_axes[0].set_ylabel("X (m)")
        self.history_axes[1].set_xlim(-99, 0)
        self.history_axes[1].set_ylim(-1, 1)
        self.history_axes[1].set_ylabel("Y (m)")
        self.history_axes[2].set_xlim(-99, 0)
        self.history_axes[2].set_ylim(0, 5)
        self.history_axes[2].set_xlabel("Time (frames)")
        self.history_axes[2].set_ylabel("Z (m)")
        self.x_history_lines = [
            self.history_axes[0].plot(
                [],
                [],
                linestyle="none",
                marker=".",
                label=f"{name} measured",
                animated=True,
            )[0]
            for name in DISPLAY_KEYPOINT_NAMES
        ]
        self.y_history_lines = [
            self.history_axes[1].plot(
                [],
                [],
                linestyle="none",
                marker=".",
                label=f"{name} measured",
                animated=True,
            )[0]
            for name in DISPLAY_KEYPOINT_NAMES
        ]
        self.z_history_lines = [
            self.history_axes[2].plot(
                [],
                [],
                linestyle="none",
                marker=".",
                label=f"{name} measured",
                animated=True,
            )[0]
            for name in DISPLAY_KEYPOINT_NAMES
        ]
        self.estimate_x_history_lines = [
            self.history_axes[0].plot(
                [],
                [],
                color=line.get_color(),
                label=f"{name} estimated",
                animated=True,
            )[0]
            for name, line in zip(DISPLAY_KEYPOINT_NAMES, self.x_history_lines)
        ]
        self.estimate_y_history_lines = [
            self.history_axes[1].plot(
                [],
                [],
                color=line.get_color(),
                label=f"{name} estimated",
                animated=True,
            )[0]
            for name, line in zip(DISPLAY_KEYPOINT_NAMES, self.y_history_lines)
        ]
        self.estimate_z_history_lines = [
            self.history_axes[2].plot(
                [],
                [],
                color=line.get_color(),
                label=f"{name} estimated",
                animated=True,
            )[0]
            for name, line in zip(DISPLAY_KEYPOINT_NAMES, self.z_history_lines)
        ]
        self.history_axes[0].legend(ncol=2, fontsize=7)
        self.history_axes[1].legend(ncol=2, fontsize=7)
        self.history_axes[2].legend(ncol=2, fontsize=7)
        self.history_figure.canvas.draw()
        self.history_background = self.history_figure.canvas.copy_from_bbox(
            self.history_figure.bbox
        )

        self.figure_3d = plt.figure()
        self.axis_3d = self.figure_3d.add_subplot(projection="3d")
        self.axis_3d.set_xlim(-1, 1)
        self.axis_3d.set_ylim(1, -1)
        self.axis_3d.set_zlim(0, 5)
        self.axis_3d.set_xlabel("X (m)")
        self.axis_3d.set_ylabel("Y (m)")
        self.axis_3d.set_zlabel("Z depth (m)")
        self.axis_3d.view_init(elev=20, azim=-60, vertical_axis="y")
        self.keypoint_plots_3d = [
            self.axis_3d.plot(
                [],
                [],
                [],
                color="red",
                marker="x",
                linestyle="none",
                markersize=8,
                label="measured" if index == 0 else None,
                animated=True,
            )[0]
            for index, _ in enumerate(DISPLAY_KEYPOINT_NAMES)
        ]
        self.estimate_plots_3d = [
            self.axis_3d.plot(
                [],
                [],
                [],
                color="blue",
                marker="o",
                markerfacecolor="none",
                linestyle="none",
                markersize=8,
                label="estimated" if index == 0 else None,
                animated=True,
            )[0]
            for index, _ in enumerate(DISPLAY_KEYPOINT_NAMES)
        ]
        self.skeleton_indexes_3d = [
            (
                self.display_keypoint_indexes.index(start),
                self.display_keypoint_indexes.index(end),
            )
            for start, end in SKELETON
            if start in self.display_keypoint_indexes
            and end in self.display_keypoint_indexes
        ]
        self.skeleton_plots_3d = [
            self.axis_3d.plot(
                [], [], [], color="lime", linewidth=2, animated=True
            )[0]
            for _ in self.skeleton_indexes_3d
        ]
        self.estimate_skeleton_plots_3d = [
            self.axis_3d.plot(
                [],
                [],
                [],
                color="blue",
                linestyle="--",
                linewidth=2,
                animated=True,
            )[0]
            for _ in self.skeleton_indexes_3d
        ]
        self.axis_3d.legend()
        self.figure_3d.canvas.draw()
        self.background_3d = self.figure_3d.canvas.copy_from_bbox(
            self.figure_3d.bbox
        )
        self.figure.canvas.mpl_connect("resize_event", self._resize)
        self.history_figure.canvas.mpl_connect("resize_event", self._resize)
        self.figure_3d.canvas.mpl_connect("resize_event", self._resize)
        plt.ioff()

    def _resize(self, event):
        event.canvas.draw()
        if event.canvas == self.figure.canvas:
            self.background = event.canvas.copy_from_bbox(self.figure.bbox)
        elif event.canvas == self.history_figure.canvas:
            self.history_background = event.canvas.copy_from_bbox(
                self.history_figure.bbox
            )
        else:
            self.background_3d = event.canvas.copy_from_bbox(
                self.figure_3d.bbox
            )

    def loop(self):
        self.running = True
        while self.running:
            frames = self.pipeline.wait_for_frames()
            frames = self.align.process(frames)
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            depth_image = np.rot90(np.asanyarray(depth_frame.get_data()))
            color_image = np.rot90(np.asanyarray(color_frame.get_data()))

            result = inference_topdown(
                self.model, color_image[:, :, ::-1]
            )[0]
            keypoints = result.pred_instances.keypoints[0]
            scores = result.pred_instances.keypoint_scores[0]
            selected_keypoints = keypoints[self.display_keypoint_indexes]
            selected_scores = scores[self.display_keypoint_indexes]
            visible_keypoints = selected_keypoints[selected_scores > 0.3]

            points_3d = []
            for keypoint in selected_keypoints:
                x = int(np.clip(round(keypoint[0]), 0, 479))
                y = int(np.clip(round(keypoint[1]), 0, 639))
                depth = depth_image[y, x] * self.depth_scale
                original_x = 639 - y
                original_y = x
                if depth > 0:
                    point = rs.rs2_deproject_pixel_to_point(
                        self.color_intrinsics,
                        [original_x, original_y],
                        depth,
                    )
                    point = [point[1], -point[0], point[2]]
                else:
                    point = [np.nan, np.nan, np.nan]
                points_3d.append(point)
            points_3d = np.array(points_3d)
            lines = [
                [keypoints[start], keypoints[end]]
                for start, end in SKELETON
                if start in self.display_keypoint_indexes
                and end in self.display_keypoint_indexes
                if scores[start] > 0.3 and scores[end] > 0.3
            ]

            with self.lock:
                self.sample_id += 1
                self.timestamp = time.monotonic()
                self.sample_ids.append(self.sample_id)
                del self.sample_ids[:-100]
                for name, point in zip(DISPLAY_KEYPOINT_NAMES, points_3d):
                    self.xyz[name] = point.copy()
                for x_values, y_values, z_values, point in zip(
                    self.x_history,
                    self.y_history,
                    self.z_history,
                    points_3d,
                ):
                    x_values.append(point[0])
                    y_values.append(point[1])
                    z_values.append(point[2])
                    del x_values[:-100]
                    del y_values[:-100]
                    del z_values[:-100]
                self.depth_image = depth_image
                self.color_image = color_image
                self.visible_keypoints = visible_keypoints
                self.lines = lines
                self.points_3d = points_3d

        self.pipeline.stop()
        self.running = False

    def get(self, keypoint_name):
        with self.lock:
            return self.xyz[keypoint_name].copy()

    def get_sample(self, keypoint_name):
        with self.lock:
            return (
                self.sample_id,
                self.timestamp,
                self.xyz[keypoint_name].copy(),
            )

    def set_estimate(self, keypoint_name, xyz, sample_id=None):
        xyz = np.asarray(xyz)
        with self.lock:
            if sample_id is None:
                sample_id = self.sample_id
            self.estimated_xyz[keypoint_name] = xyz.copy()
            ids = self.estimate_sample_ids[keypoint_name]
            x_values = self.estimate_x_history[keypoint_name]
            y_values = self.estimate_y_history[keypoint_name]
            z_values = self.estimate_z_history[keypoint_name]
            if ids and ids[-1] == sample_id:
                x_values[-1] = xyz[0]
                y_values[-1] = xyz[1]
                z_values[-1] = xyz[2]
            else:
                ids.append(sample_id)
                x_values.append(xyz[0])
                y_values.append(xyz[1])
                z_values.append(xyz[2])
                del ids[:-100]
                del x_values[:-100]
                del y_values[:-100]
                del z_values[:-100]

    def draw(self):
        if not (
            plt.fignum_exists(self.figure.number)
            and plt.fignum_exists(self.history_figure.number)
            and plt.fignum_exists(self.figure_3d.number)
        ):
            self.running = False
            return

        with self.lock:
            if self.sample_id == self.drawn_sample_id:
                same_sample = True
            else:
                same_sample = False
                depth_image = self.depth_image[::2, ::2].copy()
                color_image = self.color_image[::2, ::2].copy()
                visible_keypoints = self.visible_keypoints.copy()
                lines = self.lines.copy()
                points_3d = self.points_3d.copy()
                estimated_points_3d = np.array(
                    [
                        self.estimated_xyz[name]
                        for name in DISPLAY_KEYPOINT_NAMES
                    ]
                )
                sample_id = self.sample_id
                self.drawn_sample_id = sample_id
                draw_graphs = sample_id - self.graph_drawn_sample_id >= 2
                if draw_graphs:
                    self.graph_drawn_sample_id = sample_id
                sample_ids = self.sample_ids.copy()
                x_history = [values.copy() for values in self.x_history]
                y_history = [values.copy() for values in self.y_history]
                z_history = [values.copy() for values in self.z_history]
                estimate_sample_ids = {
                    name: values.copy()
                    for name, values in self.estimate_sample_ids.items()
                }
                estimate_x_history = {
                    name: values.copy()
                    for name, values in self.estimate_x_history.items()
                }
                estimate_y_history = {
                    name: values.copy()
                    for name, values in self.estimate_y_history.items()
                }
                estimate_z_history = {
                    name: values.copy()
                    for name, values in self.estimate_z_history.items()
                }

        if same_sample:
            time.sleep(0.001)
            return

        self.figure.canvas.restore_region(self.background)
        self.depth_plot.set_data(depth_image)
        self.color_plot.set_data(color_image)
        self.keypoint_plot.set_offsets(visible_keypoints)
        self.skeleton_plot.set_segments(lines)
        self.axes[0].draw_artist(self.depth_plot)
        self.axes[1].draw_artist(self.color_plot)
        self.axes[1].draw_artist(self.skeleton_plot)
        self.axes[1].draw_artist(self.keypoint_plot)
        self.figure.canvas.blit(self.figure.bbox)

        if draw_graphs:
            self.history_figure.canvas.restore_region(
                self.history_background
            )
            measured_time = np.array(sample_ids) - sample_id
            for x_line, y_line, z_line, x_values, y_values, z_values in zip(
                self.x_history_lines,
                self.y_history_lines,
                self.z_history_lines,
                x_history,
                y_history,
                z_history,
            ):
                x_line.set_data(measured_time, x_values)
                y_line.set_data(measured_time, y_values)
                z_line.set_data(measured_time, z_values)
            for name, x_line, y_line, z_line in zip(
                DISPLAY_KEYPOINT_NAMES,
                self.estimate_x_history_lines,
                self.estimate_y_history_lines,
                self.estimate_z_history_lines,
            ):
                estimated_time = (
                    np.array(estimate_sample_ids[name]) - sample_id
                )
                x_line.set_data(estimated_time, estimate_x_history[name])
                y_line.set_data(estimated_time, estimate_y_history[name])
                z_line.set_data(estimated_time, estimate_z_history[name])
                self.history_axes[0].draw_artist(x_line)
                self.history_axes[1].draw_artist(y_line)
                self.history_axes[2].draw_artist(z_line)
            for x_line, y_line, z_line in zip(
                self.x_history_lines,
                self.y_history_lines,
                self.z_history_lines,
            ):
                self.history_axes[0].draw_artist(x_line)
                self.history_axes[1].draw_artist(y_line)
                self.history_axes[2].draw_artist(z_line)
            self.history_figure.canvas.blit(self.history_figure.bbox)

            self.figure_3d.canvas.restore_region(self.background_3d)
            for plot, point in zip(self.keypoint_plots_3d, points_3d):
                plot.set_data_3d(
                    np.array([point[0]]),
                    np.array([point[1]]),
                    np.array([point[2]]),
                )
                self.axis_3d.draw_artist(plot)
            for plot, (start, end) in zip(
                self.skeleton_plots_3d, self.skeleton_indexes_3d
            ):
                points = points_3d[[start, end]]
                plot.set_data_3d(points[:, 0], points[:, 1], points[:, 2])
                self.axis_3d.draw_artist(plot)
            for plot, point in zip(
                self.estimate_plots_3d, estimated_points_3d
            ):
                plot.set_data_3d(
                    np.array([point[0]]),
                    np.array([point[1]]),
                    np.array([point[2]]),
                )
                self.axis_3d.draw_artist(plot)
            for plot, (start, end) in zip(
                self.estimate_skeleton_plots_3d,
                self.skeleton_indexes_3d,
            ):
                points = estimated_points_3d[[start, end]]
                plot.set_data_3d(points[:, 0], points[:, 1], points[:, 2])
                self.axis_3d.draw_artist(plot)
            self.figure_3d.canvas.blit(self.figure_3d.bbox)
        self.figure.canvas.flush_events()

    def stop(self):
        self.running = False
