### Config ###
DISPLAY_KEYPOINT_NAMES = [
    "nose",
    "right_shoulder",
    "right_hip",
    "right_knee",
    "right_ankle",
]



import matplotlib.pyplot as plt
import numpy as np
import pyrealsense2 as rs
from matplotlib.collections import LineCollection
from mmpose.apis import inference_topdown, init_model


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



model = init_model(CONFIG_FILE, CHECKPOINT_FILE, device="cuda:0")


pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
profile = pipeline.start(config)
align = rs.align(rs.stream.color)
depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
color_intrinsics = profile.get_stream(
    rs.stream.color
).as_video_stream_profile().get_intrinsics()

plt.ion()
figure, axes = plt.subplots(1, 2)
depth_plot = axes[0].imshow(
    np.zeros((640, 480)), cmap="viridis", vmin=0, vmax=5000
)
color_plot = axes[1].imshow(np.zeros((640, 480, 3), dtype=np.uint8))
skeleton_plot = LineCollection([], colors="lime", linewidths=2)
keypoint_plot = axes[1].scatter([], [], c="red", s=15)
axes[1].add_collection(skeleton_plot)
axes[0].set_title("Depth")
axes[1].set_title("RGB")
figure.colorbar(depth_plot, ax=axes[0], label="Depth")
figure.canvas.draw()

history_figure, history_axes = plt.subplots(3, 1, figsize=(8, 8))
history_axes[0].set_xlim(0, 99)
history_axes[0].set_ylim(0, 480)
history_axes[0].set_ylabel("X coordinate")
history_axes[1].set_xlim(0, 99)
history_axes[1].set_ylim(0, 640)
history_axes[1].set_ylabel("Y coordinate")
history_axes[2].set_xlim(0, 99)
history_axes[2].set_ylim(0, 5000)
history_axes[2].set_xlabel("Time (frames)")
history_axes[2].set_ylabel("Depth (mm)")
x_history_lines = [
    history_axes[0].plot([], [], label=name)[0] for name in DISPLAY_KEYPOINT_NAMES
]
y_history_lines = [
    history_axes[1].plot([], [], label=name)[0] for name in DISPLAY_KEYPOINT_NAMES
]
depth_history_lines = [
    history_axes[2].plot([], [], label=name)[0] for name in DISPLAY_KEYPOINT_NAMES
]
history_axes[0].legend(ncol=2, fontsize=7)
history_axes[1].legend(ncol=2, fontsize=7)
history_axes[2].legend(ncol=2, fontsize=7)
history_figure.canvas.draw()
history_background = history_figure.canvas.copy_from_bbox(history_figure.bbox)
display_keypoint_indexes = [
    KEYPOINT_NAMES.index(name) for name in DISPLAY_KEYPOINT_NAMES
]
x_history = [[] for _ in DISPLAY_KEYPOINT_NAMES]
y_history = [[] for _ in DISPLAY_KEYPOINT_NAMES]
depth_history = [[] for _ in DISPLAY_KEYPOINT_NAMES]

figure_3d = plt.figure()
axis_3d = figure_3d.add_subplot(projection="3d")
axis_3d.set_xlim(-1, 1)
axis_3d.set_ylim(1, -1)
axis_3d.set_zlim(0, 5)
axis_3d.set_xlabel("X (m)")
axis_3d.set_ylabel("Y (m)")
axis_3d.set_zlabel("Z depth (m)")
axis_3d.view_init(elev=20, azim=-60, vertical_axis="y")
keypoint_plots_3d = [axis_3d.plot([], [], [], "ro")[0] for _ in DISPLAY_KEYPOINT_NAMES]
skeleton_indexes_3d = [
    (display_keypoint_indexes.index(start), display_keypoint_indexes.index(end))
    for start, end in SKELETON
    if start in display_keypoint_indexes and end in display_keypoint_indexes
]
skeleton_plots_3d = [
    axis_3d.plot([], [], [], color="lime", linewidth=2)[0]
    for _ in skeleton_indexes_3d
]
figure_3d.canvas.draw()
background_3d = figure_3d.canvas.copy_from_bbox(figure_3d.bbox)

while (
    plt.fignum_exists(figure.number)
    and plt.fignum_exists(history_figure.number)
    and plt.fignum_exists(figure_3d.number)
):
    frames = pipeline.wait_for_frames()
    frames = align.process(frames)
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    depth_image = np.rot90(np.asanyarray(depth_frame.get_data()))
    color_image = np.rot90(np.asanyarray(color_frame.get_data()))
    result = inference_topdown(model, color_image[:, :, ::-1])[0]
    keypoints = result.pred_instances.keypoints[0]
    scores = result.pred_instances.keypoint_scores[0]
    selected_keypoints = keypoints[display_keypoint_indexes]
    selected_scores = scores[display_keypoint_indexes]
    visible_keypoints = selected_keypoints[selected_scores > 0.3]
    depth_values = []
    points_3d = []
    for keypoint in selected_keypoints:
        x = int(np.clip(round(keypoint[0]), 0, 479))
        y = int(np.clip(round(keypoint[1]), 0, 639))
        depth = depth_image[y, x] * depth_scale
        depth_values.append(depth * 1000)
        original_x = 639 - y
        original_y = x
        if depth > 0:
            point = rs.rs2_deproject_pixel_to_point(
                color_intrinsics, [original_x, original_y], depth
            )
            point = [point[1], -point[0], point[2]]
        else:
            point = [np.nan, np.nan, np.nan]
        points_3d.append(point)
    points_3d = np.array(points_3d)
    lines = [
        [keypoints[start], keypoints[end]]
        for start, end in SKELETON
        if start in display_keypoint_indexes
        and end in display_keypoint_indexes
        if scores[start] > 0.3 and scores[end] > 0.3
    ]
    for x_values, y_values, z_values, index, depth in zip(
        x_history,
        y_history,
        depth_history,
        display_keypoint_indexes,
        depth_values,
    ):
        x_values.append(keypoints[index][0])
        y_values.append(keypoints[index][1])
        z_values.append(depth)
        del x_values[:-100]
        del y_values[:-100]
        del z_values[:-100]

    depth_plot.set_data(depth_image)
    color_plot.set_data(color_image)
    keypoint_plot.set_offsets(visible_keypoints)
    skeleton_plot.set_segments(lines)
    axes[0].draw_artist(depth_plot)
    axes[1].draw_artist(color_plot)
    axes[1].draw_artist(skeleton_plot)
    axes[1].draw_artist(keypoint_plot)
    figure.canvas.blit(figure.bbox)
    figure.canvas.flush_events()

    history_figure.canvas.restore_region(history_background)
    for x_line, y_line, z_line, x_values, y_values, z_values in zip(
        x_history_lines,
        y_history_lines,
        depth_history_lines,
        x_history,
        y_history,
        depth_history,
    ):
        x_line.set_data(range(100 - len(x_values), 100), x_values)
        y_line.set_data(range(100 - len(y_values), 100), y_values)
        z_line.set_data(range(100 - len(z_values), 100), z_values)
        history_axes[0].draw_artist(x_line)
        history_axes[1].draw_artist(y_line)
        history_axes[2].draw_artist(z_line)
    history_figure.canvas.blit(history_figure.bbox)
    history_figure.canvas.flush_events()

    figure_3d.canvas.restore_region(background_3d)
    for plot, point in zip(keypoint_plots_3d, points_3d):
        plot.set_data_3d(
            np.array([point[0]]),
            np.array([point[1]]),
            np.array([point[2]]),
        )
        axis_3d.draw_artist(plot)
    for plot, (start, end) in zip(skeleton_plots_3d, skeleton_indexes_3d):
        points = points_3d[[start, end]]
        plot.set_data_3d(points[:, 0], points[:, 1], points[:, 2])
        axis_3d.draw_artist(plot)
    figure_3d.canvas.blit(figure_3d.bbox)
    figure_3d.canvas.flush_events()

pipeline.stop()
