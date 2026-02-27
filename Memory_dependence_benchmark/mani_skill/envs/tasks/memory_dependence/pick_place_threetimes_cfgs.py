"""
SwapThreeCubes-v1 configs.
This file defines the parameters for the SwapThreeCubes task for different robots.
"""

PICK_PLACE_THREETIMES_CONFIGS = {
    "panda": {
        "cube_half_size": 0.02,
        "goal_thresh": 0.025,
        "cube_spawn_half_size": 0.1,
        "cube_spawn_center": (0, 0),
        "max_goal_height": 0.3,

        "sensor_cam_eye_pos": [0.3,0,0.6,],  # sensor cam is the camera used for visual observation generation
        "sensor_cam_target_pos": [-0.1, 0, 0.1],

        "human_cam_eye_pos": [0.6,0.7,0.6,],  # human cam is the camera used for human rendering (i.e. eval videos)
        "human_cam_target_pos": [0.0, 0.0, 0.35],
    },
    "fetch": {
        "cube_half_size": 0.02,
        "goal_thresh": 0.025,
        "cube_spawn_half_size": 0.1,
        "cube_spawn_center": (0, 0),
        "max_goal_height": 0.3,
        "sensor_cam_eye_pos": [0.3, 0, 0.6],
        "sensor_cam_target_pos": [-0.1, 0, 0.1],
        "human_cam_eye_pos": [0.6, 0.7, 0.6],
        "human_cam_target_pos": [0.0, 0.0, 0.35],
    },
    "xarm6_robotiq": {
        "cube_half_size": 0.02,
        "goal_thresh": 0.025,
        "cube_spawn_half_size": 0.1,
        "cube_spawn_center": (0, 0),
        "max_goal_height": 0.3,
        "sensor_cam_eye_pos": [0.3, 0, 0.6],
        "sensor_cam_target_pos": [-0.1, 0, 0.1],
        "human_cam_eye_pos": [0.6, 0.7, 0.6],
        "human_cam_target_pos": [0.0, 0.0, 0.35],
    },
    "so100": {
        "cube_half_size": 0.0125,
        "goal_thresh": 0.0125 * 1.25,
        "cube_spawn_half_size": 0.05,
        "cube_spawn_center": (-0.46, 0),
        "max_goal_height": 0.08,
        "sensor_cam_eye_pos": [-0.27, 0, 0.4],
        "sensor_cam_target_pos": [-0.56, 0, -0.25],
        "human_cam_eye_pos": [-0.1, 0.3, 0.4],
        "human_cam_target_pos": [-0.46, 0.0, 0.1],
    },
    "widowxai": {
        "cube_half_size": 0.018,
        "goal_thresh": 0.018 * 1.25,
        "cube_spawn_half_size": 0.05,
        "cube_spawn_center": (-0.25, 0),
        "max_goal_height": 0.2,
        "sensor_cam_eye_pos": [0.0, 0, 0.35],
        "sensor_cam_target_pos": [-0.2, 0, 0.1],
        "human_cam_eye_pos": [0.45, 0.5, 0.5],
        "human_cam_target_pos": [-0.2, 0.0, 0.2],
    },
    "widowxai_wristcam": {
        "cube_half_size": 0.018,
        "goal_thresh": 0.018 * 1.25,
        "cube_spawn_half_size": 0.05,
        "cube_spawn_center": (-0.25, 0),
        # "cube_spawn_center": (0, 0),
        "max_goal_height": 0.2,
        "sensor_cam_eye_pos": [0.0, 0, 0.35],
        "sensor_cam_target_pos": [-0.2, 0, 0.1],
        "human_cam_eye_pos": [0.45, 0.5, 0.5],
        "human_cam_target_pos": [-0.2, 0.0, 0.2],
    },
}
