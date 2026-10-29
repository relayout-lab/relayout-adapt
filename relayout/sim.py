"""LIBERO environment helpers shared by data conversion, generation and evaluation.

Images are kept exactly as the simulator renders them (no flip) for both training and
evaluation, so there is a single convention. Only human-facing videos are flipped.
"""
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

TASK_BDDL = "libero_90/LIVING_ROOM_SCENE5_put_the_red_mug_on_the_left_plate.bddl"
TASK_TEXT = "put the red mug on the left plate"
RESOLUTION = 128
FPS = 20
MAX_STEPS = 300
SETTLE_STEPS = 10
NOOP = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)


def bddl_path(rel=TASK_BDDL):
    from libero.libero import get_libero_path
    return str(Path(get_libero_path("bddl_files")) / rel)


def make_env(bddl=None, resolution=RESOLUTION, seed=0):
    from libero.libero.envs import OffScreenRenderEnv
    env = OffScreenRenderEnv(
        bddl_file_name=bddl or bddl_path(),
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(seed)
    return env


def quat_to_axisangle(q):
    """robosuite quaternion (x, y, z, w) -> axis-angle (3,)."""
    q = np.asarray(q, dtype=np.float64).copy()
    q[3] = np.clip(q[3], -1.0, 1.0)
    den = np.sqrt(1.0 - q[3] * q[3])
    if np.isclose(den, 0.0):
        return np.zeros(3)
    return q[:3] * 2.0 * np.arccos(q[3]) / den


def policy_obs(obs):
    """Raw robosuite observation -> arrays stored in the dataset / fed to the policy."""
    state = np.concatenate([
        obs["robot0_eef_pos"],
        quat_to_axisangle(obs["robot0_eef_quat"]),
        obs["robot0_gripper_qpos"],
    ]).astype(np.float32)
    return {
        "observation.images.agentview": np.ascontiguousarray(obs["agentview_image"]),
        "observation.images.wrist": np.ascontiguousarray(obs["robot0_eye_in_hand_image"]),
        "observation.state": state,
    }


def human_frame(obs):
    """Side-by-side upright frame for videos (simulator images are upside down)."""
    return np.concatenate([obs["agentview_image"][::-1, ::-1],
                           obs["robot0_eye_in_hand_image"][::-1, ::-1]], axis=1)


def reset_to(env, state, settle=SETTLE_STEPS):
    """Reset to a flattened MuJoCo state, then let objects settle with a no-op action."""
    env.reset()
    obs = env.set_init_state(state)
    for _ in range(settle):
        obs, _, _, _ = env.step(NOOP)
    return obs
