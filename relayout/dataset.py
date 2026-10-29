"""Write episodes (lists of policy observations + actions) as a LeRobotDataset."""
import shutil
from pathlib import Path

from relayout import sim


def features(resolution=sim.RESOLUTION):
    image = {"dtype": "video", "shape": (resolution, resolution, 3),
             "names": ["height", "width", "channel"]}
    return {
        "observation.images.agentview": image,
        "observation.images.wrist": dict(image),
        "observation.state": {"dtype": "float32", "shape": (8,), "names": None},
        "action": {"dtype": "float32", "shape": (7,), "names": None},
    }


def write(episodes, root, repo_id, overwrite=False):
    """episodes: iterable of lists of frames {policy_obs keys..., "action"}."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    root = Path(root)
    if root.exists():
        if not overwrite:
            raise FileExistsError(root)
        shutil.rmtree(root)
    ds = LeRobotDataset.create(repo_id=repo_id, fps=sim.FPS, features=features(),
                               root=root, robot_type="panda", use_videos=True,
                               image_writer_threads=4)
    n = 0
    for frames in episodes:
        for fr in frames:
            ds.add_frame({**fr, "task": sim.TASK_TEXT})
        ds.save_episode()
        n += 1
    ds.finalize()
    return n
