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


def use_stats_from(root, src_root):
    """Replace a dataset's normalisation stats with another dataset's (the original is kept as stats_own.json).

    Fine-tuning in lerobot-train re-normalises with the new dataset's stats. Copying the base policy's
    training stats keeps the input/output scaling identical across fine-tuning conditions.
    """
    meta, src = Path(root) / "meta", Path(src_root) / "meta" / "stats.json"
    if not (meta / "stats_own.json").exists():
        shutil.copy(meta / "stats.json", meta / "stats_own.json")
    shutil.copy(src, meta / "stats.json")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="dataset utilities")
    ap.add_argument("--use-stats-from", nargs=2, metavar=("SRC", "DST"), required=True,
                    help="copy SRC/meta/stats.json into DST (keeps DST/meta/stats_own.json)")
    a = ap.parse_args()
    use_stats_from(a.use_stats_from[1], a.use_stats_from[0])
    print("stats of", a.use_stats_from[1], "now from", a.use_stats_from[0])
