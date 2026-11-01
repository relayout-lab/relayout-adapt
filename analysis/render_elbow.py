"""Side view at the grasp of one source demo converted to an accepted bin and to a failing robot-side bin
(Fig. 6 of the report). Marks the target hand position (white ring), the actual hand (yellow dot) and
the elbow (joint 4) angle against its lower limit.

The two cases are attempts from the second-round generation with results/alloc/allocations.json.

    python -m analysis.render_elbow     # -> outputs/analysis/fig_elbow.png, fig_elbow_<bin>.png, fig_elbow.json
"""
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

CASES = [("r4a0", "r4a0_12_seed_5012", 38, "4 cm・0°：採用"), ("r12a180", "r12a180_59_seed_5109", 38, "12 cm・180°（ロボット側）：再生に失敗")]
W, H, CAM = 640, 480, "sideview"
CROP = (230, 170, 640, 480)  # around the arm and the mug
FONT = os.environ.get("FONT", "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")  # any CJK font for the labels


def font(size):
    return ImageFont.truetype(FONT, size, index=0) if Path(FONT).exists() else ImageFont.load_default()


def main():
    from robosuite.utils.camera_utils import get_camera_transform_matrix, project_points_from_world_to_camera
    from relayout import sim
    from relayout.generate import offset_schedule, track_action
    from relayout.rollout import MUG, PLATE, body_pos
    out_dir = Path("outputs/analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    env = sim.make_env()
    src = np.load("data/source/sources_0_50.npz", allow_pickle=True)["sources"].item()
    tiles, info = [], {}
    for b, init, sk, title in CASES:
        z = np.load(f"data/alloc/targets/{b}.npz")
        state = dict(zip(map(str, z["ids"]), z["states"]))[init]
        s = src[sk]
        obs = sim.reset_to(env, state)
        e = env.env
        m, d = e.sim.model, e.sim.data
        lo, hi = m.jnt_range[3]
        mug, plate = body_pos(env, MUG), body_pos(env, PLATE)
        off, g = offset_schedule(s, (mug - s["mug0"])[:2], (plate - s["plate0"])[:2])
        for t in range(g):  # follow the converted trajectory up to the step where the fingers start closing
            target = s["pos"][t + 1] + off[t + 1]
            obs, _, _, _ = env.step(track_action(s, t, target, obs))
        frame = e.sim.render(width=W, height=H, camera_name=CAM)[::-1]
        img = Image.fromarray(frame)
        mat = get_camera_transform_matrix(e.sim, CAM, H, W)
        (ty, tx), (ay, ax) = project_points_from_world_to_camera(np.array([target, obs["robot0_eef_pos"]]), mat, H, W)
        q = float(d.qpos[e.robots[0].joint_indexes[3]])
        err = 100 * float(np.linalg.norm(target - obs["robot0_eef_pos"]))
        crop = Image.fromarray(frame).crop(CROP)  # unlabelled close-up, for the video
        for im, (x0, y0) in ((img, (0, 0)), (crop, CROP[:2])):
            dr = ImageDraw.Draw(im)
            dr.ellipse((tx - x0 - 11, ty - y0 - 11, tx - x0 + 11, ty - y0 + 11), outline=(255, 255, 255), width=4)
            dr.ellipse((ax - x0 - 6, ay - y0 - 6, ax - x0 + 6, ay - y0 + 6), fill=(255, 210, 40))
        dr = ImageDraw.Draw(img)
        dr.rectangle((0, 0, W, 66), fill=(250, 250, 247))
        dr.text((12, 6), title, font=font(22), fill=(26, 26, 46))
        dr.text((12, 36), f"ひじ {q:.2f} rad（曲げられる端 {lo:.2f}）・手先のずれ {err:.1f} cm",
                font=font(18), fill=(200, 55, 45) if q - lo < 0.01 else (30, 140, 90))
        tiles.append(img)
        crop.save(out_dir / f"fig_elbow_{b}.png")
        info[b] = {"elbow": round(q, 3), "elbow_lo": round(float(lo), 3), "err_cm": round(err, 2)}
        print(title, f"step {g} elbow {q:.3f} (lo {lo:.2f}) err {err:.2f} cm")
    out = Image.new("RGB", (2 * W + 16, H), (250, 250, 247))
    for i, im in enumerate(tiles):
        out.paste(im, (i * (W + 16), 0))
    out.save(out_dir / "fig_elbow.png")
    (out_dir / "fig_elbow.json").write_text(json.dumps(info, indent=1))
    env.close()


if __name__ == "__main__":
    main()
