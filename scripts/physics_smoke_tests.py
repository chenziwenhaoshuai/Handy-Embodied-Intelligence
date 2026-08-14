from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import CONFIG, MODEL_XML_PATH, write_model_xml  # noqa: E402
from env import SelfRightingEnv  # noqa: E402


def _set_plate_pose(env: SelfRightingEnv, quat: np.ndarray) -> None:
    env.data.qpos[:] = 0.0
    env.data.qvel[:] = 0.0
    env.data.qpos[0:3] = np.array([0.0, 0.0, env._plate_center_z_for_quat(quat)])
    env.data.qpos[3:7] = quat
    mujoco.mj_forward(env.model, env.data)


def _quat_y(angle: float) -> np.ndarray:
    return np.array([np.cos(angle / 2.0), 0.0, np.sin(angle / 2.0), 0.0], dtype=np.float64)


def test_model_loads() -> tuple[bool, str]:
    path = write_model_xml(MODEL_XML_PATH, CONFIG)
    model = mujoco.MjModel.from_xml_path(str(path))
    return model.nq >= 9 and model.nu == 2, f"nq={model.nq}, nv={model.nv}, nu={model.nu}"


def test_plate_falls() -> tuple[bool, str]:
    env = SelfRightingEnv(max_tilt_degrees=60.0)
    env.reset(seed=1, options={"max_tilt_degrees": 0.0})
    _set_plate_pose(env, _quat_y(np.deg2rad(55.0)))
    start_normal = env._plate_normal_world().copy()
    for _ in range(180):
        _, _, _, _, info = env.step(np.zeros(2, dtype=np.float32))
    end_normal = env._plate_normal_world().copy()
    normal_delta = float(np.linalg.norm(end_normal - start_normal))
    env.close()
    return normal_delta > 0.02, f"normal_delta={normal_delta:.3f}, end_normal_z={end_normal[2]:.3f}"


def test_joint_axes() -> tuple[bool, str]:
    env = SelfRightingEnv(max_tilt_degrees=0.0)
    env.reset(seed=2, options={"max_tilt_degrees": 0.0})
    for _ in range(40):
        env.step(np.array([1.0, 0.0], dtype=np.float32))
    joint_x = float(env.data.qpos[env._joint_qpos[0]])
    env.reset(seed=3, options={"max_tilt_degrees": 0.0})
    for _ in range(40):
        env.step(np.array([0.0, 1.0], dtype=np.float32))
    joint_y = float(env.data.qpos[env._joint_qpos[1]])
    env.close()
    return abs(joint_x) > 0.25 and abs(joint_y) > 0.25, f"joint_x={joint_x:.3f}, joint_y={joint_y:.3f}"


def test_tip_reaches_multiple_directions() -> tuple[bool, str]:
    env = SelfRightingEnv(max_tilt_degrees=0.0)
    mujoco.mj_resetData(env.model, env.data)
    env.data.qpos[0:3] = np.array([0.0, 0.0, 0.8])
    env.data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
    mujoco.mj_forward(env.model, env.data)
    center = env.data.site_xpos[env._tip_site_id].copy()
    displacements: list[np.ndarray] = []
    limit_x = np.deg2rad(CONFIG.robot.joint_x_limit_degrees)
    limit_y = np.deg2rad(CONFIG.robot.joint_y_limit_degrees)
    for qx, qy in [(limit_x, 0.0), (-limit_x, 0.0), (0.0, limit_y), (0.0, -limit_y)]:
        env.data.qpos[env._joint_qpos] = np.array([qx, qy], dtype=np.float64)
        mujoco.mj_forward(env.model, env.data)
        displacements.append(env.data.site_xpos[env._tip_site_id][:2].copy() - center[:2])
    env.close()
    angles = [float(np.arctan2(d[1], d[0])) for d in displacements if np.linalg.norm(d) > 0.05]
    unique_quadrants = {round(((angle + np.pi) / (np.pi / 2.0))) % 4 for angle in angles}
    return len(unique_quadrants) >= 4, f"tip_xy_angles={[round(a, 2) for a in angles]}"


def test_tip_contact_and_no_penetration() -> tuple[bool, str]:
    env = SelfRightingEnv(max_tilt_degrees=80.0)
    env.reset(seed=5, options={"max_tilt_degrees": 0.0})
    _set_plate_pose(env, _quat_y(np.deg2rad(135.0)))
    min_tip_z = float("inf")
    max_force = 0.0
    contact_seen = False
    for _ in range(180):
        _, _, _, _, info = env.step(np.array([0.25, 1.0], dtype=np.float32))
        min_tip_z = min(min_tip_z, float(info["tip_position"][2]))
        max_force = max(max_force, float(info["tip_contact_force"]))
        contact_seen = contact_seen or bool(info["tip_contact"])
    env.close()
    return contact_seen and min_tip_z > -0.03, f"contact={contact_seen}, min_tip_z={min_tip_z:.4f}, max_force={max_force:.3f}"


def test_support_reaction_and_lever_attempt() -> tuple[bool, str]:
    env = SelfRightingEnv(max_tilt_degrees=80.0)
    env.reset(seed=6, options={"max_tilt_degrees": 0.0})
    _set_plate_pose(env, _quat_y(np.deg2rad(135.0)))
    start_normal_z = float(info_normal_z(env))
    max_force = 0.0
    best_normal_z = start_normal_z
    actions = [
        np.array([0.25, 1.0], dtype=np.float32),
        np.array([0.0, 0.75], dtype=np.float32),
        np.array([0.25, 0.75], dtype=np.float32),
        np.array([0.0, 0.5], dtype=np.float32),
    ]
    for action in actions:
        for _ in range(60):
            _, _, _, _, info = env.step(action)
            max_force = max(max_force, float(info["tip_contact_force"]))
            best_normal_z = max(best_normal_z, float(info["plate_normal"][2]))
    env.close()
    improved = best_normal_z > start_normal_z + 0.02
    return max_force > 5.0 and improved, f"start_normal_z={start_normal_z:.3f}, best_normal_z={best_normal_z:.3f}, max_force={max_force:.3f}"


def info_normal_z(env: SelfRightingEnv) -> float:
    return float(env._plate_normal_world()[2])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    tests = [
        ("Test 1 XML loads", test_model_loads),
        ("Test 2 plate can fall", test_plate_falls),
        ("Test 3 two joints move independently", test_joint_axes),
        ("Test 4 tip reaches horizontal directions", test_tip_reaches_multiple_directions),
        ("Test 5/6 tip contacts ground without large penetration", test_tip_contact_and_no_penetration),
        ("Test 7/8 support force and lever attempt", test_support_reaction_and_lever_attempt),
    ]
    failed = 0
    for name, fn in tests:
        ok, detail = fn()
        status = "PASS" if ok else "FAIL"
        print(f"{status}: {name} - {detail}")
        failed += 0 if ok else 1
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
