"""Gymnasium environment for the MuJoCo self-righting plate robot."""

from __future__ import annotations

from collections import deque
from dataclasses import replace
from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from config import CONFIG, MODEL_XML_PATH, EnvConfig, write_model_xml


class IMUSimulator:
    """Minimal IMU-like observation layer with optional noise, bias, and delay."""

    def __init__(self, config: EnvConfig):
        self.config = config.imu
        self._delay: deque[np.ndarray] = deque(maxlen=max(1, config.imu.delay_steps + 1))

    def reset(self) -> None:
        self._delay.clear()

    def observe(
        self,
        theta_x: float,
        theta_y: float,
        omega_x: float,
        omega_y: float,
        rng: np.random.Generator,
    ) -> tuple[float, float, float, float]:
        values = np.array([theta_x, theta_y, omega_x, omega_y], dtype=np.float64)

        if self.config.orientation_noise_std > 0.0:
            values[:2] += rng.normal(0.0, self.config.orientation_noise_std, size=2)
        if self.config.angular_velocity_noise_std > 0.0:
            values[2:] += rng.normal(0.0, self.config.angular_velocity_noise_std, size=2)

        values[:2] += np.array(self.config.orientation_bias, dtype=np.float64)
        values[2:] += np.array(self.config.angular_velocity_bias, dtype=np.float64)

        self._delay.append(values)
        if self.config.delay_steps <= 0:
            return tuple(values.tolist())  # type: ignore[return-value]
        while len(self._delay) <= self.config.delay_steps:
            self._delay.append(values.copy())
        delayed = self._delay[0]
        return tuple(delayed.tolist())  # type: ignore[return-value]


class SelfRightingEnv(gym.Env[np.ndarray, np.ndarray]):
    """Self-righting plate robot.

    Action:
        [target_joint_x, target_joint_y], normalized to [-1, 1].

    Observation:
        [
            sin(theta_x), cos(theta_x), sin(theta_y), cos(theta_y),
            omega_x, omega_y,
            joint_x, joint_y,
        ]
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        config: EnvConfig = CONFIG,
        model_path: str | Path = MODEL_XML_PATH,
        render_mode: str | None = None,
        max_tilt_degrees: float | None = None,
        min_tilt_degrees: float | None = None,
        regenerate_xml: bool = True,
    ) -> None:
        super().__init__()
        if max_tilt_degrees is not None or min_tilt_degrees is not None:
            config = replace(
                config,
                reset=replace(
                    config.reset,
                    max_tilt_degrees=(
                        config.reset.max_tilt_degrees if max_tilt_degrees is None else max_tilt_degrees
                    ),
                    min_tilt_degrees=(
                        config.reset.min_tilt_degrees if min_tilt_degrees is None else min_tilt_degrees
                    ),
                ),
            )
        self.config = config
        self.model_path = Path(model_path)
        if regenerate_xml:
            write_model_xml(self.model_path, self.config)

        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self._viewer: Any | None = None
        self._renderer: mujoco.Renderer | None = None

        joint_limits = np.deg2rad(
            [self.config.robot.joint_x_limit_degrees, self.config.robot.joint_y_limit_degrees]
        )
        self._target_low = -joint_limits.astype(np.float64)
        self._target_high = joint_limits.astype(np.float64)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)

        self._plate_body_id = self.model.body("plate").id
        self._ground_geom_id = self.model.geom("ground").id
        self._tip_geom_id = self.model.geom("rod_tip").id
        self._tip_site_id = self.model.site("rod_tip_site").id
        self._upper_rod_body_id = self.model.body("joint_y_body").id
        self._joint_qpos = np.array(
            [self.model.joint("joint_x").qposadr[0], self.model.joint("joint_y").qposadr[0]],
            dtype=np.int32,
        )
        self._joint_qvel = np.array(
            [self.model.joint("joint_x").dofadr[0], self.model.joint("joint_y").dofadr[0]],
            dtype=np.int32,
        )

        self._imu = IMUSimulator(self.config)
        self._step_count = 0
        self._success_count = 0
        self._servo_accumulator = self.config.control.servo_dt
        self._last_action = np.zeros(2, dtype=np.float64)
        self._commanded_target_angles = np.zeros(2, dtype=np.float64)
        self._servo_target_angles = np.zeros(2, dtype=np.float64)
        self._last_torque = np.zeros(2, dtype=np.float64)
        self._prev_upright = 0.0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        reset_options = options or {}
        max_tilt = np.deg2rad(float(reset_options.get("max_tilt_degrees", self.config.reset.max_tilt_degrees)))
        min_tilt = np.deg2rad(float(reset_options.get("min_tilt_degrees", self.config.reset.min_tilt_degrees)))
        min_tilt = min(min_tilt, max_tilt)
        tilt_magnitude = self.np_random.uniform(min_tilt, max_tilt)
        tilt_direction = self.np_random.uniform(-np.pi, np.pi)
        tilt_x = tilt_magnitude * np.cos(tilt_direction)
        tilt_y = tilt_magnitude * np.sin(tilt_direction)
        yaw = self.np_random.uniform(-np.pi, np.pi) if self.config.reset.random_yaw else 0.0
        quat = _quat_from_euler_xyz(tilt_x, tilt_y, yaw)

        self.data.qpos[0:3] = np.array([0.0, 0.0, self._plate_center_z_for_quat(quat)])
        self.data.qpos[3:7] = quat

        if self.config.reset.initial_angular_velocity_std > 0.0:
            self.data.qvel[3:6] = self.np_random.normal(
                0.0, self.config.reset.initial_angular_velocity_std, size=3
            )

        if self.config.reset.random_servo_angles:
            span = self._target_high * self.config.reset.servo_angle_fraction
            self.data.qpos[self._joint_qpos] = self.np_random.uniform(-span, span)

        mujoco.mj_forward(self.model, self.data)
        self._settle_after_reset()
        self._imu.reset()
        self._step_count = 0
        self._success_count = 0
        self._servo_accumulator = self.config.control.servo_dt
        self._last_action[:] = 0.0
        self._commanded_target_angles = self.data.qpos[self._joint_qpos].copy()
        self._servo_target_angles = self._commanded_target_angles.copy()
        self._last_torque[:] = 0.0
        self._prev_upright = float(np.clip(self._plate_normal_world()[2], -1.0, 1.0))
        self.data.ctrl[:] = 0.0

        obs = self._get_obs()
        return obs, self._get_info()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        clipped_action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        previous_action = self._last_action.copy()
        self._last_action = clipped_action
        self._commanded_target_angles = self._normalized_action_to_target(clipped_action)
        self._servo_accumulator = self.config.control.servo_dt

        for _ in range(self.config.control.policy_steps):
            if self._servo_accumulator + 1e-12 >= self.config.control.servo_dt:
                self._apply_servo_pd()
                self._servo_accumulator -= self.config.control.servo_dt
            mujoco.mj_step(self.model, self.data)
            self._servo_accumulator += self.config.control.physics_dt

        self._step_count += 1
        obs = self._get_obs()
        reward, terminated = self._compute_reward(clipped_action, previous_action)
        truncated = self._step_count >= self.config.max_episode_steps
        info = self._get_info()
        info["target_angles"] = self._commanded_target_angles.copy()
        info["servo_target_angles"] = self._servo_target_angles.copy()
        info["last_action"] = clipped_action.copy()
        info["success_count"] = self._success_count

        if self.render_mode == "human":
            self.render()
        return obs, reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        if self.render_mode == "human":
            if self._viewer is None:
                import mujoco.viewer

                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._viewer.sync()
            return None
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model)
            self._renderer.update_scene(self.data, camera="overview")
            return self._renderer.render()
        return None

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def get_tip_contact(self) -> bool:
        contact, _ = self._tip_contact_and_force()
        return contact

    def _normalized_action_to_target(self, action: np.ndarray) -> np.ndarray:
        return self._target_low + (action + 1.0) * 0.5 * (self._target_high - self._target_low)

    def _apply_servo_pd(self) -> None:
        self._advance_servo_target()
        joint_pos = self.data.qpos[self._joint_qpos]
        joint_vel = self.data.qvel[self._joint_qvel]
        torque = (
            self.config.control.servo_kp * (self._servo_target_angles - joint_pos)
            - self.config.control.servo_kd * joint_vel
        )
        limit = self.config.control.actuator_torque_limit
        self._last_torque = np.clip(torque, -limit, limit)
        self.data.ctrl[:] = self._last_torque

    def _advance_servo_target(self) -> None:
        max_delta = self.config.control.servo_target_velocity_limit * self.config.control.servo_dt
        delta = np.clip(
            self._commanded_target_angles - self._servo_target_angles,
            -max_delta,
            max_delta,
        )
        self._servo_target_angles += delta

    def _settle_after_reset(self) -> None:
        """Let the randomly initialized robot land before the RL episode starts.

        This prevents the policy from exploiting spawn-time falling/bouncing
        energy. During settling, RL is not queried and motors are unpowered.
        """

        self.data.ctrl[:] = 0.0
        fixed_steps = max(0, round(self.config.reset.settle_seconds / self.config.control.physics_dt))
        max_steps = max(fixed_steps, round(self.config.reset.settle_max_seconds / self.config.control.physics_dt))

        for settle_step in range(max_steps):
            mujoco.mj_step(self.model, self.data)
            if settle_step < fixed_steps:
                continue
            base_velocity = float(np.linalg.norm(self.data.qvel[:6]))
            joint_velocity = float(np.linalg.norm(self.data.qvel[self._joint_qvel]))
            if (
                base_velocity < self.config.reset.settle_velocity_threshold
                and joint_velocity < self.config.reset.settle_velocity_threshold
            ):
                break
        mujoco.mj_forward(self.model, self.data)

    def _get_obs(self) -> np.ndarray:
        theta_x, theta_y, omega_x, omega_y = self._imu_observation()
        joint_pos = self.data.qpos[self._joint_qpos]

        return np.array(
            [
                np.sin(theta_x),
                np.cos(theta_x),
                np.sin(theta_y),
                np.cos(theta_y),
                omega_x,
                omega_y,
                joint_pos[0],
                joint_pos[1],
            ],
            dtype=np.float32,
        )

    def _imu_observation(self) -> tuple[float, float, float, float]:
        theta_x, theta_y, omega_x, omega_y = self._ground_truth_tilt_omega()
        return self._imu.observe(theta_x, theta_y, omega_x, omega_y, self.np_random)

    def _ground_truth_tilt_omega(self) -> tuple[float, float, float, float]:
        body_z = self._plate_normal_world()
        # theta_x/theta_y are tilt components of the plate normal relative to world Z.
        theta_x = float(np.arctan2(-body_z[1], body_z[2]))
        theta_y = float(np.arctan2(body_z[0], body_z[2]))
        omega = self.data.cvel[self._plate_body_id, :3]
        return theta_x, theta_y, float(omega[0]), float(omega[1])

    def _plate_normal_world(self) -> np.ndarray:
        rotation = self.data.xmat[self._plate_body_id].reshape(3, 3)
        return rotation[:, 2].copy()

    def _compute_reward(self, action: np.ndarray, previous_action: np.ndarray) -> tuple[float, bool]:
        reward_config = self.config.reward
        body_z = self._plate_normal_world()
        upright = float(np.clip(body_z[2], -1.0, 1.0))
        upright_progress = upright - self._prev_upright
        rod_upright = float(np.clip(self._upper_rod_axis_world()[2], -1.0, 1.0))
        joint_pos = self.data.qpos[self._joint_qpos]
        joint_vel = self.data.qvel[self._joint_qvel]
        joint_limits = np.deg2rad(
            [self.config.robot.joint_x_limit_degrees, self.config.robot.joint_y_limit_degrees]
        )
        joint_center_error = float(np.mean((joint_pos / joint_limits) ** 2))
        stand_gate = _smoothstep(
            self.config.reward.stand_gate_low,
            self.config.reward.stand_gate_high,
            upright,
        )
        _, _, omega_x, omega_y = self._ground_truth_tilt_omega()
        angular_speed = float(np.linalg.norm([omega_x, omega_y]))
        joint_speed = float(np.linalg.norm(joint_vel))
        tip_contact = self.get_tip_contact()
        action_penalty_gate = reward_config.fallen_action_penalty_fraction + (
            1.0 - reward_config.fallen_action_penalty_fraction
        ) * stand_gate
        stillness = float(np.exp(-(4.0 * angular_speed + 0.8 * joint_speed)))

        reward = reward_config.upright_weight * upright
        reward += reward_config.upright_progress_weight * upright_progress
        reward += reward_config.rod_upright_weight * rod_upright
        reward += stand_gate * reward_config.stand_rod_upright_weight * rod_upright
        reward -= reward_config.joint_center_weight * joint_center_error
        reward -= stand_gate * reward_config.stand_joint_center_weight * joint_center_error
        reward -= reward_config.angular_velocity_weight * angular_speed
        reward -= stand_gate * reward_config.stand_angular_velocity_weight * angular_speed
        reward -= stand_gate * reward_config.stand_joint_velocity_weight * joint_speed
        reward += stand_gate * reward_config.stillness_bonus_weight * stillness
        reward -= action_penalty_gate * reward_config.action_magnitude_weight * float(np.dot(action, action))
        reward -= action_penalty_gate * reward_config.action_change_weight * float(np.sum((action - previous_action) ** 2))
        if tip_contact and reward_config.tip_contact_weight > 0.0:
            reward += reward_config.tip_contact_weight

        success = self._is_success()
        if success:
            self._success_count += 1
        else:
            self._success_count = 0

        terminated = self._success_count >= self.config.success_hold_steps
        if terminated:
            reward += reward_config.success_bonus

        if body_z[2] < -0.45:
            reward -= reward_config.failure_penalty
        self._prev_upright = upright
        return float(reward), terminated

    def _is_success(self) -> bool:
        theta_x, theta_y, omega_x, omega_y = self._ground_truth_tilt_omega()
        angle_limit = np.deg2rad(self.config.reward.success_angle_degrees)
        omega_limit = self.config.reward.success_angular_velocity
        joint_limit = np.deg2rad(self.config.reward.success_joint_angle_degrees)
        rod_axis_z_limit = np.cos(np.deg2rad(self.config.reward.success_rod_angle_degrees))
        return (
            abs(theta_x) < angle_limit
            and abs(theta_y) < angle_limit
            and abs(omega_x) < omega_limit
            and abs(omega_y) < omega_limit
            and abs(self.data.qpos[self._joint_qpos[0]]) < joint_limit
            and abs(self.data.qpos[self._joint_qpos[1]]) < joint_limit
            and self._upper_rod_axis_world()[2] > rod_axis_z_limit
        )

    def _get_info(self) -> dict[str, Any]:
        theta_x, theta_y, omega_x, omega_y = self._ground_truth_tilt_omega()
        contact, contact_force = self._tip_contact_and_force()
        return {
            "tip_contact": contact,
            "tip_contact_force": contact_force,
            "is_success": self._is_success(),
            "stand_gate": _smoothstep(
                self.config.reward.stand_gate_low,
                self.config.reward.stand_gate_high,
                self._plate_normal_world()[2],
            ),
            "plate_angle": np.array([theta_x, theta_y], dtype=np.float64),
            "angular_velocity": np.array([omega_x, omega_y], dtype=np.float64),
            "joint_angles": self.data.qpos[self._joint_qpos].copy(),
            "joint_velocities": self.data.qvel[self._joint_qvel].copy(),
            "joint_torques": self._last_torque.copy(),
            "tip_position": self.data.site_xpos[self._tip_site_id].copy(),
            "plate_normal": self._plate_normal_world(),
            "upper_rod_axis": self._upper_rod_axis_world(),
        }

    def _upper_rod_axis_world(self) -> np.ndarray:
        rotation = self.data.xmat[self._upper_rod_body_id].reshape(3, 3)
        return rotation[:, 2].copy()

    def _tip_contact_and_force(self) -> tuple[bool, float]:
        total_force = 0.0
        found = False
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            pair = {contact.geom1, contact.geom2}
            if self._tip_geom_id in pair and self._ground_geom_id in pair:
                found = True
                force = np.zeros(6, dtype=np.float64)
                mujoco.mj_contactForce(self.model, self.data, contact_index, force)
                total_force += float(np.linalg.norm(force[:3]))
        return found, total_force

    def _plate_center_z_for_quat(self, quat: np.ndarray) -> float:
        rotation = _quat_to_matrix(quat)
        corners = _box_corners(
            self.config.robot.plate_length,
            self.config.robot.plate_width,
            self.config.robot.plate_thickness,
        )
        min_corner_z = float(np.min((rotation @ corners.T)[2]))
        return -min_corner_z + self.config.reset.clearance


def _box_corners(length: float, width: float, thickness: float) -> np.ndarray:
    half = np.array([length / 2.0, width / 2.0, thickness / 2.0], dtype=np.float64)
    signs = np.array(
        [
            [-1, -1, -1],
            [-1, -1, 1],
            [-1, 1, -1],
            [-1, 1, 1],
            [1, -1, -1],
            [1, -1, 1],
            [1, 1, -1],
            [1, 1, 1],
        ],
        dtype=np.float64,
    )
    return signs * half


def _smoothstep(edge0: float, edge1: float, value: float) -> float:
    if edge0 == edge1:
        return 1.0 if value >= edge1 else 0.0
    x = np.clip((value - edge0) / (edge1 - edge0), 0.0, 1.0)
    return float(x * x * (3.0 - 2.0 * x))


def _quat_from_euler_xyz(rx: float, ry: float, rz: float) -> np.ndarray:
    cx, sx = np.cos(rx / 2.0), np.sin(rx / 2.0)
    cy, sy = np.cos(ry / 2.0), np.sin(ry / 2.0)
    cz, sz = np.cos(rz / 2.0), np.sin(rz / 2.0)
    return np.array(
        [
            cx * cy * cz + sx * sy * sz,
            sx * cy * cz - cx * sy * sz,
            cx * sy * cz + sx * cy * sz,
            cx * cy * sz - sx * sy * cz,
        ],
        dtype=np.float64,
    )


def _quat_to_matrix(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = quat
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
