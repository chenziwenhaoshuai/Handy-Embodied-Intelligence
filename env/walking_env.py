"""Commanded walking task for the two-servo plate robot."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from gymnasium import spaces

from config import CONFIG, MODEL_XML_PATH, EnvConfig
from env.self_righting_env import SelfRightingEnv


class WalkingEnv(SelfRightingEnv):
    """Start upright and learn to move according to a joystick-like command.

    Observation:
        [
            sin(theta_x), cos(theta_x), sin(theta_y), cos(theta_y),
            omega_x, omega_y,
            joint_x, joint_y,
            command_vx, command_vy,
            sin(phase), cos(phase),
        ]

    The command and phase are deployable controller-side signals. Ground-truth
    base position is used only inside the training reward.
    """

    def __init__(
        self,
        config: EnvConfig = CONFIG,
        model_path: str | Path = MODEL_XML_PATH,
        render_mode: str | None = None,
        command_speed: float = 0.08,
        command_change_seconds: float = 2.0,
        fixed_command: tuple[float, float] | None = None,
        gait_period_seconds: float = 1.0,
        regenerate_xml: bool = True,
    ) -> None:
        config = replace(
            config,
            reset=replace(
                config.reset,
                min_tilt_degrees=0.0,
                max_tilt_degrees=0.0,
                random_servo_angles=False,
                settle_seconds=0.0,
                settle_max_seconds=0.0,
            ),
        )
        super().__init__(
            config=config,
            model_path=model_path,
            render_mode=render_mode,
            regenerate_xml=regenerate_xml,
        )
        self.command_speed = float(command_speed)
        self.command_change_steps = max(
            1,
            round(command_change_seconds / self.config.control.actual_policy_dt),
        )
        self.fixed_command = None if fixed_command is None else np.asarray(fixed_command, dtype=np.float64)
        self.gait_period_steps = max(1, round(gait_period_seconds / self.config.control.actual_policy_dt))
        self.command_velocity = np.zeros(2, dtype=np.float64)
        self._prev_xy = np.zeros(2, dtype=np.float64)
        self._last_xy_velocity = np.zeros(2, dtype=np.float64)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)

    def set_command(self, vx: float, vy: float) -> None:
        self.command_velocity[:] = [vx, vy]
        self.fixed_command = self.command_velocity.copy()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        half_t = self.config.robot.plate_thickness / 2.0
        self.data.qpos[0:3] = np.array([0.0, 0.0, half_t + self.config.reset.clearance], dtype=np.float64)
        self.data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.data.qpos[self._joint_qpos] = 0.0
        self.data.qvel[:] = 0.0

        self._imu.reset()
        self._step_count = 0
        self._success_count = 0
        self._servo_accumulator = self.config.control.servo_dt
        self._last_action[:] = 0.0
        self._commanded_target_angles[:] = 0.0
        self._servo_target_angles[:] = 0.0
        self._last_torque[:] = 0.0
        self._prev_upright = 1.0
        self.data.ctrl[:] = 0.0
        self.command_velocity = self._initial_command()

        mujoco.mj_forward(self.model, self.data)
        self._prev_xy = self.data.qpos[0:2].copy()
        self._last_xy_velocity[:] = 0.0
        return self._get_obs(), self._get_info()

    def _initial_command(self) -> np.ndarray:
        if self.fixed_command is not None:
            return self.fixed_command.copy()
        return self._sample_command()

    def _sample_command(self) -> np.ndarray:
        directions = np.array(
            [
                [1.0, 0.0],
                [-1.0, 0.0],
                [0.0, 1.0],
                [0.0, -1.0],
                [0.0, 0.0],
            ],
            dtype=np.float64,
        )
        direction = directions[int(self.np_random.integers(0, len(directions)))]
        return direction * self.command_speed

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.fixed_command is None and self._step_count > 0 and self._step_count % self.command_change_steps == 0:
            self.command_velocity = self._sample_command()
        return super().step(action)

    def _get_obs(self) -> np.ndarray:
        theta_x, theta_y, omega_x, omega_y = self._imu_observation()
        joint_pos = self.data.qpos[self._joint_qpos]
        phase = 2.0 * np.pi * ((self._step_count % self.gait_period_steps) / self.gait_period_steps)

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
                self.command_velocity[0],
                self.command_velocity[1],
                np.sin(phase),
                np.cos(phase),
            ],
            dtype=np.float32,
        )

    def _compute_reward(self, action: np.ndarray, previous_action: np.ndarray) -> tuple[float, bool]:
        body_z = self._plate_normal_world()
        upright = float(np.clip(body_z[2], -1.0, 1.0))
        current_xy = self.data.qpos[0:2].copy()
        xy_velocity = (current_xy - self._prev_xy) / self.config.control.actual_policy_dt
        self._prev_xy = current_xy
        self._last_xy_velocity = xy_velocity

        velocity_error = float(np.linalg.norm(xy_velocity - self.command_velocity))
        command_norm = float(np.linalg.norm(self.command_velocity))
        if command_norm > 1e-6:
            forward_velocity = float(np.dot(xy_velocity, self.command_velocity / command_norm))
        else:
            forward_velocity = -float(np.linalg.norm(xy_velocity))

        theta_x, theta_y, omega_x, omega_y = self._ground_truth_tilt_omega()
        angular_speed = float(np.linalg.norm([omega_x, omega_y]))
        joint_speed = float(np.linalg.norm(self.data.qvel[self._joint_qvel]))

        reward = 3.0 * np.exp(-35.0 * velocity_error * velocity_error)
        reward += 2.0 * forward_velocity
        reward += 1.5 * upright
        reward -= 0.8 * (theta_x * theta_x + theta_y * theta_y)
        reward -= 0.05 * angular_speed
        reward -= 0.04 * joint_speed
        reward -= 0.01 * float(np.dot(action, action))
        reward -= 0.03 * float(np.sum((action - previous_action) ** 2))

        fell = upright < 0.45
        if fell:
            reward -= 10.0
        return float(reward), bool(fell)

    def _get_info(self) -> dict[str, Any]:
        info = super()._get_info()
        info["command_velocity"] = self.command_velocity.copy()
        info["xy_velocity"] = self._last_xy_velocity.copy()
        info["xy_position"] = self.data.qpos[0:2].copy()
        return info
