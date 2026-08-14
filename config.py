"""Central configuration for the MuJoCo self-righting plate robot.

Coordinate convention:
    X: forward/backward
    Y: left/right
    Z: upward
    gravity = [0, 0, -9.81]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
ASSETS_DIR = ROOT_DIR / "assets"
MODEL_XML_PATH = ASSETS_DIR / "self_righting.xml"


@dataclass(frozen=True)
class RobotConfig:
    plate_length: float = 0.18
    plate_width: float = 0.12
    plate_thickness: float = 0.024
    plate_mass: float = 0.28

    lower_rod_length: float = 0.16
    upper_rod_length: float = 0.25
    rod_radius: float = 0.007
    rod_width: float = 0.018
    rod_depth: float = 0.010
    lower_rod_mass: float = 0.035
    upper_rod_mass: float = 0.055
    rod_tip_radius: float = 0.018
    rod_tip_mass: float = 0.025
    servo_mount_mass: float = 0.035

    @property
    def rod_length(self) -> float:
        return self.lower_rod_length + self.upper_rod_length

    joint_x_limit_degrees: float = 78.0
    joint_y_limit_degrees: float = 115.0
    joint_damping: float = 0.035
    joint_armature: float = 0.0015

    ground_friction: tuple[float, float, float] = (1.25, 0.05, 0.002)
    plate_friction: tuple[float, float, float] = (1.0, 0.03, 0.001)
    rod_tip_friction: tuple[float, float, float] = (1.65, 0.08, 0.004)


@dataclass(frozen=True)
class ControlConfig:
    physics_dt: float = 0.002  # 500 Hz
    servo_dt: float = 0.005  # 200 Hz low-level PD updates
    policy_dt: float = 1.0 / 30.0  # about 30 Hz action hold

    servo_kp: float = 18.0
    servo_kd: float = 0.65
    actuator_torque_limit: float = 3.5
    servo_target_velocity_limit_degrees: float = 180.0

    @property
    def policy_steps(self) -> int:
        return max(1, round(self.policy_dt / self.physics_dt))

    @property
    def actual_policy_dt(self) -> float:
        return self.policy_steps * self.physics_dt

    @property
    def servo_target_velocity_limit(self) -> float:
        return self.servo_target_velocity_limit_degrees * 3.141592653589793 / 180.0


@dataclass(frozen=True)
class ResetConfig:
    min_tilt_degrees: float = 55.0
    max_tilt_degrees: float = 85.0
    random_yaw: bool = False
    initial_angular_velocity_std: float = 0.0
    random_servo_angles: bool = True
    servo_angle_fraction: float = 0.10
    clearance: float = 0.004
    settle_seconds: float = 0.75
    settle_max_seconds: float = 2.0
    settle_velocity_threshold: float = 0.08


@dataclass(frozen=True)
class IMUConfig:
    orientation_noise_std: float = 0.0
    angular_velocity_noise_std: float = 0.0
    orientation_bias: tuple[float, float] = (0.0, 0.0)
    angular_velocity_bias: tuple[float, float] = (0.0, 0.0)
    delay_steps: int = 0


@dataclass(frozen=True)
class RewardConfig:
    upright_weight: float = 2.0
    upright_progress_weight: float = 8.0
    rod_upright_weight: float = 1.0
    stand_rod_upright_weight: float = 4.0
    joint_center_weight: float = 0.15
    stand_joint_center_weight: float = 3.0
    stand_gate_low: float = 0.45
    stand_gate_high: float = 0.9
    angular_velocity_weight: float = 0.08
    action_magnitude_weight: float = 0.015
    action_change_weight: float = 0.035
    fallen_action_penalty_fraction: float = 0.15
    stand_joint_velocity_weight: float = 0.25
    stand_angular_velocity_weight: float = 0.45
    stillness_bonus_weight: float = 1.5
    tip_contact_weight: float = 0.0
    success_bonus: float = 150.0
    failure_penalty: float = 1.0

    success_angle_degrees: float = 7.0
    success_angular_velocity: float = 0.35
    success_hold_time: float = 0.35
    success_joint_angle_degrees: float = 10.0
    success_rod_angle_degrees: float = 12.0


@dataclass(frozen=True)
class EnvConfig:
    robot: RobotConfig = field(default_factory=RobotConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    reset: ResetConfig = field(default_factory=ResetConfig)
    imu: IMUConfig = field(default_factory=IMUConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)

    episode_seconds: float = 8.0

    @property
    def max_episode_steps(self) -> int:
        return max(1, round(self.episode_seconds / self.control.actual_policy_dt))

    @property
    def success_hold_steps(self) -> int:
        return max(1, round(self.reward.success_hold_time / self.control.actual_policy_dt))


CONFIG = EnvConfig()


def _triple(values: tuple[float, float, float]) -> str:
    return " ".join(f"{value:.6g}" for value in values)


def generate_mjcf(config: EnvConfig = CONFIG) -> str:
    """Generate the MJCF model from the centralized Python config."""

    robot = config.robot
    control = config.control
    half_l = robot.plate_length / 2.0
    half_w = robot.plate_width / 2.0
    half_t = robot.plate_thickness / 2.0
    joint_x_limit = robot.joint_x_limit_degrees * 3.141592653589793 / 180.0
    joint_y_limit = robot.joint_y_limit_degrees * 3.141592653589793 / 180.0
    lower_tip_z = robot.lower_rod_length
    upper_tip_z = robot.upper_rod_length + robot.rod_tip_radius
    rod_half_w = robot.rod_width / 2.0
    rod_half_d = robot.rod_depth / 2.0

    return f"""<mujoco model="self_righting_plate_robot">
  <compiler angle="radian" autolimits="true"/>

  <option timestep="{control.physics_dt:.8f}" gravity="0 0 -9.81"
          integrator="RK4" solver="Newton" iterations="80" tolerance="1e-10"/>

  <default>
    <geom condim="4" solimp="0.92 0.98 0.001" solref="0.008 1"/>
    <joint limited="true" damping="{robot.joint_damping:.8f}" armature="{robot.joint_armature:.8f}"/>
  </default>

  <asset>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512"
             rgb1="0.24 0.32 0.30" rgb2="0.18 0.24 0.23"/>
    <material name="ground_mat" texture="grid" texrepeat="6 6" reflectance="0.12"/>
    <material name="plate_mat" rgba="0.18 0.35 0.74 1"/>
    <material name="rod_mat" rgba="0.92 0.75 0.26 1"/>
    <material name="servo_mat" rgba="0.12 0.13 0.14 1"/>
    <material name="tip_mat" rgba="0.95 0.30 0.24 1"/>
  </asset>

  <worldbody>
    <light name="key" pos="0 -1.5 2.5" dir="0 0 -1" diffuse="0.9 0.9 0.9"/>
    <camera name="overview" pos="0.65 -0.75 0.45" xyaxes="0.76 0.65 0 -0.28 0.33 0.90"/>

    <geom name="ground" type="plane" size="2.0 2.0 0.05" material="ground_mat"
          friction="{_triple(robot.ground_friction)}"/>

    <body name="plate" pos="0 0 {half_t + 0.02:.8f}">
      <freejoint name="root"/>

      <geom name="plate_collision" type="box"
            size="{half_l:.8f} {half_w:.8f} {half_t:.8f}"
            mass="{robot.plate_mass:.8f}" material="plate_mat"
            friction="{_triple(robot.plate_friction)}"/>

      <site name="plate_center" pos="0 0 0" size="0.006" rgba="0.1 0.1 0.1 1"/>
      <site name="plate_top" pos="0 0 {half_t:.8f}" size="0.006" rgba="0.1 0.8 0.1 1"/>

      <!-- joint_x is the lower/base servo. It rotates about the plate-local X axis,
           tilting the lower rod left/right in Y-Z. -->
      <body name="joint_x_body" pos="0 0 {half_t:.8f}">
        <joint name="joint_x" type="hinge" axis="1 0 0"
               range="{-joint_x_limit:.8f} {joint_x_limit:.8f}"/>

        <geom name="base_servo_case" type="box" pos="0 0 0.012"
              size="0.030 0.021 0.017" mass="{robot.servo_mount_mass:.8f}"
              material="servo_mat" contype="0" conaffinity="0"/>

        <geom name="lower_rod" type="box"
              pos="0 0 {robot.lower_rod_length * 0.5:.8f}"
              size="{rod_half_w:.8f} {rod_half_d:.8f} {robot.lower_rod_length * 0.5:.8f}"
              mass="{robot.lower_rod_mass:.8f}"
              material="rod_mat" friction="0.8 0.02 0.001"/>
        <site name="lower_rod_end" pos="0 0 {lower_tip_z:.8f}" size="0.008" rgba="0.1 0.8 0.1 1"/>

        <!-- joint_y is the second/elbow servo mounted at the end of the lower rod.
             It rotates about the local Y axis and bends the upper rod relative to
             the lower rod. The serial X/Y hinge pair keeps two orthogonal tilt DOF
             while making the physical stick a two-section mechanism. -->
        <body name="joint_y_body" pos="0 0 {robot.lower_rod_length:.8f}">
          <joint name="joint_y" type="hinge" axis="0 1 0"
                 range="{-joint_y_limit:.8f} {joint_y_limit:.8f}"/>

          <geom name="elbow_servo_case" type="box" pos="0 0 0"
                size="0.024 0.019 0.017" mass="{robot.servo_mount_mass:.8f}"
                material="servo_mat" contype="0" conaffinity="0"/>

          <geom name="upper_rod" type="box"
                pos="0 0 {robot.upper_rod_length * 0.5:.8f}"
                size="{rod_half_w:.8f} {rod_half_d:.8f} {robot.upper_rod_length * 0.5:.8f}"
                mass="{robot.upper_rod_mass:.8f}"
                material="rod_mat" friction="0.8 0.02 0.001"/>

          <geom name="rod_tip" type="sphere" pos="0 0 {upper_tip_z:.8f}"
                size="{robot.rod_tip_radius:.8f}" mass="{robot.rod_tip_mass:.8f}"
                material="tip_mat" friction="{_triple(robot.rod_tip_friction)}"/>
          <site name="rod_tip_site" pos="0 0 {upper_tip_z:.8f}"
                size="0.010" rgba="1 0 0 1"/>
        </body>
      </body>
    </body>
  </worldbody>

  <actuator>
    <motor name="joint_x_motor" joint="joint_x" gear="1"
           ctrllimited="true" ctrlrange="{-control.actuator_torque_limit:.8f} {control.actuator_torque_limit:.8f}"/>
    <motor name="joint_y_motor" joint="joint_y" gear="1"
           ctrllimited="true" ctrlrange="{-control.actuator_torque_limit:.8f} {control.actuator_torque_limit:.8f}"/>
  </actuator>
</mujoco>
"""


def write_model_xml(path: Path = MODEL_XML_PATH, config: EnvConfig = CONFIG) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(generate_mjcf(config), encoding="utf-8")
    return path
