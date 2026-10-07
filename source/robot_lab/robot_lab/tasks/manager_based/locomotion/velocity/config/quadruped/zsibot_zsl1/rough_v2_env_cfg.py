"""Independent ZSL1 rough-ground configuration with the existing actor interface.

Install beside rough_env_cfg.py and rough_v2_mdp.py. This module never changes
the legacy rough/flat configuration, observation order, actuators, or action
scales. The actor remains a 45-value, single-frame observation at 50 Hz.
"""

import math

import isaaclab.envs.mdp as isaac_mdp
import isaaclab.terrains as terrain_gen
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from .rough_env_cfg import ZsibotZSL1RoughEnvCfg
from .rough_v2_mdp import feet_slide_world


@configclass
class ZsibotZSL1RoughV2EnvCfg(ZsibotZSL1RoughEnvCfg):
    """Upright locomotion with a gradual terrain curriculum and moderate DR."""

    def __post_init__(self):
        super().__post_init__()

        # Keep all observation/action/actuator/control settings from rough.
        # The rough parent's zero-weight cleanup is guarded by its exact class
        # name, so terms still exist here and can be configured before cleanup.
        self.scene.terrain.terrain_generator = terrain_gen.TerrainGeneratorCfg(
            size=(8.0, 8.0),
            border_width=20.0,
            num_rows=10,
            num_cols=20,
            curriculum=True,
            difficulty_range=(0.0, 1.0),
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
            use_cache=False,
            sub_terrains={
                "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.20),
                "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
                    proportion=0.15,
                    step_height_range=(0.02, 0.16),
                    step_width=0.30,
                    platform_width=2.0,
                    border_width=1.0,
                    holes=False,
                ),
                "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
                    proportion=0.20,
                    step_height_range=(0.02, 0.16),
                    step_width=0.30,
                    platform_width=2.0,
                    border_width=1.0,
                    holes=False,
                ),
                "boxes": terrain_gen.MeshRandomGridTerrainCfg(
                    proportion=0.15,
                    grid_width=0.45,
                    grid_height_range=(0.02, 0.12),
                    platform_width=2.0,
                ),
                # This terrain type uses a fixed noise range, independent of
                # difficulty; keep it mild rather than claiming a noise course.
                "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
                    proportion=0.20,
                    noise_range=(-0.02, 0.02),
                    noise_step=0.005,
                    downsampled_scale=0.20,
                    border_width=0.25,
                ),
                "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
                    proportion=0.05,
                    slope_range=(0.0, 0.30),
                    platform_width=2.0,
                    border_width=0.25,
                ),
                "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
                    proportion=0.05,
                    slope_range=(0.0, 0.30),
                    platform_width=2.0,
                    border_width=0.25,
                ),
            },
        )
        self.scene.terrain.max_init_terrain_level = 0
        # Keep the inherited terrain_levels term; it is a training curriculum,
        # not an obstacle-success metric. Acceptance must measure crossings.
        self.curriculum.command_levels_lin_vel = None
        self.curriculum.command_levels_ang_vel = None

        # Train normal locomotion from near-upright states, not a mixed
        # get-up task from roll/pitch uniformly distributed over +/-pi.
        self.events.randomize_reset_base.params = {
            "pose_range": {
                "x": (-0.20, 0.20), "y": (-0.20, 0.20), "z": (0.0, 0.03),
                "roll": (-0.08, 0.08), "pitch": (-0.08, 0.08),
                "yaw": (-math.pi, math.pi),
            },
            "velocity_range": {
                "x": (-0.10, 0.10), "y": (-0.10, 0.10), "z": (0.0, 0.0),
                "roll": (-0.10, 0.10), "pitch": (-0.10, 0.10), "yaw": (-0.10, 0.10),
            },
        }
        self.events.randomize_reset_joints.params["position_range"] = (1.0, 1.0)
        self.events.randomize_reset_joints.params["velocity_range"] = (0.0, 0.0)
        self.events.randomize_apply_external_force_torque = None
        self.events.randomize_push_robot = None
        self.events.randomize_com_positions = None
        self.events.randomize_rigid_body_material.params.update(
            static_friction_range=(0.50, 1.0),
            dynamic_friction_range=(0.40, 0.90),
            restitution_range=(0.0, 0.05),
            make_consistent=True,
        )
        for name in ("randomize_rigid_body_mass_base", "randomize_rigid_body_mass_others"):
            getattr(self.events, name).params.update(
                mass_distribution_params=(0.90, 1.10),
                operation="scale", distribution="uniform", recompute_inertia=True,
            )
        self.events.randomize_actuator_gains.params.update(
            stiffness_distribution_params=(0.80, 1.20),
            damping_distribution_params=(0.80, 1.20),
            operation="scale", distribution="uniform",
        )

        self.commands.base_velocity.ranges.lin_vel_x = (-0.6, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.8, 0.8)
        self.commands.base_velocity.rel_standing_envs = 0.05
        # Preserve heading-command semantics and ten-second resampling.

        self.terminations.illegal_contact = DoneTerm(
            func=isaac_mdp.illegal_contact,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[self.base_link_name]),
                "threshold": 1.0,
            },
        )
        self.terminations.bad_orientation = DoneTerm(
            func=isaac_mdp.bad_orientation,
            params={"limit_angle": math.pi / 3.0},
        )

        # Track motion while preserving the existing modest regularizers.
        self.rewards.track_lin_vel_xy_exp.weight = 3.0
        self.rewards.track_ang_vel_z_exp.weight = 1.5
        self.rewards.joint_pos_penalty.weight = -0.15
        self.rewards.joint_pos_penalty.params["stand_still_scale"] = 5.0
        self.rewards.joint_mirror.weight = 0.0
        self.rewards.feet_height_body.weight = 0.0
        self.rewards.upward.weight = 0.0
        self.rewards.flat_orientation_l2.weight = -0.20
        self.rewards.lin_vel_z_l2.weight = -1.0
        # RewardManager scales by dt: -25 gives a -0.5 terminal event at 50 Hz.
        self.rewards.is_terminated.weight = -25.0

        foot_names = ["FL_FOOT_LINK", "FR_FOOT_LINK", "RR_FOOT_LINK", "RL_FOOT_LINK"]
        self.rewards.feet_slide = RewTerm(
            func=feet_slide_world,
            weight=-0.10,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=foot_names, preserve_order=True),
                "asset_cfg": SceneEntityCfg("robot", body_names=foot_names, preserve_order=True),
                "contact_threshold": 1.0,
            },
        )
        self.rewards.feet_stumble.weight = -0.10
        self.rewards.feet_air_time.weight = 0.50
        self.rewards.feet_air_time.params["threshold"] = 0.25
        self.rewards.feet_air_time_variance.weight = -0.10
        self.rewards.feet_gait.weight = 0.40
        # GaitReward divides timing squared-error by std (not std**2).
        # With max_err=.2, all-four-feet-planted reward becomes about .0163,
        # versus .4 for matching alternating diagonal timing (legacy: .318/.5).
        self.rewards.feet_gait.params["std"] = 0.10
        self.rewards.feet_gait.params["max_err"] = 0.20

        # This subclass does not match the legacy parent's exact-name guard.
        self.disable_zero_weight_rewards()
