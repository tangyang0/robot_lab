"""Self-contained ZSL1 rough-v11 environment config.

Single-file flattening of the V2->V11 config lineage (each comment marks the
generation that introduced the setting). Weight-wise V11 cold-starts from the
flat policy's actor; this file reproduces the exact environment those weights
were trained in. Verified field-by-field against the chained configs.
"""

import copy
import math

import isaaclab.envs.mdp as isaac_mdp
import isaaclab.terrains as terrain_gen
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from .rough_env_cfg import ZsibotZSL1RoughEnvCfg
from .rough_v11_mdp import (
    StraightMixtureVelocityCommandCfg,
    feet_slide_world,
    feet_width,
    reversal_command,
)


@configclass
class ZsibotZSL1RoughV11EnvCfg(ZsibotZSL1RoughEnvCfg):
    """The full fix-stack rough environment: upright, wide, straight, limited."""

    def __post_init__(self):
        super().__post_init__()

        # ---------------- (V2) terrain: gentle geometry, easy start --------------
        self.scene.terrain.terrain_generator = terrain_gen.TerrainGeneratorCfg(
            size=(8.0, 8.0),
            border_width=20.0,
            num_rows=10,
            num_cols=40,  # (V5) 20 -> 40 columns
            curriculum=True,
            difficulty_range=(0.0, 1.0),
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
            use_cache=False,
            sub_terrains={
                "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.20),
                "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
                    proportion=0.10,  # (V5)
                    step_height_range=(0.02, 0.16),
                    step_width=0.30,
                    platform_width=2.0,
                    border_width=1.0,
                    holes=False,
                ),
                "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
                    proportion=0.15,  # (V5)
                    step_height_range=(0.02, 0.16),
                    step_width=0.30,
                    platform_width=2.0,
                    border_width=1.0,
                    holes=False,
                ),
                "boxes": terrain_gen.MeshRandomGridTerrainCfg(
                    proportion=0.10,  # (V5)
                    grid_width=0.45,
                    grid_height_range=(0.02, 0.12),
                    platform_width=2.0,
                ),
                # (V4) texture restored to the legacy challenging range.
                "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
                    proportion=0.40,  # (V5)
                    noise_range=(0.02, 0.10),
                    noise_step=0.02,
                    downsampled_scale=0.10,
                    border_width=0.25,
                ),
                "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
                    proportion=0.025,  # (V5)
                    slope_range=(0.0, 0.30),
                    platform_width=2.0,
                    border_width=0.25,
                ),
                "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
                    proportion=0.025,  # (V5)
                    slope_range=(0.0, 0.30),
                    platform_width=2.0,
                    border_width=0.25,
                ),
            },
        )
        self.scene.terrain.max_init_terrain_level = 0
        self.curriculum.command_levels_lin_vel = None
        self.curriculum.command_levels_ang_vel = None

        # ---------------- (V2) near-upright resets, no reset wrenches ----------
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
        self.events.randomize_com_positions = None
        # (V10) slot reuse: abrupt vx sign flips x1.0-1.6 every 2-3 s for 25% envs.
        self.events.randomize_apply_external_force_torque = EventTerm(
            func=reversal_command,
            mode="interval",
            interval_range_s=(2.0, 3.0),
            params={"term_command_name": "base_velocity", "fraction": 0.25,
                    "vx_range": (-0.6, 1.0), "amplify": (1.0, 1.6)},
        )
        # (V7) lateral pushes, gentle magnitude.
        self.events.randomize_push_robot = EventTerm(
            func=isaac_mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(8.0, 12.0),
            params={"velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}},
        )
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

        # ---------------- (V2+V3+V8) commands: narrow box, 50% straight --------
        command = StraightMixtureVelocityCommandCfg()
        old = self.commands.base_velocity
        for key, value in vars(old).items():
            if not key.startswith("_") and key != "class_type":
                setattr(command, key, copy.deepcopy(value))
        command.heading_command = False
        command.rel_heading_envs = 0.0
        command.ranges.heading = None
        command.ranges.lin_vel_x = (-0.6, 1.0)
        command.ranges.lin_vel_y = (-0.3, 0.3)
        command.ranges.ang_vel_z = (-0.8, 0.8)
        command.rel_standing_envs = 0.05
        command.straight_fraction = 0.50
        command.straight_vx_range = (0.20, 0.90)
        command.resampling_time_range = (4.0, 6.0)  # (V8) frequent reversals
        self.commands.base_velocity = command

        # ---------------- (V2) fall terminations --------------------------------
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

        # ---------------- rewards -----------------------------------------------
        self.rewards.track_lin_vel_xy_exp.weight = 3.0
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.35  # (V3)
        self.rewards.track_ang_vel_z_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.params["std"] = 0.25  # (V3)
        self.rewards.joint_pos_penalty.weight = -0.15
        self.rewards.joint_pos_penalty.params["stand_still_scale"] = 5.0
        # (V6) ABAD exempt from the default-pose pull so the stance can widen.
        self.rewards.joint_pos_penalty.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_HIP_JOINT", ".*_KNEE_JOINT"]
        )
        self.rewards.joint_mirror.weight = 0.0
        self.rewards.upward.weight = 0.0
        self.rewards.flat_orientation_l2.weight = -0.6  # (V9) kills travel-direction lean
        self.rewards.lin_vel_z_l2.weight = -1.0
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
        # (V6+V8) stance-width reward with the working soft kernel.
        self.rewards.feet_height_body = RewTerm(
            func=feet_width,
            weight=1.0,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=["FR_FOOT_LINK", "FL_FOOT_LINK", "RR_FOOT_LINK", "RL_FOOT_LINK"],
                    preserve_order=True,
                ),
                "target_width": 0.30,
                "softness": 0.15,
            },
        )
        self.rewards.feet_stumble.weight = -0.10
        self.rewards.feet_air_time.weight = 0.50
        self.rewards.feet_air_time.params["threshold"] = 0.25
        self.rewards.feet_air_time_variance.weight = -0.10
        self.rewards.feet_gait.weight = 0.40
        self.rewards.feet_gait.params["std"] = 0.10
        self.rewards.feet_gait.params["max_err"] = 0.20
        self.rewards.action_rate_l2.weight = -0.03  # (V8) smoother actions

        # ---------------- (V5) deployment joint-target limits --------------------
        self.actions.joint_pos.clip = {
            ".*_ABAD_JOINT": (-0.4790000021457672, 0.4790000021457672),
            ".*_HIP_JOINT": (-1.1489999294281006, 2.9690001010894775),
            ".*_KNEE_JOINT": (-2.8990001678466797, -0.6509999632835388),
        }
        # (V5) PhysX headroom for 32k-env instances.
        self.sim.physx.gpu_max_rigid_patch_count = 2**20
        self.sim.physx.gpu_collision_stack_size = 2**27

        # This subclass does not match the legacy parent's exact-name guard.
        self.disable_zero_weight_rewards()
