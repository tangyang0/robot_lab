"""Softer lateral pushes: keep the stance/push incentives of V6 with less
straight-line disturbance (V6's +-0.5 pushes cost 0.8 m/s flat tracking)."""

import isaaclab.envs.mdp as isaac_mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass

from .rough_v6_env_cfg import ZsibotZSL1RoughV6EnvCfg


@configclass
class ZsibotZSL1RoughV7EnvCfg(ZsibotZSL1RoughV6EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # Same push schedule, gentler magnitude: +-0.5 (V6) traded 0.8 m/s
        # straight-line flat tracking for rough-terrain gains; +-0.3 targets
        # keeping both. Stance-width reward and ABAD exemption are inherited.
        self.events.randomize_push_robot = EventTerm(
            func=isaac_mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(8.0, 12.0),
            params={"velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}},
        )
