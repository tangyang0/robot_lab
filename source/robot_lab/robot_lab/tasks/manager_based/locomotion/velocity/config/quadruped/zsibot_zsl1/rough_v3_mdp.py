"""V3 command term; foot rewards remain inherited from v2."""

from isaaclab.envs.mdp.commands import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.utils import configclass
import torch

from .rough_v3_core import sample_velocity_mixture


class StraightMixtureVelocityCommand(UniformVelocityCommand):
    """Direct angular-velocity commands with an explicit forward/zero-yaw mode."""

    def __init__(self, cfg, env):
        if cfg.heading_command or cfg.rel_heading_envs != 0:
            raise ValueError("V3 requires heading_command=False and rel_heading_envs=0")
        super().__init__(cfg, env)
        self.is_straight_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _resample_command(self, env_ids):
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        ranges = self.cfg.ranges
        commands, standing, straight = sample_velocity_mixture(
            len(ids), (ranges.lin_vel_x, ranges.lin_vel_y, ranges.ang_vel_z),
            device=self.device, standing_fraction=self.cfg.rel_standing_envs,
            straight_fraction=self.cfg.straight_fraction,
            straight_vx_range=self.cfg.straight_vx_range,
        )
        self.vel_command_b[ids] = commands
        self.is_standing_env[ids] = standing
        self.is_straight_env[ids] = straight


@configclass
class StraightMixtureVelocityCommandCfg(UniformVelocityCommandCfg):
    class_type: type = StraightMixtureVelocityCommand
    straight_fraction: float = 0.5
    straight_vx_range: tuple[float, float] = (0.2, 0.9)
