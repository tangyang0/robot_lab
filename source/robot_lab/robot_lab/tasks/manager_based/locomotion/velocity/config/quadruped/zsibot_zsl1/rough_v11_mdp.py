"""Self-contained mdp module for the ZSL1 rough-v11 task.

Flattened from the V2-V11 lineage: command sampling and mirror maps
(rough_v3_core), world-frame foot slide (rough_v2_mdp), stance-width reward
(rough_v6_mdp), abrupt command reversal (rough_v10_mdp) and the
straight-mixture command term (rough_v3_mdp). Contents are byte-identical
to the lineage modules; only imports were unified.
"""

from __future__ import annotations

# ---- rough_v3_core ----
from dataclasses import dataclass
from functools import lru_cache
import math

import torch


def sample_velocity_mixture(
    count, ranges, *, device, standing_fraction=0.05, straight_fraction=0.5,
    straight_vx_range=(0.2, 0.9), generator=None,
):
    """Sample direct body velocities; every output remains inside current ranges.

    Straight fraction is conditional on non-standing samples. A straight mode
    is available only where the current ranges contain (positive vx, 0, 0).
    Fixed evaluation commands take precedence over both mixture and standing.
    The old 0.2 m/s xy deadband is retained only where zero xy is in range.
    """
    bounds = [tuple(map(float, pair)) for pair in ranges]
    if len(bounds) != 3 or any(len(p) != 2 or p[0] > p[1] for p in bounds):
        raise ValueError("Expected ordered vx/vy/yaw ranges")
    if not all(math.isfinite(v) for p in bounds for v in p):
        raise ValueError("Command ranges must be finite")
    if not 0 <= standing_fraction <= 1 or not 0 <= straight_fraction <= 1:
        raise ValueError("Command fractions must lie in [0, 1]")
    if len(straight_vx_range) != 2 or not 0 < straight_vx_range[0] <= straight_vx_range[1]:
        raise ValueError("Straight vx interval must be positive and ordered")

    def rand(*shape):
        return torch.rand(*shape, device=device, generator=generator)

    lo = torch.tensor([b[0] for b in bounds], device=device)
    hi = torch.tensor([b[1] for b in bounds], device=device)
    commands = lo + rand(count, 3) * (hi - lo)
    standing = torch.zeros(count, dtype=torch.bool, device=device)
    straight = torch.zeros_like(standing)
    if all(a == b for a, b in bounds):
        # Exact constants, including .1 m/s commands below the old deadband.
        standing[:] = all(a == 0 for a, _ in bounds)
        return commands, standing, straight

    zero_in = [a <= 0 <= b for a, b in bounds]
    if all(zero_in):
        standing = rand(count) < standing_fraction
    if all(zero_in[:2]):
        low_xy = commands[:, :2].norm(dim=1) <= 0.2
        commands[low_xy, :2] = 0

    vx_lo = max(bounds[0][0], float(straight_vx_range[0]))
    vx_hi = min(bounds[0][1], float(straight_vx_range[1]))
    if zero_in[1] and zero_in[2] and vx_lo <= vx_hi:
        straight = (~standing) & (rand(count) < straight_fraction)
        vx = vx_lo + rand(count) * (vx_hi - vx_lo)
        commands[straight, 0] = vx[straight]
        commands[straight, 1:] = 0
    commands[standing] = 0
    return commands, standing, straight


@lru_cache(maxsize=64)
def _map_buffers(indices, signs, device, dtype):
    return (torch.tensor(indices, device=device, dtype=torch.long),
            torch.tensor(signs, device=device, dtype=dtype))


@dataclass(frozen=True)
class SignedPermutation:
    indices: tuple[int, ...]
    signs: tuple[int, ...]

    def __post_init__(self):
        n = len(self.indices)
        if sorted(self.indices) != list(range(n)) or len(self.signs) != n:
            raise ValueError("Mirror must be a complete permutation")
        if any(s not in (-1, 1) for s in self.signs):
            raise ValueError("Mirror signs must be +/-1")
        if any(self.indices[self.indices[i]] != i or
               self.signs[i] * self.signs[self.indices[i]] != 1 for i in range(n)):
            raise ValueError("Mirror must be an involution")

    def apply(self, values):
        if values.shape[-1] != len(self.indices):
            raise ValueError(f"Expected {len(self.indices)} values, got {values.shape}")
        indices, signs = _map_buffers(self.indices, self.signs, str(values.device), values.dtype)
        return values.index_select(-1, indices) * signs


def joint_mirror(joint_names):
    """Resolve FAR<->FBL and RAR<->RBL; only ABAD changes sign."""
    names = list(joint_names)
    legs = {"FAR": "FBL", "FBL": "FAR", "RAR": "RBL", "RBL": "RAR"}
    expected = {f"{leg}_{joint}_JOINT" for leg in legs for joint in ("ABAD", "HIP", "KNEE")}
    if len(names) != 12 or set(names) != expected:
        raise ValueError(f"Unexpected ZSL1 joint set: {names}")
    opposite = [legs[name.split("_", 1)[0]] + "_" + name.split("_", 1)[1] for name in names]
    return SignedPermutation(tuple(names.index(name) for name in opposite),
                             tuple(-1 if "_ABAD_" in name else 1 for name in names))


def ray_mirror(ray_starts):
    """Find the actual y-reflected ray indices, independent of grid ordering."""
    points = torch.as_tensor(ray_starts, dtype=torch.float64, device="cpu")
    if points.ndim != 2 or points.shape[1] != 3 or len(points) != 187:
        raise ValueError("Expected the current 187-ray scanner")
    reflected = points * torch.tensor([1, -1, 1], dtype=points.dtype)
    distance = (reflected[:, None] - points[None]).abs().amax(dim=-1)
    error, indices = distance.min(dim=1)
    if error.max().item() > 1e-5:
        raise ValueError("Scanner rays are not left/right symmetric")
    return SignedPermutation(tuple(indices.tolist()), (1,) * len(points))


def observation_mirror(terms, action_names, ray_starts=None):
    """Build offsets from actual ordered term names and widths, never assumed slices."""
    signs_by_name = {"base_lin_vel": (1, -1, 1), "base_ang_vel": (-1, 1, -1),
                     "projected_gravity": (1, -1, 1), "velocity_commands": (1, -1, -1)}
    indices, signs = [], []
    for term in terms:
        name, width = term["name"], term["width"]
        if name in signs_by_name:
            piece = SignedPermutation((0, 1, 2), signs_by_name[name])
        elif name in ("joint_pos", "joint_vel"):
            piece = joint_mirror(term["joint_names"])
        elif name == "actions":
            piece = joint_mirror(action_names)
        elif name == "height_scan":
            piece = ray_mirror(ray_starts)
        else:
            raise ValueError(f"Unsupported observation term: {name}")
        if width != len(piece.indices):
            raise ValueError(f"Unexpected width for {name}: {width}")
        offset = len(indices)
        indices.extend(offset + i for i in piece.indices)
        signs.extend(piece.signs)
    return SignedPermutation(tuple(indices), tuple(signs))


def augment_with_maps(obs, actions, maps):
    """RSL contract: first original batch, then reflected batch; handle None inputs."""
    if obs is None:
        obs_aug = None
    else:
        if set(obs.keys()) != {"policy", "critic"}:
            raise ValueError("Expected policy and critic TensorDict groups")
        n = obs.batch_size[0]
        obs_aug = obs.repeat(2)
        for group in ("policy", "critic"):
            obs_aug[group][n:] = maps[group].apply(obs[group])
    actions_aug = None if actions is None else torch.cat((actions, maps["actions"].apply(actions)), dim=0)
    return obs_aug, actions_aug

# ---- rough_v2_mdp ----
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_slide_world(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    contact_threshold: float = 1.0,
) -> torch.Tensor:
    """Penalize horizontal foot motion relative to the static world ground.

    A planted foot has zero world velocity even while the base moves. Unlike
    the legacy reward, this deliberately does not subtract base velocity.
    Matching ordered body names in both SceneEntityCfg objects is required.
    The short contact history prevents a one-step contact flicker from
    immediately removing the slip penalty.
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    robot = env.scene[asset_cfg.name]
    contact = (
        sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
        .norm(dim=-1)
        .amax(dim=1)
        > contact_threshold
    )
    horizontal_speed = robot.data.body_lin_vel_w[:, asset_cfg.body_ids, :2].norm(dim=-1)
    return torch.sum(horizontal_speed * contact, dim=1)

# ---- rough_v6_mdp ----
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_width(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target_width: float = 0.30,
    softness: float = 0.05,
) -> torch.Tensor:
    """One-sided stance-width reward in the body frame.

    Ordered feet: [FR, FL, RR, RL]. Width is the mean lateral distance of the
    front and rear foot pairs expressed in the body frame, so world yaw does
    not matter. Returns 1.0 whenever both pairs are at least ``target_width``
    apart; narrower (adducted) stances decay with a squared-hinge kernel. The
    URDF default pose puts the feet 0.319 m apart, so the term is saturated
    at the natural stance and only acts against crossing the legs inward.
    """
    robot = env.scene[asset_cfg.name]
    feet_w = robot.data.body_pos_w[:, asset_cfg.body_ids, :] - robot.data.root_pos_w[:, None, :]
    # world -> body: rotate by the conjugate of the root quaternion (w,x,y,z)
    qw = robot.data.root_quat_w[:, 0:1].unsqueeze(-1)   # (N,1,1)
    qvec = -robot.data.root_quat_w[:, 1:4].unsqueeze(1)  # (N,1,3), broadcasts per foot
    t = 2.0 * torch.cross(qvec, feet_w, dim=-1)
    feet_b = feet_w + qw * t + torch.cross(qvec, t, dim=-1)
    y = feet_b[..., 1]
    width = 0.5 * (torch.abs(y[:, 1] - y[:, 0]) + torch.abs(y[:, 3] - y[:, 2]))
    deficit = torch.clamp(target_width - width, min=0.0)
    return torch.exp(-torch.square(deficit / softness))

# ---- rough_v10_mdp ----
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def reversal_command(env: ManagerBasedEnv, env_ids, term_command_name: str, fraction: float, vx_range, amplify) -> None:
    """Abruptly reverse the forward command for a random subset of envs.

    The real robot is driven back -> release -> forward; hard sign flips of vx
    must therefore be routine in training, not something the policy only sees
    at the 4-6 s resampling boundaries. Flipped values are clipped back into
    the trained vx box; standing commands (|vx| <= 0.05) are left alone.
    """
    cmd_term = env.command_manager.get_term(term_command_name)
    vx = cmd_term.vel_command_b[env_ids, 0]
    mask = (torch.rand_like(vx) < fraction) & (vx.abs() > 0.05)
    gain = 1.0 + torch.rand_like(vx) * (amplify[1] - amplify[0]) + amplify[0] - 1.0
    flipped = (-vx * gain).clamp(vx_range[0], vx_range[1])
    cmd_term.vel_command_b[env_ids[mask], 0] = flipped[mask]

# ---- rough_v3_mdp ----
from isaaclab.envs.mdp.commands import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.utils import configclass
import torch



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
