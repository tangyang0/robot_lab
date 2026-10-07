"""Torch-only command sampling and left/right maps, usable without Isaac Sim."""

from __future__ import annotations

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
