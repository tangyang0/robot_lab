"""RSL left/right augmentation for the actual ZSL1 policy45/critic235 schema."""

from __future__ import annotations

import torch

from .rough_v3_core import augment_with_maps, joint_mirror, observation_mirror


def _field(obj, name):
    return obj[name] if isinstance(obj, dict) else getattr(obj, name)


def _names_for_entity(entity, robot):
    if entity.name != "robot":
        raise ValueError("Only the ZSL1 robot joint entity is supported")
    ids = entity.joint_ids
    if isinstance(ids, slice):
        return list(robot.joint_names[ids])
    if isinstance(ids, torch.Tensor):
        ids = ids.detach().cpu().tolist()
    return [robot.joint_names[int(i)] for i in ids]


def _require_symmetric_scale_clip(cfg, mapping):
    scale = getattr(cfg, "scale", None)
    if scale is not None:
        scales = torch.as_tensor(scale).detach().cpu().flatten()
        if scales.numel() != 1 and (scales.numel() != len(mapping.indices) or
                not torch.equal(scales, scales[list(mapping.indices)])):
            raise ValueError("Observation scales are not reflection-compatible")
    clip = getattr(cfg, "clip", None)
    if clip is not None and (len(clip) != 2 or clip[0] != -clip[1]):
        raise ValueError("Observation clipping must be symmetric")
    if getattr(cfg, "modifiers", None):
        raise ValueError("Unreviewed observation modifiers cannot be mirrored")


def _require_symmetric_action_clip(clip, action_map):
    # Only ABAD changes sign under left/right reflection. HIP/KNEE keep
    # their sign, so their absolute target windows need not be zero-centred.
    endpoints = torch.stack((action_map.apply(clip[..., 0]), action_map.apply(clip[..., 1])), dim=-1)
    reflected = torch.sort(endpoints, dim=-1).values
    if not torch.equal(clip, reflected):
        raise ValueError("Action clips are not left/right symmetric")


def build_runtime_maps(env):
    """Resolve each joint term separately, including the critic's native order."""
    raw = env.unwrapped
    manager, robot = raw.observation_manager, raw.scene["robot"]
    if list(raw.action_manager.active_terms) != ["joint_pos"]:
        raise ValueError("Expected exactly the joint_pos action term")
    action = raw.action_manager.get_term("joint_pos")
    # Installed IsaacLab exposes resolved names in this field; avoid guessing
    # native articulation order or reinterpreting regexes independently.
    action_names = list(action._joint_names)
    action_map = joint_mirror(action_names)
    if action.action_dim != 12 or not action.cfg.use_default_offset:
        raise ValueError("Expected 12 default-offset position actions")
    for attr, signed in (("_scale", False), ("_offset", True)):
        value = torch.as_tensor(getattr(action, attr)).detach().cpu()
        if value.ndim == 0:
            value = value.expand(12)
        else:
            value = value.reshape(-1, 12)
        reflected = action_map.apply(value) if signed else value[..., list(action_map.indices)]
        if not torch.allclose(value, reflected, atol=1e-7, rtol=0):
            raise ValueError(f"Action {attr} is not left/right symmetric")
    if action.cfg.clip is not None:
        _require_symmetric_action_clip(action._clip.detach().cpu(), action_map)

    maps = {"actions": action_map}
    for group, width in (("policy", 45), ("critic", 235)):
        if not manager.group_obs_concatenate[group] or tuple(manager.group_obs_dim[group]) != (width,):
            raise ValueError(f"Unexpected {group} observation shape")
        group_cfg = _field(manager.cfg, group)
        names = manager.active_terms[group]
        dims = manager.group_obs_term_dim[group]
        if len(names) != len(dims):
            raise ValueError("Observation name/dimension mismatch")
        schema, configs, rays = [], [], None
        for name, dim in zip(names, dims):
            if len(dim) != 1:
                raise ValueError("Only single-frame flat terms are supported")
            cfg = _field(group_cfg, name)
            if cfg.history_length not in (None, 0):
                raise ValueError("Observation history changes need a new symmetry review")
            term = {"name": name, "width": int(dim[0])}
            if name in ("joint_pos", "joint_vel"):
                term["joint_names"] = _names_for_entity(cfg.params["asset_cfg"], robot)
                jm = joint_mirror(term["joint_names"])
                native_ids = [robot.joint_names.index(n) for n in term["joint_names"]]
                attr = "default_joint_pos" if name == "joint_pos" else "default_joint_vel"
                default = getattr(robot.data, attr)[:, native_ids].detach().cpu()
                if not torch.allclose(default, jm.apply(default), atol=1e-7, rtol=0):
                    raise ValueError(f"{name} relative offsets are not mirror-compatible")
            elif name == "height_scan":
                sensor = raw.scene.sensors[cfg.params["sensor_cfg"].name]
                if sensor.cfg.ray_alignment != "yaw":
                    raise ValueError("Expected yaw-aligned height scanner")
                rays = sensor.ray_starts[0].detach().cpu()
                directions = sensor.ray_directions[0].detach().cpu()
                if not torch.allclose(directions, directions.new_tensor([0, 0, -1]).expand_as(directions)):
                    raise ValueError("Expected vertical downward rays")
            schema.append(term)
            configs.append(cfg)
        mapping = observation_mirror(schema, action_names, rays)
        if len(mapping.indices) != width:
            raise ValueError(f"Unexpected {group} width after resolving terms")
        offset = 0
        for term, cfg in zip(schema, configs):
            from .rough_v3_core import SignedPermutation
            end = offset + term["width"]
            piece = SignedPermutation(tuple(i - offset for i in mapping.indices[offset:end]),
                                      mapping.signs[offset:end])
            _require_symmetric_scale_clip(cfg, piece)
            offset = end
        maps[group] = mapping
    return maps


@torch.no_grad()
def compute_symmetric_states(env, obs=None, actions=None):
    raw = env.unwrapped
    maps = getattr(raw, "_zsl1_v5_symmetry_maps", None)
    if maps is None:
        maps = build_runtime_maps(env)
        raw._zsl1_v5_symmetry_maps = maps
    return augment_with_maps(obs, actions, maps)
