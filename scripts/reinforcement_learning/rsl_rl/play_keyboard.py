# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0
# Modified variant of play.py: keyboard teleop reads from the SSH terminal
# (stdin/termios) instead of isaaclab.devices.Se2Keyboard, which cannot receive
# OS keyboard events on a headless cloud container (no DISPLAY, no /dev/input).

"""Play an RSL-RL checkpoint with terminal keyboard teleop.

Usage (interactive SSH session only):
  CUDA_VISIBLE_DEVICES=0 ENABLE_CAMERAS=0 LIVESTREAM=0 \
  /cpfs/user/tangyang/miniconda3/envs/isaaclab_2.3.2/bin/python \
    scripts/reinforcement_learning/rsl_rl/play_keyboard.py \
    --task RobotLab-Isaac-Velocity-Rough-V5-Zsibot-ZSL1-v0 \
    --checkpoint logs/rsl_rl/zsibot_zsl1_rough_v5/2026-10-06_09-57-58_deployment_limits_rough_32k/eval/snapshots/v5_model1600.pt \
    --real-time

Keys: UP/DOWN forward/backward, LEFT/RIGHT strafe, Z/C turn, SPACE clear, Q quit.
Terminal key-repeat refreshes a hold timeout; releasing a key zeroes that
direction after --key-timeout seconds. Speeds are fixed constants below and
stay inside the V5 training command ranges (vx [-0.6, 1.0], vy [-0.3, 0.3],
yaw [-0.8, 0.8]).
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play an RL agent with terminal keyboard teleop.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Ignored; keyboard mode always uses one environment.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--key-timeout", type=float, default=0.25,
    help="Seconds after the last key event before a held direction resets to zero.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import select
import termios
import time
import tty

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import robot_lab.tasks  # noqa: F401  # isort: skip

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Teleop speeds: same values as the audited V4/V5 deployment profile, all
# inside the V5 training command ranges.
FORWARD_SPEED = 0.4
BACKWARD_SPEED = 0.4
LATERAL_SPEED = 0.2
TURN_SPEED = 0.5


class StdinSe2:
    """SE(2) teleop from the controlling terminal, mirroring policy_deploy.py.

    A POSIX terminal has no key-release events, so key-repeat events refresh a
    short activity timeout per direction. Not a tty (e.g. a background run)
    degrades to a permanent zero command instead of failing.
    """

    _steps = {
        "UP": (FORWARD_SPEED, 0.0, 0.0),
        "DOWN": (-BACKWARD_SPEED, 0.0, 0.0),
        # +vy is left, matching the deployment script convention.
        "LEFT": (0.0, LATERAL_SPEED, 0.0),
        "RIGHT": (0.0, -LATERAL_SPEED, 0.0),
        "TURN_LEFT": (0.0, 0.0, TURN_SPEED),
        "TURN_RIGHT": (0.0, 0.0, -TURN_SPEED),
    }
    _escape_keys = {
        b"\x1b[A": "UP", b"\x1b[B": "DOWN", b"\x1b[C": "RIGHT", b"\x1b[D": "LEFT",
    }
    _byte_keys = {
        b"w": "UP", b"s": "DOWN", b"a": "LEFT", b"d": "RIGHT",
        b"8": "UP", b"2": "DOWN", b"4": "LEFT", b"6": "RIGHT",
        b"z": "TURN_LEFT", b"c": "TURN_RIGHT", b"7": "TURN_LEFT", b"9": "TURN_RIGHT",
    }

    def __init__(self, hold_timeout: float):
        self.hold_timeout = float(hold_timeout)
        self.command = [0.0, 0.0, 0.0]
        self.quit = False
        self._active = {}
        self._buffer = bytearray()
        self._old_settings = None

    def __enter__(self):
        if sys.stdin.isatty():
            self._old_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
            print("[keyboard] terminal teleop active")
        else:
            print("[keyboard] stdin is not a tty; command stays [0, 0, 0]")
        print("[keyboard] UP/DOWN forward/back, LEFT/RIGHT strafe, Z/C turn, SPACE clear, Q quit")
        return self

    def __exit__(self, *_):
        if self._old_settings is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_settings)

    def poll(self):
        if self._old_settings is None:
            return
        while select.select([sys.stdin], [], [], 0.0)[0]:
            chunk = os.read(sys.stdin.fileno(), 64)
            if not chunk:
                break
            self._buffer.extend(chunk)
        now = time.monotonic()
        while self._buffer:
            if self._buffer[0:1] == b"\x1b":
                if len(self._buffer) < 3:
                    break
                key = self._escape_keys.get(bytes(self._buffer[:3]))
                del self._buffer[:3]
                if key is not None:
                    self._active[key] = now
                continue
            key = bytes(self._buffer[:1]).lower()
            del self._buffer[:1]
            if key == b" ":
                self._active.clear()
                continue
            if key == b"q":
                self.quit = True
                continue
            mapped = self._byte_keys.get(key)
            if mapped is not None:
                self._active[mapped] = now
        for name in [n for n, t in self._active.items() if now - t > self.hold_timeout]:
            del self._active[name]
        self.command = [0.0, 0.0, 0.0]
        for name in self._active:
            for i, v in enumerate(self._steps[name]):
                self.command[i] += v

    def advance(self):
        self.poll()
        return list(self.command)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent and terminal keyboard teleop."""
    task_name = args_cli.task.split(":")[-1]

    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    print("[INFO] Keyboard teleop mode: forcing a single environment")
    env_cfg.scene.num_envs = 1

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    env_cfg.scene.terrain.max_init_terrain_level = None
    if env_cfg.scene.terrain.terrain_generator is not None:
        env_cfg.scene.terrain.terrain_generator.num_rows = 5
        env_cfg.scene.terrain.terrain_generator.num_cols = 5
        env_cfg.scene.terrain.terrain_generator.curriculum = False

    env_cfg.observations.policy.enable_corruption = False
    env_cfg.events.randomize_apply_external_force_torque = None
    env_cfg.events.push_robot = None
    env_cfg.curriculum.command_levels_lin_vel = None
    env_cfg.curriculum.command_levels_ang_vel = None

    env_cfg.terminations.time_out = None
    env_cfg.commands.base_velocity.debug_vis = False
    controller = StdinSe2(args_cli.key_timeout)
    env_cfg.observations.policy.velocity_commands = ObsTerm(
        func=lambda env: torch.tensor(controller.advance(), dtype=torch.float32).unsqueeze(0).to(env.device),
    )

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play_keyboard"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    policy = runner.get_inference_policy(device=env.unwrapped.device)

    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt
    robot = env.unwrapped.scene["robot"]

    obs = env.get_observations()
    last_print = 0.0
    with controller:
        while simulation_app.is_running():
            start_time = time.time()
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
                policy_nn.reset(dones)
            if controller.quit:
                print("[keyboard] Q pressed, exiting")
                break
            now = time.monotonic()
            if now - last_print > 1.0:
                last_print = now
                cmd = controller.command
                vel_b = robot.data.root_lin_vel_b[0].cpu().numpy()
                yaw_wz = robot.data.root_ang_vel_b[0, 2].item()
                gravity = robot.data.projected_gravity_b[0].cpu().numpy()
                print(
                    f"cmd [vx {cmd[0]:+.2f} vy {cmd[1]:+.2f} wz {cmd[2]:+.2f}] "
                    f"meas [vx {vel_b[0]:+.2f} vy {vel_b[1]:+.2f} wz {yaw_wz:+.2f}] "
                    f"upright_z {gravity[2]:+.2f} done {int(dones[0])}"
                )
            sleep_time = dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
