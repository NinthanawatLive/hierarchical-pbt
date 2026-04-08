"""
Go2EnvWithCamera — Subclass of Genesis Go2Env for M3 VLM Dispatch.

Design decisions (Apr 8, 2026):
  Q2: (a) subclass — clean inheritance, no upstream fork
  Q3: (a) disable episode timeout (episode_length_s = 99999)

Changes from base Go2Env:
  1. Injects a camera sensor before scene.build()
  2. Disables episode auto-reset (episode_length_s overridden to 99999)
  3. Exposes grab_frame() for slow loop to capture RGB images

IMPORTANT: This file must be verified against the actual Genesis Go2Env.__init__
on Colab. The override pattern depends on Genesis version (tested: 0.4.3).

Usage:
  from pbt_meta.sim.go2_env_camera import Go2EnvWithCamera
  env = Go2EnvWithCamera(num_envs=1, env_cfg=env_cfg, obs_cfg=obs_cfg,
                          reward_cfg=reward_cfg, command_cfg=command_cfg,
                          camera_cfg=camera_cfg)
"""

import numpy as np

try:
    import genesis as gs
except ImportError:
    gs = None


DEFAULT_CAMERA_CFG = {
    "name": "vlm_cam",
    "width": 320,
    "height": 240,
    "fov": 60,
    "pos_offset": (0.3, 0.0, 0.15),
    "lookat_offset": (2.0, 0.0, -0.2),
}


class Go2EnvWithCamera:
    """
    Wraps Go2Env to inject a camera before scene.build() and disable
    episode auto-reset for Phase 1 inference.

    Strategy: We reconstruct the Go2Env setup flow manually rather than
    calling super().__init__(), because Genesis Go2Env calls scene.build()
    inside __init__ — there's no hook point to inject camera between
    robot addition and build.
    """

    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg,
                 camera_cfg=None, show_viewer=False, device="cuda"):
        self.num_envs = num_envs
        self.device = device
        self.show_viewer = show_viewer
        env_cfg = dict(env_cfg)
        env_cfg["episode_length_s"] = 99999
        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg
        self.camera_cfg = camera_cfg or DEFAULT_CAMERA_CFG
        self.dt = 0.02
        self.max_episode_length = int(env_cfg["episode_length_s"] / self.dt)
        self._build_env_with_camera()

    def _build_env_with_camera(self):
        """Reconstruct Go2Env init flow with camera injection before build."""
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt),
            show_viewer=self.show_viewer,
            viewer_options=gs.options.ViewerOptions(max_FPS=60),
        )
        self.scene.add_entity(gs.morphs.Plane())
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(file="urdf/go2/urdf/go2.urdf", pos=(0, 0, 0.42)),
        )
        cam_cfg = self.camera_cfg
        self.camera = self.scene.add_camera(
            res=(cam_cfg["width"], cam_cfg["height"]),
            pos=cam_cfg["pos_offset"],
            lookat=cam_cfg["lookat_offset"],
            fov=cam_cfg["fov"],
            GUI=False,
        )
        self.scene.build(n_envs=self.num_envs)
        self.num_dof = self.robot.n_dofs
        self.num_actions = self.num_dof
        self.default_dof_pos = np.array(
            [0, 0, 0, 0, 0.8, 0.8, 1.0, 1.0, -1.5, -1.5, -1.5, -1.5],
            dtype=np.float32,
        )
        self.kp = self.env_cfg.get("kp", 20.0)
        self.kd = self.env_cfg.get("kd", 0.5)
        self.episode_step = 0
        self._camera_attached = True

    # ---- Camera interface for SlowLoop ----

    def grab_frame(self, env_idx=0):
        """Capture RGB frame (advances sim by 1 step for render)."""
        self._update_camera_pose(env_idx)
        self.camera.start_recording()
        self.scene.step()
        frame = self.camera.stop_recording()
        if hasattr(frame, 'shape') and len(frame.shape) == 4:
            frame = frame[-1]
        return frame

    def grab_frame_no_step(self, env_idx=0):
        """Capture RGB frame WITHOUT advancing simulation."""
        self._update_camera_pose(env_idx)
        try:
            self.camera.render()
            frame = self.camera.get_rgb()
            return np.array(frame)
        except (AttributeError, RuntimeError):
            return self.grab_frame(env_idx)

    def _update_camera_pose(self, env_idx=0):
        """Move camera to follow the robot."""
        cam_cfg = self.camera_cfg
        try:
            base_pos = self.robot.get_pos()
            if hasattr(base_pos, 'cpu'):
                base_pos = base_pos.cpu().numpy()
            if len(base_pos.shape) > 1:
                base_pos = base_pos[env_idx]
            cam_pos = (
                base_pos[0] + cam_cfg["pos_offset"][0],
                base_pos[1] + cam_cfg["pos_offset"][1],
                base_pos[2] + cam_cfg["pos_offset"][2],
            )
            lookat = (
                base_pos[0] + cam_cfg["lookat_offset"][0],
                base_pos[1] + cam_cfg["lookat_offset"][1],
                base_pos[2] + cam_cfg["lookat_offset"][2],
            )
            self.camera.set_pose(pos=cam_pos, lookat=lookat)
        except Exception:
            pass

    # ---- Passthrough interface for FastLoop ----

    @property
    def base_pos(self):
        return self.robot.get_pos()

    @property
    def base_lin_vel(self):
        return self.robot.get_vel()

    @property
    def base_ang_vel(self):
        return self.robot.get_ang()

    def get_obs(self):
        raise NotImplementedError(
            "Adapt get_obs() to match your Go2Env observation space."
        )

    def step_sim(self):
        self.scene.step()
        self.episode_step += 1

    def set_dof_velocity_target(self, actions):
        self.robot.control_dofs_position(
            self.default_dof_pos + actions, kp=self.kp, kd=self.kd,
        )
