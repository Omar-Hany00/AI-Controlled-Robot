"""Interactive MuJoCo mobile manipulator with true hold-to-move controls.

Requirements:
    pip install "mujoco>=3.3" "glfw>=2.6"

Run:
    python mujoco_mobile_base.py

Controls:
    Arrow keys       Drive / turn the base while held
    1 / 2            Yaw the arm + / - while held
    3 / 4            Move the shoulder + / - while held
    5 / 6            Move the elbow + / - while held
    7 / 8            Open / close the gripper while held
    R                Restore the home pose and reset the cube
    H or F1          Toggle the in-window controls overlay
    Esc              Quit

Mouse:
    Left drag        Rotate camera vertically (hold Shift for horizontal)
    Right drag       Pan camera vertically (hold Shift for horizontal)
    Middle drag      Zoom
    Wheel            Zoom

Unlike ``mujoco.viewer.launch_passive`` key callbacks, this program owns the
GLFW window. GLFW exposes PRESS and RELEASE actions, so commands depend on the
actual current key state instead of operating-system key-repeat events.
"""

from __future__ import annotations

import logging
import sys
import time
import requests
from dataclasses import dataclass
from typing import Final

try:
    import glfw
    import mujoco
except ImportError as error:  # Give an actionable error rather than a traceback.
    package = error.name or "a required package"
    raise SystemExit(
        f"Missing {package!r}. Install the dependencies with:\n"
        '  pip install "mujoco>=3.3" "glfw>=2.6"'
    ) from error


LOG = logging.getLogger("mobile_manipulator")

API_URL = "http://127.0.0.1:8000/state"

last_command = "STOP"
last_poll = 0.0
POLL_INTERVAL = 0.05  # 20 Hz

# MuJoCo always uses SI units internally. Keeping these values in radians avoids
# any ambiguity between XML compiler angle settings and Python control values.
DEG: Final[float] = 3.141592653589793 / 180.0
SIM_STEP: Final[float] = 0.002
RENDER_INTERVAL: Final[float] = 1.0 / 60.0
WHEEL_SPEED: Final[float] = 13.0
TURN_SPEED: Final[float] = 8.0
ARM_RATE: Final[float] = 1.20
GRIPPER_RATE: Final[float] = 0.050
MAX_RENDER_GEOMS: Final[int] = 2_000


SCENE_XML = f"""
<mujoco model="mobile_manipulator">
  <compiler angle="radian" inertiafromgeom="true" autolimits="true"/>
  <option timestep="{SIM_STEP}" integrator="implicitfast" gravity="0 0 -9.81"
          iterations="80" ls_iterations="20"/>
  <size nconmax="400" njmax="300"/>

  <visual>
    <global azimuth="145" elevation="-25"/>
    <quality shadowsize="4096" offsamples="4"/>
    <map znear="0.02" zfar="30"/>
  </visual>

  <default>
    <joint damping="1" armature="0.01"/>
    <geom condim="4" friction="1.2 0.015 0.001" solref="0.008 1" solimp="0.95 0.99 0.001"/>
  </default>

  <asset>
    <texture name="floor_grid" type="2d" builtin="checker" width="512" height="512"
             rgb1="0.075 0.105 0.135" rgb2="0.12 0.16 0.20" mark="edge" markrgb="0.2 0.3 0.4"/>
    <material name="floor" texture="floor_grid" texrepeat="6 6" texuniform="true"/>
    <material name="chassis_blue" rgba="0.06 0.28 0.72 1" metallic="0.45" roughness="0.28"/>
    <material name="dark_rubber" rgba="0.035 0.04 0.05 1" roughness="0.8"/>
    <material name="hub_metal" rgba="0.52 0.57 0.62 1" metallic="0.9" roughness="0.2"/>
    <material name="arm_metal" rgba="0.28 0.32 0.38 1" metallic="0.85" roughness="0.25"/>
    <material name="gripper_red" rgba="0.78 0.08 0.10 1" metallic="0.4" roughness="0.32"/>
    <material name="crate_orange" rgba="0.92 0.40 0.05 1" metallic="0.15" roughness="0.45"/>
    <material name="goal_green" rgba="0.05 0.86 0.35 0.34"/>
  </asset>

  <worldbody>
    <light name="key" directional="true" pos="0 0 5" dir="-0.35 -0.25 -1"
           diffuse="0.95 0.95 0.95" specular="0.2 0.2 0.2"/>
    <light name="fill" pos="-2 -2 3" diffuse="0.28 0.34 0.45"/>
    <geom name="floor" type="plane" size="12 12 0.1" material="floor"/>

    <!-- Base root is 0.16m high: 0.10m radius wheels then touch z=0 exactly. -->
    <body name="base_link" pos="0 0 0.16">
      <freejoint name="base_freejoint"/>
      <inertial pos="0 0 0" mass="17" diaginertia="0.34 0.62 0.78"/>
      <geom name="chassis" type="box" size="0.34 0.22 0.06" material="chassis_blue"/>
      <geom name="top_plate" type="box" pos="0 0 0.067" size="0.25 0.17 0.012" material="hub_metal"/>
      <geom name="front_bumper" type="capsule" pos="0.37 0 0" size="0.026 0.20" quat="0.707107 0 0.707107 0" material="dark_rubber"/>

      <body name="front_left_wheel" pos="0.23 0.245 -0.06">
        <joint name="wheel_fl" type="hinge" axis="0 1 0" damping="0.12" armature="0.025"/>
        <geom name="wheel_fl_tire" type="cylinder" size="0.10 0.038" quat="0.707107 0.707107 0 0" material="dark_rubber"/>
        <geom name="wheel_fl_hub" type="cylinder" size="0.048 0.040" quat="0.707107 0.707107 0 0" material="hub_metal" contype="0" conaffinity="0"/>
      </body>
      <body name="front_right_wheel" pos="0.23 -0.245 -0.06">
        <joint name="wheel_fr" type="hinge" axis="0 1 0" damping="0.12" armature="0.025"/>
        <geom name="wheel_fr_tire" type="cylinder" size="0.10 0.038" quat="0.707107 0.707107 0 0" material="dark_rubber"/>
        <geom name="wheel_fr_hub" type="cylinder" size="0.048 0.040" quat="0.707107 0.707107 0 0" material="hub_metal" contype="0" conaffinity="0"/>
      </body>
      <body name="rear_left_wheel" pos="-0.23 0.245 -0.06">
        <joint name="wheel_rl" type="hinge" axis="0 1 0" damping="0.12" armature="0.025"/>
        <geom name="wheel_rl_tire" type="cylinder" size="0.10 0.038" quat="0.707107 0.707107 0 0" material="dark_rubber"/>
        <geom name="wheel_rl_hub" type="cylinder" size="0.048 0.040" quat="0.707107 0.707107 0 0" material="hub_metal" contype="0" conaffinity="0"/>
      </body>
      <body name="rear_right_wheel" pos="-0.23 -0.245 -0.06">
        <joint name="wheel_rr" type="hinge" axis="0 1 0" damping="0.12" armature="0.025"/>
        <geom name="wheel_rr_tire" type="cylinder" size="0.10 0.038" quat="0.707107 0.707107 0 0" material="dark_rubber"/>
        <geom name="wheel_rr_hub" type="cylinder" size="0.048 0.040" quat="0.707107 0.707107 0 0" material="hub_metal" contype="0" conaffinity="0"/>
      </body>

      <body name="arm_yaw_link" pos="0.12 0 0.082">
        <joint name="arm_yaw" type="hinge" axis="0 0 1" range="{-150 * DEG:.8f} {150 * DEG:.8f}" damping="2.0" armature="0.035"/>
        <geom name="yaw_pedestal" type="cylinder" size="0.055 0.045" material="hub_metal"/>
        <geom name="yaw_collars" type="cylinder" pos="0 0 0.048" size="0.062 0.008" material="arm_metal"/>

        <body name="shoulder_link" pos="0 0 0.065">
          <joint name="arm_shoulder" type="hinge" axis="0 1 0" range="{-92 * DEG:.8f} {92 * DEG:.8f}" damping="2.8" armature="0.045"/>
          <geom name="shoulder_motor" type="cylinder" quat="0.707107 0.707107 0 0" size="0.050 0.035" material="hub_metal"/>
          <geom name="upper_arm" type="capsule" pos="0 0 0.145" size="0.032 0.135" material="arm_metal"/>

          <body name="elbow_link" pos="0 0 0.29">
            <joint name="arm_elbow" type="hinge" axis="0 1 0" range="{-125 * DEG:.8f} {125 * DEG:.8f}" damping="2.2" armature="0.035"/>
            <geom name="elbow_motor" type="cylinder" quat="0.707107 0.707107 0 0" size="0.043 0.031" material="hub_metal"/>
            <geom name="forearm" type="capsule" pos="0 0 0.125" size="0.026 0.112" material="arm_metal"/>

            <body name="wrist" pos="0 0 0.25">
              <geom name="palm" type="box" size="0.032 0.075 0.020" material="hub_metal"/>
              <geom name="wrist_pad" type="box" pos="0 0 0.030" size="0.036 0.042 0.012" material="arm_metal"/>
              <body name="left_finger" pos="0 0.058 0.058">
                <joint name="finger_left" type="slide" axis="0 -1 0" range="0 0.045" damping="4.5" armature="0.004"/>
                <geom name="left_finger_geom" type="box" size="0.014 0.012 0.052" material="gripper_red" friction="1.4 0.02 0.001"/>
              </body>
              <body name="right_finger" pos="0 -0.058 0.058">
                <joint name="finger_right" type="slide" axis="0 1 0" range="0 0.045" damping="4.5" armature="0.004"/>
                <geom name="right_finger_geom" type="box" size="0.014 0.012 0.052" material="gripper_red" friction="1.4 0.02 0.001"/>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>

    <!-- A freely moving object makes the gripper useful rather than decorative. -->
    <body name="cargo_cube" pos="0.56 0 0.075">
      <freejoint name="cargo_freejoint"/>
      <geom name="cargo" type="box" size="0.04 0.04 0.04" mass="0.45" material="crate_orange" friction="1.25 0.02 0.001"/>
    </body>
    <geom name="goal" type="cylinder" pos="-0.60 0 0.002" size="0.15 0.003" material="goal_green" contype="0" conaffinity="0"/>
  </worldbody>

  <actuator>
    <!-- Velocity targets are actively braked to zero as soon as arrow keys release. -->
    <velocity name="drive_fl" joint="wheel_fl" kv="24" ctrlrange="-25 25" forcerange="-32 32"/>
    <velocity name="drive_fr" joint="wheel_fr" kv="24" ctrlrange="-25 25" forcerange="-32 32"/>
    <velocity name="drive_rl" joint="wheel_rl" kv="24" ctrlrange="-25 25" forcerange="-32 32"/>
    <velocity name="drive_rr" joint="wheel_rr" kv="24" ctrlrange="-25 25" forcerange="-32 32"/>
    <position name="servo_yaw" joint="arm_yaw" kp="180" kv="18" ctrlrange="{-150 * DEG:.8f} {150 * DEG:.8f}" forcerange="-80 80"/>
    <position name="servo_shoulder" joint="arm_shoulder" kp="230" kv="24" ctrlrange="{-92 * DEG:.8f} {92 * DEG:.8f}" forcerange="-110 110"/>
    <position name="servo_elbow" joint="arm_elbow" kp="180" kv="20" ctrlrange="{-125 * DEG:.8f} {125 * DEG:.8f}" forcerange="-85 85"/>
    <position name="servo_finger_left" joint="finger_left" kp="110" kv="12" ctrlrange="0 0.045" forcerange="-30 30"/>
    <position name="servo_finger_right" joint="finger_right" kp="110" kv="12" ctrlrange="0 0.045" forcerange="-30 30"/>
  </actuator>
</mujoco>
"""


@dataclass
class ArmTargets:
    """Persistent position-servo targets, all stored in SI units."""

    yaw: float = 0.0
    shoulder: float = 0.0
    elbow: float = -0.45
    aperture: float = 0.0


class KeyState:
    """Tracks keys by GLFW code and clears safely on focus loss."""

    def __init__(self) -> None:
        self._pressed: set[int] = set()

    def set(self, key: int, is_pressed: bool) -> None:
        if is_pressed:
            self._pressed.add(key)
        else:
            self._pressed.discard(key)

    def down(self, key: int) -> bool:
        return key in self._pressed

    def axis(self, positive: int, negative: int) -> int:
        """Return -1, 0 or 1; opposing keys cancel deterministically."""
        return int(self.down(positive)) - int(self.down(negative))

    def clear(self) -> None:
        self._pressed.clear()


class MobileManipulator:
    """Couples the model, deterministic controller, renderer and GLFW events."""

    WHEEL_ACTUATORS: Final[tuple[str, ...]] = (
        "drive_fl",
        "drive_fr",
        "drive_rl",
        "drive_rr",
    )
    POSITION_ACTUATORS: Final[tuple[str, ...]] = (
        "servo_yaw",
        "servo_shoulder",
        "servo_elbow",
        "servo_finger_left",
        "servo_finger_right",
    )

    def __init__(self) -> None:
        self.model = mujoco.MjModel.from_xml_string(SCENE_XML)
        self.data = mujoco.MjData(self.model)
        self.keys = KeyState()
        self.targets = ArmTargets()
        self.show_help = True
        self.window: glfw._GLFWwindow | None = None
        self._mouse_left = False
        self._mouse_right = False
        self._mouse_middle = False
        self._last_cursor: tuple[float, float] | None = None
        self.current_command = "STOP"

        self.actuator_id = {
            name: self._named_id(mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in (*self.WHEEL_ACTUATORS, *self.POSITION_ACTUATORS)
        }
        self.joint_qpos = {
            name: int(self.model.jnt_qposadr[self._named_id(mujoco.mjtObj.mjOBJ_JOINT, name)])
            for name in ("arm_yaw", "arm_shoulder", "arm_elbow", "finger_left", "finger_right")
        }

        self.camera = mujoco.MjvCamera()
        self.option = mujoco.MjvOption()
        self.perturb = mujoco.MjvPerturb()
        self.scene = mujoco.MjvScene(self.model, maxgeom=MAX_RENDER_GEOMS)
        self.context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)
        mujoco.mjv_defaultCamera(self.camera)
        mujoco.mjv_defaultOption(self.option)
        mujoco.mjv_defaultPerturb(self.perturb)
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.azimuth = 145.0
        self.camera.elevation = -25.0
        self.camera.distance = 2.6
        self.camera.lookat[:] = (0.0, 0.0, 0.32)

        self.reset()

    def _named_id(self, object_type: mujoco.mjtObj, name: str) -> int:
        object_id = mujoco.mj_name2id(self.model, object_type, name)
        if object_id < 0:
            raise RuntimeError(f"Model is missing required {object_type.name}: {name}")
        return object_id

    @staticmethod
    def _clip(value: float, lower: float, upper: float) -> float:
        return max(lower, min(upper, value))

    def reset(self) -> None:
        """Restore a contact-safe home pose and clear all stale input state."""
        mujoco.mj_resetData(self.model, self.data)
        self.keys.clear()
        self.targets = ArmTargets()

        # The default XML pose is collision-free; only the intended arm pose is
        # applied after reset. `mj_forward` refreshes positions before rendering.
        self.data.qpos[self.joint_qpos["arm_yaw"]] = self.targets.yaw
        self.data.qpos[self.joint_qpos["arm_shoulder"]] = self.targets.shoulder
        self.data.qpos[self.joint_qpos["arm_elbow"]] = self.targets.elbow
        self.data.qpos[self.joint_qpos["finger_left"]] = self.targets.aperture
        self.data.qpos[self.joint_qpos["finger_right"]] = self.targets.aperture
        self._apply_targets()
        mujoco.mj_forward(self.model, self.data)
        LOG.info("Simulation reset to the home pose.")

    def _apply_targets(self) -> None:
        """Write every actuator every step, including explicit zero wheel speed."""
        # Keyboard input
        forward = self.keys.axis(glfw.KEY_UP, glfw.KEY_DOWN)
        turn = self.keys.axis(glfw.KEY_LEFT, glfw.KEY_RIGHT)

        # Remote input
        self.poll_command()
        command = self.current_command

        if command == "FORWARD":
            forward = 1
        elif command == "BACKWARD":
            forward = -1
        elif command == "LEFT":
            turn = -1
        elif command == "RIGHT":
            turn = 1

        left = self._clip(forward * WHEEL_SPEED - turn * TURN_SPEED, -25.0, 25.0)
        right = self._clip(forward * WHEEL_SPEED + turn * TURN_SPEED, -25.0, 25.0)
        self.data.ctrl[self.actuator_id["drive_fl"]] = left
        self.data.ctrl[self.actuator_id["drive_rl"]] = left
        self.data.ctrl[self.actuator_id["drive_fr"]] = right
        self.data.ctrl[self.actuator_id["drive_rr"]] = right

        self.data.ctrl[self.actuator_id["servo_yaw"]] = self.targets.yaw
        self.data.ctrl[self.actuator_id["servo_shoulder"]] = self.targets.shoulder
        self.data.ctrl[self.actuator_id["servo_elbow"]] = self.targets.elbow
        self.data.ctrl[self.actuator_id["servo_finger_left"]] = self.targets.aperture
        self.data.ctrl[self.actuator_id["servo_finger_right"]] = self.targets.aperture

    def update_controller(self, dt: float) -> None:
        """Advance targets continuously from keyboard and remote commands."""

        # ---------------- Keyboard ----------------
        yaw = self.keys.axis(glfw.KEY_1, glfw.KEY_2)
        shoulder = self.keys.axis(glfw.KEY_3, glfw.KEY_4)
        elbow = self.keys.axis(glfw.KEY_5, glfw.KEY_6)
        gripper = self.keys.axis(glfw.KEY_7, glfw.KEY_8)

        # ---------------- Remote ----------------
        command = self.poll_command()

        if command == "YAW_LEFT":
            yaw = 1
        elif command == "YAW_RIGHT":
            yaw = -1

        elif command == "SHOULDER_FORWARD":
            shoulder = 1
        elif command == "SHOULDER_BACKWARD":
            shoulder = -1

        elif command == "ELBOW_UP":
            elbow = 1
        elif command == "ELBOW_DOWN":
            elbow = -1

        elif command == "GRIPPER_OPEN":
            gripper = 1
        elif command == "GRIPPER_CLOSE":
            gripper = -1

        self.targets.yaw = self._clip(
            self.targets.yaw + yaw * ARM_RATE * dt,
            -150.0 * DEG,
            150.0 * DEG,
        )

        self.targets.shoulder = self._clip(
            self.targets.shoulder + shoulder * ARM_RATE * dt,
            -92.0 * DEG,
            92.0 * DEG,
        )

        self.targets.elbow = self._clip(
            self.targets.elbow + elbow * ARM_RATE * dt,
            -125.0 * DEG,
            125.0 * DEG,
        )

        self.targets.aperture = self._clip(
            self.targets.aperture + gripper * GRIPPER_RATE * dt,
            0.0,
            0.045,
        )

        self._apply_targets()

    def install_callbacks(self, window: glfw._GLFWwindow) -> None:
        """Install press/release and camera callbacks on the owned GLFW window."""
        self.window = window
        glfw.set_key_callback(window, self._on_key)
        glfw.set_window_focus_callback(window, self._on_focus)
        glfw.set_mouse_button_callback(window, self._on_mouse_button)
        glfw.set_cursor_pos_callback(window, self._on_cursor)
        glfw.set_scroll_callback(window, self._on_scroll)

    def _on_key(
        self,
        window: glfw._GLFWwindow,
        key: int,
        _scancode: int,
        action: int,
        _mods: int,
    ) -> None:
        if action in (glfw.PRESS, glfw.RELEASE):
            self.keys.set(key, action == glfw.PRESS)

        # These commands are intentionally edge-triggered: GLFW repeat events
        # must never repeatedly reset the model or flicker the help panel.
        if action != glfw.PRESS:
            return
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)
        elif key == glfw.KEY_R:
            self.reset()
        elif key in (glfw.KEY_H, glfw.KEY_F1):
            self.show_help = not self.show_help

    def _on_focus(self, _window: glfw._GLFWwindow, focused: int) -> None:
        # Release events can be swallowed when a user alt-tabs; this guarantees
        # the robot never keeps driving after the window loses focus.
        if not focused:
            self.keys.clear()

    def _on_mouse_button(
        self,
        window: glfw._GLFWwindow,
        button: int,
        action: int,
        _mods: int,
    ) -> None:
        is_down = action == glfw.PRESS
        if button == glfw.MOUSE_BUTTON_LEFT:
            self._mouse_left = is_down
        elif button == glfw.MOUSE_BUTTON_RIGHT:
            self._mouse_right = is_down
        elif button == glfw.MOUSE_BUTTON_MIDDLE:
            self._mouse_middle = is_down

        if is_down:
            self._last_cursor = glfw.get_cursor_pos(window)
        elif not (self._mouse_left or self._mouse_right or self._mouse_middle):
            self._last_cursor = None

    def _on_cursor(self, window: glfw._GLFWwindow, x: float, y: float) -> None:
        if self._last_cursor is None:
            self._last_cursor = (x, y)
            return
        if not (self._mouse_left or self._mouse_right or self._mouse_middle):
            self._last_cursor = (x, y)
            return

        previous_x, previous_y = self._last_cursor
        self._last_cursor = (x, y)
        _width, height = glfw.get_window_size(window)
        if height <= 0:
            return
        dx = (x - previous_x) / height
        dy = (y - previous_y) / height
        shift = glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS or glfw.get_key(
            window, glfw.KEY_RIGHT_SHIFT
        ) == glfw.PRESS

        if self._mouse_left:
            action = mujoco.mjtMouse.mjMOUSE_ROTATE_H if shift else mujoco.mjtMouse.mjMOUSE_ROTATE_V
        elif self._mouse_right:
            action = mujoco.mjtMouse.mjMOUSE_MOVE_H if shift else mujoco.mjtMouse.mjMOUSE_MOVE_V
        else:
            action = mujoco.mjtMouse.mjMOUSE_ZOOM
        mujoco.mjv_moveCamera(self.model, action, dx, dy, self.scene, self.camera)

    def _on_scroll(self, _window: glfw._GLFWwindow, _xoffset: float, yoffset: float) -> None:
        mujoco.mjv_moveCamera(
            self.model,
            mujoco.mjtMouse.mjMOUSE_ZOOM,
            0.0,
            -0.05 * yoffset,
            self.scene,
            self.camera,
        )

    def step_for_frame(self) -> None:
        """Advance fixed-step physics for one 60 Hz render slice."""
        frame_start = self.data.time
        while self.data.time - frame_start < RENDER_INTERVAL:
            self.update_controller(self.model.opt.timestep)
            mujoco.mj_step(self.model, self.data)

    def render(self) -> None:
        if self.window is None:
            return
        width, height = glfw.get_framebuffer_size(self.window)
        if width <= 0 or height <= 0:
            return
        viewport = mujoco.MjrRect(0, 0, width, height)
        mujoco.mjv_updateScene(
            self.model,
            self.data,
            self.option,
            self.perturb,
            self.camera,
            mujoco.mjtCatBit.mjCAT_ALL,
            self.scene,
        )
        mujoco.mjr_render(viewport, self.scene, self.context)
        if self.show_help:
            pass

    def _render_overlay(self, viewport: mujoco.MjrRect) -> None:
        left = (
            "MOBILE MANIPULATOR\n"
            "Arrow keys   Drive / turn (hold)\n"
            "1/2          Arm yaw + / -\n"
            "3/4          Shoulder + / -\n"
            "5/6          Elbow + / -\n"
            "7/8          Gripper open / close\n"
            "R            Reset scene\n"
            "H or F1      Hide this help\n"
            "Esc          Quit"
        )
        right = (
            f"sim time: {self.data.time:6.2f}s\n"
            f"yaw:       {self.targets.yaw / DEG:6.1f} deg\n"
            f"shoulder:  {self.targets.shoulder / DEG:6.1f} deg\n"
            f"elbow:     {self.targets.elbow / DEG:6.1f} deg\n"
            f"gripper:   {self.targets.aperture * 1000:6.1f} mm\n"
            "\n"
            "Mouse wheel: zoom\n"
            "Mouse drag: camera"
        )
        mujoco.mjr_overlay(
            mujoco.mjtFontScale.mjFONTSCALE_150,
            mujoco.mjtGridPos.mjGRID_TOPLEFT,
            viewport,
            left,
            right,
            self.context,
        )

    def close(self) -> None:
        """Release MuJoCo GPU allocations before GLFW destroys its context."""
        self.context.free()
        self.scene.free()

    def poll_command(self):
        
        """Read the latest command from FastAPI."""

        global last_command, last_poll

        now = time.time()

        if now - last_poll < POLL_INTERVAL:
            return last_command

        last_poll = now

        try:
            response = requests.get(API_URL, timeout=0.05)

            if response.ok:
                data = response.json()
                last_command = data.get("command", "STOP")

        except requests.RequestException:
            pass

        self.current_command = last_command
        return self.current_command

def create_window() -> glfw._GLFWwindow:
    """Create a modern OpenGL window with useful failure messages."""
    if not glfw.init():
        raise RuntimeError("GLFW could not initialize. Check your graphics driver/display session.")
    glfw.window_hint(glfw.VISIBLE, glfw.TRUE)
    glfw.window_hint(glfw.SAMPLES, 4)
    window = glfw.create_window(1440, 900, "MuJoCo Mobile Manipulator", None, None)
    if window is None:
        glfw.terminate()
        raise RuntimeError("GLFW could not create an OpenGL window.")
    glfw.make_context_current(window)
    glfw.swap_interval(1)  # V-sync: avoids wasting a CPU core while interactive.
    return window


def RobotSimulation() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    window: glfw._GLFWwindow | None = None
    simulation: MobileManipulator | None = None
    try:
        window = create_window()
        simulation = MobileManipulator()
        simulation.install_callbacks(window)
        LOG.info("Ready. Focus the simulation window and use Arrow keys / 1-8.")

        while not glfw.window_should_close(window):
            simulation.step_for_frame()
            simulation.render()
            glfw.swap_buffers(window)
            glfw.poll_events()
        return 0
    except (mujoco.FatalError, RuntimeError) as error:
        LOG.error("%s", error)
        return 1
    finally:
        if simulation is not None:
            simulation.close()
        if window is not None:
            glfw.destroy_window(window)
        glfw.terminate()


if __name__ == "__main__":
    sys.exit(RobotSimulation())
