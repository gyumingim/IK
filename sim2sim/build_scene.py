"""Build the MuJoCo scene of Jungwoo's (JaxIsaacLab) V3 deploy layout from his robot_v3 URDF.

    venv/bin/python sim2sim/build_scene.py        # writes sim2sim/jax_v3_scene.xml (25550) and jax_v3_scene_23025.xml

Two specs (SPECS below; chain.py picks the scene by its spec):
- "25550", the 2026-11-11 final spec (deploy/README.md, deploy/MUJOCO_SIM2SIM.md section 0): URDF_1109 task URDF,
  `base_link` at z -0.005 so the shoulder pitch axis is at 1.2075 (47.25 cm above the table top), servo-emulated arm
  (a torque-limited servo constraint per arm joint that follows the overdamped response of Isaac's kp 500 kd 40
  emulation and stands on the commanded pose without sag, see SPECS), motor torque limits (0-3: 8, 4: 3.92,
  5: 1.96, 6: 0.92, RotationBody and the six finger joints 0.177 N m; fingers: PD kp 40 kd 2 with that force
  range), and friction per contact pair = PhysX's average of the two materials (finger-cube 2.0,
  cube-table 1.6, finger-table 1.6, other robot link-cube 1.2, other robot link-table 0.8) as explicit <pair>s
- "23025", the 11-08 spec this folder started with (deploy/README_1108.md, MUJOCO_SIM2SIM_1108.md), described in
  the rest of this text: URDF_1108 task URDF, `base_link` at z 0, sagging PD arm, no torque limit, friction 0.4

Everything follows `deploy/MUJOCO_SIM2SIM.md` section 4 and `deploy/README.md` (world frame with the floor at
z = 0, the table toward world -x):
- robot `base_link` (the stand) at (0, 0.01, 0), turned -90 deg about z (quat wxyz 0.7071, 0, 0, -0.7071);
  joint axes, limits, masses, inertias and collision meshes exactly as in `deploy/robot_v3` (V2 sign convention:
  `shoulder_roll_to_shoulder_yaw` and `under_arm_to_wrist_pitch` already flipped there, so no sign changes here)
- base_link, Shoulder_Pitch and Holder are fixed (Isaac merges them); base_link has no collision (removed in the URDF)
- robot self-collision off (`self_collision=False` in Jungwoo's robot config); the robot touches the cube, table, floor
- drives: arm 8 joints kp 100 kd 8, fingers kp 40 kd 2 armature 2e-4, torque limit 100 N m, no joint friction
- table 0.80 x 1.2 x 0.04 centred (-0.625, 0, 0.715) (top 0.735), cube 50 mm 40 g at (-0.285, 0.26, 0.760)
- PhysX runs 120 Hz (the chain's controller 60 Hz, two PhysX steps each). MuJoCo needs three engine settings to
  behave like it (sim2sim/README.md, compatibility items 2-3); each was measured against plain MuJoCo:
  MUJOCO_DT 1/1200 s (ten MuJoCo steps per PhysX step) so the contact time constant can be CONTACT_TC 0.002 s
  (default 0.02 s lets the kp 40 fingers sink 6-24 mm into the 40 g cube and fling it), and an elliptic friction
  cone with IMPRATIO 10 (MuJoCo's soft friction lets the held cube creep and tilt 34 deg; PhysX friction does not).
  Friction coefficients, masses and drive gains are not changed by any of these
- friction: PhysX averages the two materials, MuJoCo takes the larger. Robot 0.4, cube 0.4, table 0.8 make every
  pair equal to the PhysX average of the static values (cube-robot 0.4, cube-table 0.8, robot-table 0.8)

Visual meshes: the robot_v3 DAE files are byte-identical to URDF_1108's, already converted to OBJ in
motion2/arm/meshes, so the scene points there. This file is generated: change this script, not the XML.
"""

import os
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation as Rot

HERE = os.path.dirname(os.path.abspath(__file__))
URDF_DIR = os.path.join(HERE, "deploy", "robot_v3", "urdf")
VISUAL_DIR = os.path.join("..", "motion2", "arm", "meshes")          # relative to OUT
COLLISION_DIR = os.path.join("deploy", "robot_v3", "meshes", "JAXV3_Assembly")

BASE_QUAT = (0.7071067811865476, 0.0, 0.0, -0.7071067811865476)
FIXED_LINKS = ("base_link", "Shoulder_Pitch", "Holder")

ARM_JOINTS = ("shoulder_pitch_to_shoulder_roll", "shoulder_roll_to_shoulder_yaw", "shoulder_yaw_to_upper_arm",
              "upper_arm_to_under_arm", "under_arm_to_wrist_pitch", "wrist_pitch_to_wrist_yaw",
              "wrist_yaw_to_hand_base", "hand_base_to_rotation_base")
FINGER_JOINTS = ("hand_base_to_finger_1_1", "hand_base_to_finger_2_1", "rotation_base_to_finger_3_1",
                 "finger_1_1_to_finger_1_2", "finger_1_2_to_finger_2_2", "finger_3_1_to_finger_3_2")
FINGER_KP, FINGER_KD, FINGER_ARMATURE = 40.0, 2.0, 2e-4
FINGER_LINKS = ("Finger_1_1", "Finger_1_2", "Finger_2_1", "Finger_2_2", "Finger_3_1", "Finger_3_2")

SPECS = {
    "25550": {
        "urdf": "JAXV3_Assembly.urdf", "out": "jax_v3_scene.xml", "base_pos": (0.0, 0.01, -0.005),
        "arm_kp": 500.0, "arm_kd": 40.0,                       # servo emulation, README 5.3 (kp 100 kd 8 x 5)
        "arm_torque": (8.0, 8.0, 8.0, 8.0, 3.92, 1.96, 0.92, 0.177), "finger_torque": 0.177,   # ARM_JOINTS order
        # geom values only matter for pairs without a <pair> (cube-floor); the pairs below carry the spec
        # Arm servo = torque-limited constraint, not a PD actuator (module docstring): MuJoCo applies a clamped
        # actuator force explicitly, and kd 40 with a 0.177 - 0.92 N m limit on 1e-4 kg m2 joints then chatters and
        # holds nothing
        "arm_servo_constraint": True, "servo_solref": (0.002, 1.0),
        "robot_friction": 0.4, "cube_friction": 2.0, "table_friction": 1.2,
        "pair_friction": {"finger_cube": 2.0, "robot_cube": 1.2, "finger_table": 1.6, "robot_table": 0.8, "cube_table": 1.6},
    },
    "23025": {
        "urdf": "JAXV3_Assembly_1108task.urdf", "out": "jax_v3_scene_23025.xml", "base_pos": (0.0, 0.01, 0.0),
        "arm_kp": 100.0, "arm_kd": 8.0, "arm_torque": (100.0,) * 8, "finger_torque": 100.0, "arm_servo_constraint": False,
        "robot_friction": 0.4, "cube_friction": 0.4, "table_friction": 0.8, "pair_friction": None,
    },
}
DEFAULT_SPEC = "25550"
TABLE_SIZE, TABLE_CENTER = (0.80, 1.2, 0.04), (-0.625, 0.0, 0.715)
CUBE_SIDE, CUBE_MASS, CUBE_REF = 0.050, 0.040, (-0.285, 0.26, 0.760)
MUJOCO_DT = 1.0 / 1200.0
CONTACT_TC = 0.002
CONE, IMPRATIO = "elliptic", 10
COLORS = {"Steel": "0.63 0.63 0.63 1", "Acetal_Resin_White": "0.9 0.9 0.88 1", "Generic": "0.25 0.25 0.27 1"}


def _look(pos, target):
    """Camera xyaxes looking from pos at target with world z up."""
    pos, target = np.asarray(pos, float), np.asarray(target, float)
    fwd = (target - pos) / np.linalg.norm(target - pos)
    x = np.cross(fwd, (0.0, 0.0, 1.0))
    x /= np.linalg.norm(x)
    y = np.cross(x, fwd)
    return f'pos="{_f(pos)}" xyaxes="{_f(x)} {_f(y)}"'


def _f(values):
    return " ".join(f"{float(v):.9g}" for v in values)


def _origin(elem):
    o = elem.find("origin") if elem is not None else None
    xyz = [float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()]
    rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
    return xyz, rpy


def _quat(rpy):
    x, y, z, w = Rot.from_euler("xyz", rpy).as_quat()        # URDF rpy: fixed axes x, y, z
    return (w, x, y, z)


def build(spec=DEFAULT_SPEC):
    sp = SPECS[spec]
    URDF, OUT, BASE_POS = os.path.join(URDF_DIR, sp["urdf"]), os.path.join(HERE, sp["out"]), sp["base_pos"]
    ROBOT_FRICTION, CUBE_FRICTION, TABLE_FRICTION = sp["robot_friction"], sp["cube_friction"], sp["table_friction"]
    root = ET.parse(URDF).getroot()
    links = {L.get("name"): L for L in root.findall("link")}
    joints = root.findall("joint")
    children = {}
    for j in joints:
        children.setdefault(j.find("parent").get("link"), []).append(j)

    assets, meshes, col_geoms = [], set(), []

    def mesh_asset(kind, filename, scale):
        base = os.path.basename(filename)
        if kind == "visual":
            name, path = "v_" + base[:-4], os.path.join(VISUAL_DIR, base.replace(".dae", ".obj"))
        else:
            name, path = "c_" + base[:-4], os.path.join(COLLISION_DIR, base)
        if name not in meshes:
            meshes.add(name)
            assets.append(f'    <mesh name="{name}" file="{path}" scale="{scale}"/>')
        return name

    def geoms(link, ind):
        out = []
        L = links[link]
        for v in L.findall("visual"):
            xyz, rpy = _origin(v)
            m = v.find("geometry/mesh")
            name = mesh_asset("visual", m.get("filename"), m.get("scale", "1 1 1"))
            mat = v.find("material")
            rgba = COLORS.get(mat.get("name") if mat is not None else "", "0.5 0.5 0.5 1")
            out.append(f'{ind}<geom type="mesh" mesh="{name}" pos="{_f(xyz)}" quat="{_f(_quat(rpy))}" '
                       f'contype="0" conaffinity="0" group="1" density="0" rgba="{rgba}"/>')
        for c in L.findall("collision"):
            xyz, rpy = _origin(c)
            m = c.find("geometry/mesh")
            assert m is not None, f"{link}: only mesh collisions expected"
            name = mesh_asset("collision", m.get("filename"), m.get("scale", "1 1 1"))
            col_geoms.append(link)
            out.append(f'{ind}<geom name="{link}_col" type="mesh" mesh="{name}" pos="{_f(xyz)}" quat="{_f(_quat(rpy))}" '
                       f'contype="2" conaffinity="1" group="3" density="0" friction="{ROBOT_FRICTION} 0.005 0.0001" '
                       f'rgba="0.8 0.3 0.3 0.4"/>')
        return out

    def inertial(link, ind):
        i = links[link].find("inertial")
        xyz, rpy = _origin(i)
        assert np.allclose(rpy, 0.0), f"{link}: rotated inertial frame"
        m = float(i.find("mass").get("value"))
        I = i.find("inertia")
        full = [I.get(k) for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")]
        return f'{ind}<inertial pos="{_f(xyz)}" mass="{m:.9g}" fullinertia="{_f(full)}"/>'

    def body(link, joint, ind):
        lines = []
        if joint is None:
            lines.append(f'{ind}<body name="{link}" pos="{_f(BASE_POS)}" quat="{_f(BASE_QUAT)}">')
        else:
            xyz, rpy = _origin(joint)
            lines.append(f'{ind}<body name="{link}" pos="{_f(xyz)}" quat="{_f(_quat(rpy))}">')
        lines.append(inertial(link, ind + "  "))
        if joint is not None and joint.get("type") == "revolute":
            name = joint.get("name")
            axis = joint.find("axis").get("xyz")
            lim = joint.find("limit")
            arm = name in ARM_JOINTS
            armature = 0.0 if arm else FINGER_ARMATURE
            lines.append(f'{ind}  <joint name="{name}" type="hinge" axis="{axis}" limited="true" '
                         f'range="{lim.get("lower")} {lim.get("upper")}" armature="{armature:g}" damping="0" '
                         f'frictionloss="0"/>')
        else:
            assert joint is None or joint.get("type") == "fixed"
        lines += geoms(link, ind + "  ")
        for j in children.get(link, []):
            lines += body(j.find("child").get("link"), j, ind + "  ")
        lines.append(f"{ind}</body>")
        return lines

    robot = body("base_link", None, "    ")
    acts = []
    for name in ARM_JOINTS + FINGER_JOINTS:
        kp, kd = (sp["arm_kp"], sp["arm_kd"]) if name in ARM_JOINTS else (FINGER_KP, FINGER_KD)
        lim = sp["arm_torque"][ARM_JOINTS.index(name)] if name in ARM_JOINTS else sp["finger_torque"]
        if sp["arm_servo_constraint"] and name in ARM_JOINTS:
            kp = kd = 0.0           # carries the commanded angle only (ctrl); the torque comes from the servo constraint
        acts.append(f'    <position name="{name}" joint="{name}" kp="{kp:g}" kv="{kd:g}" ctrllimited="false" '
                    f'forcelimited="true" forcerange="-{lim:g} {lim:g}"/>')
    pair_default, contact = "", ""
    if sp["pair_friction"]:
        # PhysX combines friction as the average of the two materials, MuJoCo as the larger: every robot-cube,
        # robot-table and cube-table pair gets its averaged value as an explicit pair (static values, README 10)
        assert len(set(col_geoms)) == len(col_geoms), "one collision geom per link expected"
        pf, pairs = sp["pair_friction"], []
        one = lambda a, b, mu: pairs.append(f'    <pair geom1="{a}" geom2="{b}" friction="{mu:g} {mu:g} 0.005 0.0001 0.0001"/>')
        one("cube", "table", pf["cube_table"])
        for link in col_geoms:
            kind = "finger" if link in FINGER_LINKS else "robot"
            one(f"{link}_col", "cube", pf[kind + "_cube"])
            one(f"{link}_col", "table", pf[kind + "_table"])
        pair_default = f'\n    <pair solref="{CONTACT_TC:g} 1" condim="3"/>'
        contact = "  <contact>\n" + "\n".join(pairs) + "\n  </contact>\n"
    servo_bodies, servo_tendons = "", ""
    if sp["arm_servo_constraint"]:
        # One reference hinge per arm joint (a heavy rotor on the world, no gravity torque) tied to the joint by a
        # fixed tendon whose dry friction is the motor's torque limit: the constraint solver makes the joint turn at
        # the reference's speed with at most that torque, together with the contacts. chain.py sets the reference
        # speed every MuJoCo step to clip((kp / kd) (target - q), +-1 rad/s), the overdamped PD's response
        tc, dr = sp["servo_solref"]
        bodies, tendons = [], []
        for name in ARM_JOINTS:
            lim = sp["arm_torque"][ARM_JOINTS.index(name)]
            bodies.append(f'    <body name="servo_ref_{name}" pos="0 0 -1"><inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>'
                          f'<joint name="servo_ref_{name}" type="hinge" axis="0 0 1" armature="1000" damping="0"/></body>')
            tendons.append(f'    <fixed name="servo_{name}" frictionloss="{lim:g}" solreffriction="{tc:g} {dr:g}">'
                           f'<joint joint="{name}" coef="1"/><joint joint="servo_ref_{name}" coef="-1"/></fixed>')
        servo_bodies = "\n".join(bodies) + "\n"
        servo_tendons = "  <tendon>\n" + "\n".join(tendons) + "\n  </tendon>\n"
    th = [s / 2 for s in TABLE_SIZE]
    c = CUBE_SIDE / 2
    xml = f"""<!-- Generated by sim2sim/build_scene.py (spec {spec}) from deploy/robot_v3/urdf/{sp["urdf"]} (JaxIsaacLab
     artifacts/deploy_2026-11-05). Do not edit by hand. World frame of deploy/MUJOCO_SIM2SIM.md section 4. -->
<mujoco model="jax_v3_deploy_2026-11-05_{spec}">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{MUJOCO_DT:.12g}" integrator="implicitfast" gravity="0 0 -9.81" cone="{CONE}" impratio="{IMPRATIO:g}"/>
  <default>
    <geom solref="{CONTACT_TC:g} 1"/>{pair_default}
  </default>
  <visual>
    <global offwidth="1280" offheight="960"/>
  </visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1="0.82 0.82 0.82" rgb2="0.74 0.74 0.74"/>
    <material name="grid" texture="grid" texrepeat="8 8" reflectance="0"/>
{chr(10).join(assets)}
  </asset>
  <worldbody>
    <light pos="0.5 0.5 3" dir="-0.2 -0.2 -1" diffuse="0.8 0.8 0.8"/>
    <light pos="-1.5 1.5 2.5" dir="0.5 -0.5 -1" diffuse="0.4 0.4 0.4"/>
    <camera name="front" {_look((-1.75, 1.15, 1.75), (-0.15, 0.18, 0.85))}/>
    <camera name="close" {_look((-0.75, 0.75, 1.15), (-0.30, 0.27, 0.82))}/>
    <geom name="floor" type="plane" size="3 3 0.1" material="grid" contype="1" conaffinity="3" friction="1 0.005 0.0001"/>
    <geom name="table" type="box" size="{_f(th)}" pos="{_f(TABLE_CENTER)}" rgba="0.72 0.58 0.42 1"
          contype="1" conaffinity="3" friction="{TABLE_FRICTION} 0.005 0.0001"/>
{servo_bodies}    <body name="cube" pos="{_f(CUBE_REF)}">
      <freejoint name="cube"/>
      <geom name="cube" type="box" size="{c:g} {c:g} {c:g}" mass="{CUBE_MASS:g}" rgba="0.85 0.2 0.15 1"
            contype="1" conaffinity="3" friction="{CUBE_FRICTION} 0.005 0.0001"/>
    </body>
{chr(10).join(robot)}
  </worldbody>
{contact}{servo_tendons}  <actuator>
{chr(10).join(acts)}
  </actuator>
</mujoco>
"""
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(xml)
    return OUT


if __name__ == "__main__":
    import mujoco
    for spec in SPECS:
        path = build(spec)
        m = mujoco.MjModel.from_xml_path(path)
        print(f"wrote {os.path.relpath(path)}: {m.nbody} bodies, {m.njnt} joints, {m.nu} actuators, {m.ngeom} geoms, "
              f"{m.npair} contact pairs")
