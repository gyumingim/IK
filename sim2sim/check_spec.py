"""Check the MuJoCo port against numbers printed in Jungwoo's deploy folder (no physics run).

    venv/bin/python sim2sim/check_spec.py                 # default spec (25550); prints "spec ok" or fails on the first mismatch
    venv/bin/python sim2sim/check_spec.py --spec 23025    # the 11-08 spec (numbers of deploy/README_1108.md)

1. layout: shoulder pitch axis along world y through Shoulder_Pitch (0, 0.16, z), z 1.2075 (25550: 47.25 cm above
   the table top) or 1.2125 (23025), table top 0.735 (MUJOCO_SIM2SIM.md 4), robot base quat (0.7071, 0, 0, -0.7071),
   observation reference point and dz; 25550 also: motor torque limits, arm kp 500 kd 40, pair friction
2. forward kinematics: the recorded Isaac hand-over states (handover_state_*_example.json) give the same hand
   position / orientation and fingertip midpoint here (Isaac z is the sim layout's, shifted by the base offset)
3. lookup tables: README 7.1's knot table at the reference cube, 7.2's goal pose, 7.3's four reach-in deltas, 8.1's
   theta examples, 10's reach ranges; 25550 also: the 0.5 cm lower pre-grasp blend and the observation example
4. policy: the numpy actor equals the ONNX file's weights byte for byte and the meta normalizer
"""

import argparse
import json
import os
import sys

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import chain as C  # noqa: E402
import ptload  # noqa: E402


def close(a, b, tol, what):
    a, b = np.asarray(a, float), np.asarray(b, float)
    err = float(np.abs(a - b).max())
    assert err <= tol, f"{what}: max error {err:.4g} > {tol} ({a.round(3)} vs {b.round(3)})"
    return err


# numbers printed in deploy/README.md (25550) and deploy/README_1108.md (23025)
README = {
    "25550": {
        "shoulder_z": 1.2075, "obs_origin_z": 0.3575, "dz": -0.0575, "e": 0.0, "e_F": -0.005,
        "handover": "handover_state_25550_example.json",
        "knots_71": {5.0: (-37.15, 0, 1.38, 62.86, -5.19, 55.39, -8.87, -170.1),
                     5.6: (-34.78, 0, 1.40, 63.01, -5.11, 52.87, -8.93, -170.1),
                     6.2: (-32.29, 0, 1.40, 62.75, -5.10, 50.55, -8.95, -170.1),
                     6.8: (-29.71, 0, 1.39, 62.34, -5.12, 48.47, -8.95, -170.1),
                     7.4: (-26.95, 0, 1.36, 61.51, -5.22, 46.56, -8.90, -170.1),
                     8.0: (-24.13, 0, 1.33, 60.31, -5.33, 44.94, -8.84, -170.1)},
        "goal_72": (-24.13, 0, 1.33, 60.31, -5.33, 44.94, -8.84, -170.1),
        "delta_73": {(0, 0): (7.33, 0, 0.20, -1.86, -0.02, -5.46, 12.52, 0),          # F at e = 0 (README's examples)
                     (0, -0.10): (7.75, 0, -2.00, -2.69, -0.21, -5.05, 11.45, 0),
                     (0, 0.20): (9.95, 0, 5.62, -5.35, -0.85, -3.36, 16.79, 0),
                     (0.02, -0.28): (15.38, 0, -14.16, -13.01, 2.10, 3.29, 0.51, 0)},
        "pregrasp_54": (-16.80, 0, 1.53, 58.45, -5.35, 39.48, 3.68, -170.1),
        "q_l": (0, 0, 7.2, 90.0, 0, -2.6, 0), "squeeze": (1.2, 0.8),
        "torque": (8, 8, 8, 8, 3.92, 1.96, 0.92, 0.177, 0.177, 0.177, 0.177, 0.177, 0.177, 0.177),   # arm 8, fingers 6
    },
    "23025": {
        "shoulder_z": 1.2125, "obs_origin_z": 0.3625, "dz": -0.0625, "e": -0.005, "e_F": -0.005,
        "handover": "handover_state_23025_v3_example.json",
        "knots_71": {5.0: (-34.09, 0, 1.37, 62.41, -5.14, 57.47, -9.18, -170.1),
                     6.2: (-29.27, 0, 1.40, 62.26, -5.01, 52.70, -9.29, -170.1),
                     8.0: (-21.14, 0, 1.34, 59.71, -5.20, 47.25, -9.22, -170.1)},
        "goal_72": (-23.04, 0, 1.34, 57.61, -5.20, 46.55, -9.02, -170.1),
        "delta_73": {(0, 0): (7.28, 0, 0.18, -1.92, -0.04, -5.34, 12.53, 0),
                     (0, -0.10): (7.72, 0, -2.10, -2.79, -0.29, -4.92, 11.46, 0),
                     (0, 0.20): (10.10, 0, 5.97, -5.68, -0.67, -3.06, 16.82, 0),
                     (0.02, -0.28): (16.89, 0, -20.90, -14.19, -6.12, 4.49, 3.08, 0)},
        "pregrasp_54": None,
        "q_l": (1.9, 0, 7.2, 92.1, 0, -1.9, -0.2), "squeeze": (5.0, 4.0), "torque": (100,) * 14,
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default=C.SPEC, choices=sorted(C.SPECS))
    spec = ap.parse_args().spec
    sp, R = C.spec_of(spec), README[spec]
    m = mujoco.MjModel.from_xml_path(sp["scene_path"])
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    jid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
    bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
    gid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)

    # 1. layout
    zs = R["shoulder_z"]
    j = jid(C.ISAAC[0])
    close(d.xaxis[j], (0, 1, 0), 1e-6, "shoulder pitch axis direction")
    close(d.xanchor[j][[0, 2]], (0.0, zs), 1e-6, "shoulder pitch axis x, z")
    close(d.xpos[bid("Shoulder_Pitch")], (0.0, 0.16, zs), 1e-6, "Shoulder_Pitch frame")
    close(d.xpos[bid("base_link")], sp["base_pos"], 1e-9, "base_link position")
    close(d.xquat[bid("base_link")], (0.5 ** 0.5, 0, 0, -0.5 ** 0.5), 1e-9, "base quat")
    t = gid("table")
    close(d.geom_xpos[t][2] + m.geom_size[t][2], 0.735, 1e-9, "table top")
    close(d.geom_xpos[t][0] + m.geom_size[t][0], -0.225, 1e-9, "table front edge")
    close(zs - 0.735, sp["d"], 1e-9, "shoulder axis -> table top")
    close(sp["obs_origin"], (0, 0, R["obs_origin_z"]), 1e-9, "observation reference point")
    close(d.xpos[bid("Shoulder_Pitch")] - sp["obs_origin"], (0, 0.16, 0.85), 1e-6, "reference point below the shoulder axis")
    close(sp["dz"], R["dz"], 1e-9, "observation 30 dz")
    close((0.735 + C.CUBE_HALF - sp["obs_origin"][2]) - sp["dz"], 0.460, 1e-9, "8.1: cube on the table reads 0.460")
    act = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, C.ISAAC[i]) for i in C.ARM_IDS + C.FINGER_IDS]
    if sp["servo"]:     # arm: servo constraints (tendon friction = torque limit), fingers: PD actuators with the force range
        ten = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_TENDON, "servo_" + C.ISAAC[i]) for i in C.ARM_IDS]
        close(list(m.tendon_frictionloss[ten]) + list(m.actuator_forcerange[act[8:], 1]), R["torque"], 1e-9, "5.3 torque limits")
        close(m.actuator_gainprm[act, 0], [0.0] * 8 + [40.0] * 6, 1e-9, "5.3 kp (arm: constraint)")
        close(-m.actuator_biasprm[act, 2], [0.0] * 8 + [2.0] * 6, 1e-9, "5.3 kd")
        close(sp["servo_rate"], 500.0 / 40.0, 1e-12, "5.3 servo emulation kp / kd")
    else:
        close(m.actuator_forcerange[act, 1], R["torque"], 1e-9, "5.3 torque limits")
        close(m.actuator_gainprm[act, 0], [100.0] * 8 + [40.0] * 6, 1e-9, "5.3 kp")
        close(-m.actuator_biasprm[act, 2], [8.0] * 8 + [2.0] * 6, 1e-9, "5.3 kd")
    close(np.degrees(sp["q_l"]), R["q_l"], 1e-9, "7.7 L pose")
    close(sp["squeeze"], R["squeeze"], 0, "7.5 squeeze")
    print(f"layout: shoulder pitch axis z {zs} along world y, table top 0.735 ({sp['d'] * 100:g} cm below the axis), "
          f"front edge x -0.225, reference point z {R['obs_origin_z']}, dz {R['dz']}")
    if spec == "25550":
        # 10: PhysX average of the two materials, as explicit pairs
        want = {"cube": {True: 2.0, False: 1.2}, "table": {True: 1.6, False: 0.8}}
        seen = {}
        for k in range(m.npair):
            n1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, m.pair_geom1[k])
            n2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, m.pair_geom2[k])
            seen[frozenset((n1, n2))] = float(m.pair_friction[k, 0])
        assert seen.pop(frozenset(("cube", "table"))) == 1.6, "cube-table friction"
        robot = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) for g in range(m.ngeom) if m.geom_contype[g] == 2]
        for g in robot:
            for other in ("cube", "table"):
                mu = seen.pop(frozenset((g, other)))
                assert mu == want[other][g.startswith("Finger_")], f"friction {g}-{other}: {mu}"
        assert not seen and len(robot) == 16, (seen, len(robot))
        assert sp["obs_target"] and sp["return_a7"] == 1.0 and sp["feedback"] == "target"
        print("motors and friction: arm servo constraints (response of kp 500 kd 40), fingers PD kp 40 kd 2, torque limits 8/8/8/8/3.92/1.96/0.92 N m and 0.177 "
              "(RotationBody, fingers); pairs finger-cube 2.0, cube-table 1.6, finger-table 1.6, other link-cube 1.2, -table 0.8")

    # 2. forward kinematics vs the Isaac hand-over states
    ex = json.load(open(os.path.join(C.DEPLOY, R["handover"])))
    worst = 0.0
    for e in ex["envs"]:
        for n, v in zip(ex["joint_names_arm"] + ex["joint_names_fingers"], e["arm_q_deg"] + e["finger_q_deg"]):
            d.qpos[m.jnt_qposadr[jid(n)]] = np.radians(v)
        mujoco.mj_kinematics(m, d)
        off = np.array(sp["base_pos"]) - np.array(e["robot_base_pos_m"])
        tips = np.mean([d.xpos[bid(n)] for n in C.TIP_LINKS], axis=0)
        worst = max(worst, close(d.xpos[bid(C.HAND)], np.array(e["hand_pos_m"]) + off, 5e-4, "hand position"),
                    close(tips, np.array(e["fingertip_mid_pos_m"]) + off, 5e-4, "fingertip midpoint"))
        q = d.xquat[bid(C.HAND)]
        q = q if np.dot(q, e["hand_quat_wxyz"]) >= 0 else -q
        close(q, e["hand_quat_wxyz"], 5e-4, "hand orientation")
    print(f"forward kinematics: {len(ex['envs'])} Isaac hand-over states ({R['handover']}) within {worst * 1000:.2f} mm")

    # 3. lookup tables
    T = C.Tables()
    e_ = sp["e"]
    close(e_, R["e"], 1e-12, "table blend e")
    S = T.raise_knots(0, 0, e_)
    knots = np.vstack([np.zeros(8), S[0], S])
    for tk, row in R["knots_71"].items():
        close(np.degrees(C.catmull_rom(C.P_T, sp["path"], tk) + C.catmull_rom(C.S_T, knots, tk)), row, 0.006, f"7.1 knot {tk}")
    close(np.degrees(C.GOAL_BASE + S[5]), R["goal_72"], 0.006, "7.2 goal")
    for (x, y), row in R["delta_73"].items():
        dq = T.final_pose(x, y, e_) - T.raise_knots(x, y, e_)[5]
        dq[6] += C.ROLL_IN
        close(np.degrees(dq), row, 0.006, f"7.3 delta at {(x, y)}")
    if R["pregrasp_54"]:
        dq = T.final_pose(0, 0, e_) - S[5]
        dq[6] += C.ROLL_IN
        close(np.degrees(C.GOAL_BASE + S[5] + dq), R["pregrasp_54"], 0.011, "5.4 pre-grasp pose")
    # 7.3 (25550): the chain itself blends F at e_F = max(e - 0.005, -0.01), i.e. F_base + (1/6)(F_m3c - F_base) at e = 0
    ch = C.Chain(0.0, 0.0, 0.0, spec=spec, tables=T)
    close(ch.e_F, R["e_F"], 1e-12, "pre-grasp blend e_F")
    Fb, Fm = T._bilinear(T.final["basec"]["dq"][-1], 0, 0), T._bilinear(T.final["m3c"]["dq"][-1], 0, 0)
    close(ch.F, Fb + (R["e_F"] / -0.03) * (Fm - Fb), 1e-12, "pre-grasp F blend")
    if spec == "25550":
        for e_low, want in ((-0.01, -0.01), (-0.004, -0.009), (0.01, 0.005)):
            close(C.Chain(0.0, 0.0, 0.0, spec=spec, tables=T, policy=ch.policy, model=m, e=e_low).e_F, want, 1e-12, "e_F floor")
        # 8.1 example: the reference cube on the table reads (-0.26, -0.40, 0.460) in 28-30; 0-13 are the targets sent
        obs = ch.observation(np.zeros(6), ch.tip_mid())
        close(obs[28:31], (-0.26, -0.40, 0.460), 1e-6, "8.1 observation 28-30")
        close(obs[9], -0.7222, 0, "8.1 observation 9")
        ch.tg_prev = ch.target.copy()
        ch.target[ch.arm[3]] = 0.01
        obs = ch.observation(np.zeros(6), ch.tip_mid())
        close(obs[3], 2 * (0.01 + np.radians(105)) / np.radians(210) - 1, 1e-9, "8.1 observation 3 = target, V2 limits")
        close(obs[17], 0.01 / C.DT * 0.2, 1e-9, "8.1 observation 17 = target change / dt x 0.2")
    for (x, y), th in {(0, 0): 0.0, (0, -0.10): 15.9, (0, 0.20): -33.8, (0.02, -0.28): 57.6, (0.02, 0.28): -54.6}.items():
        close(T.theta_deg(x, y), th, 0.06, f"8.1 theta at {(x, y)}")
    ranges = {0.02: (-0.28, 0.28), 0.0: (-0.26, 0.28), -0.04: (-0.24, 0.26), -0.08: (-0.18, 0.22),
              -0.12: (-0.16, 0.16), -0.14: (-0.10, 0.08)}
    for x, (lo, hi) in ranges.items():
        ys = [y for y in np.round(np.arange(-0.30, 0.3001, 0.02), 2) if T.valid(x, y)]
        close((min(ys), max(ys)), (lo, hi), 1e-9, f"10 reach range at x {x}")
    print(f"lookup tables: 7.1 knots, 7.2 goal, 7.3 deltas, 8.1 theta, 10 reach ranges as the README (e {e_}, pre-grasp e_F {ch.e_F})")

    # 4. policy
    name = sp["policy"]
    a = ptload.load_actor(os.path.join(C.DEPLOY, name + ".pt"))
    meta = json.load(open(os.path.join(C.DEPLOY, name + "_meta.json")))
    close(a["mean"], meta["obs_normalizer"]["mean"], 0, "normalizer mean")
    close(a["std"], meta["obs_normalizer"]["std"], 0, "normalizer std")
    onnx = open(os.path.join(C.DEPLOY, name + ".onnx"), "rb").read()
    for k, (w, b) in enumerate(a["layers"]):
        assert w.astype(np.float32).tobytes() in onnx and b.astype(np.float32).tobytes() in onnx, f"layer {k} not in ONNX"
    assert [w.shape for w, _ in a["layers"]] == [(256, 45), (256, 256), (11, 256)]
    print(f"policy {name}: 45 -> 256 -> 256 -> 11, weights byte-identical to the ONNX file, normalizer = meta")
    print(f"spec ok ({spec})")


if __name__ == "__main__":
    main()
