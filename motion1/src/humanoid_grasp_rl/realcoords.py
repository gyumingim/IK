"""Real robot coordinates -> motion1's sim world, and whether the arm can serve a cube there.

numpy only, and nothing from this package (its __init__ loads MuJoCo), so a script on the real robot
loads the file on its own:

    sys.path.insert(0, "<repo>/motion1/src/humanoid_grasp_rl")
    from realcoords import real_to_sim
    real_to_sim(0.10, -0.38, 0.4775)   # cube 38 cm ahead, 10 cm to the left, table 477.5 mm down
    plan((0.10, -0.30, -20.0), (0.05, -0.27), 0.4775)   # that cube turned -20 deg, and where it goes

Real coordinates (user, 2026-11-05), in metres: origin on the axis of the shoulder roll motor (the
first arm joint, `shoulder_roll`, at its joint origin: in CAD the motor's corner is 1 mm from it),
x to the robot's left, y backwards (the front is -y), z up; right-handed.
The sim world is that frame turned by SIM_FROM_REAL_YAW about z. The spec's -90 degrees puts the
front on sim -x: x_real = -y_sim, y_real = x_sim. The front is the stand's FRONT side, where the
table is: the real setup turns the robot -90 degrees as JaxIsaacLab does (env.py, robot_yaw; user
2026-11-12, as on 2026-11-10; on 2026-11-11 it was the side the shoulder bracket sticks out to).
The arm then hangs beside the stand on sim +y, its left seen from the table.
A cube's yaw is in degrees, counter-clockwise seen from above, of one of its faces' normals from the
real +x axis. In the sim it is that plus SIM_FROM_REAL_YAW, folded to [-45, 45) since a cube looks
the same every 90 degrees (env.fold_yaw).
run_tests.sh checks the constants below against env.py and the model.
"""
from pathlib import Path

import numpy as np

# The shoulder roll joint's origin in the sim world: the shoulder frame at x 0.34, y 0, and the joint
# 0.052 from it along the stand's -x, which the real setup's robot, turned -90 degrees, points to
# world +y (motion2/arm/make_arm_urdf.py, URDF_1108; env.py, robot_yaw).
ORIGIN_XY = (0.34, 0.052)
SIM_FROM_REAL_YAW = -90.0
# The real setup the reach map was solved for (env.py SETUPS["real"]): table top above the floor,
# D (shoulder axis down to the table top) and the cube.
TABLE_Z = 0.735
MAP_D = 0.4775
CUBE_SIZE = 0.05
# Where the grasp centre (the `end_effector` site, between the fingers) is sent before it goes down:
# this far above the cube's top face (the spec asks for 2-3 cm).
APPROACH = 0.025
# A D further than this from MAP_D is a table the map was not solved for.
D_TOLERANCE = 0.005
REACH_FILE = Path(__file__).with_name("reachmap_real.npz")
# Per cube yaw and hand turn (scripts/build_reachmap.py --yaws), for a turned cube.
YAW_REACH_FILE = Path(__file__).with_name("reachmap_real_yaw.npz")


def fold_yaw(degrees: float) -> float:
    """Same as env.fold_yaw (run_tests.sh checks): [-45, 45)."""
    return float((degrees + 45.0) % 90.0 - 45.0)


def hand_turns(cube_yaw_deg: float) -> list:
    """Same as env.hand_turns (run_tests.sh checks): the cube's yaw or a quarter more, least first."""
    y = fold_yaw(cube_yaw_deg)
    return [0.0] if y == 0.0 else sorted((y, y + 90.0, y - 90.0), key=abs)


def to_sim(point, origin=(ORIGIN_XY[0], ORIGIN_XY[1], TABLE_Z + MAP_D), yaw=SIM_FROM_REAL_YAW):
    """A real point (x, y, z about the origin) in the sim world, the origin at `origin` there."""
    c, s = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
    x, y, z = point
    return np.asarray(origin, dtype=float) + np.array([c * x - s * y, s * x + c * y, z])


def reachable_spot(xy, reach_file=REACH_FILE) -> tuple[bool, str]:
    """Is this sim-world cube spot inside the reach map: in the grid, its nearest cell feasible and
    the spot inside that cell (no interpolation between cells)?"""
    with np.load(reach_file) as reach:
        xs, ys, feasible = reach["xs"], reach["ys"], reach["feasible"]
    half_x, half_y = (xs[1] - xs[0]) / 2, (ys[1] - ys[0]) / 2
    if not (xs[0] - half_x <= xy[0] <= xs[-1] + half_x and ys[0] - half_y <= xy[1] <= ys[-1] + half_y):
        return False, "outside the reach map's grid"
    i, j = int(np.argmin(np.abs(xs - xy[0]))), int(np.argmin(np.abs(ys - xy[1])))
    return (True, "in the reach band") if feasible[i, j] else (False, "the arm cannot serve that cell")


def _cell(xs, ys, xy):
    """The nearest grid cell of a spot, or None outside the grid."""
    half_x, half_y = (xs[1] - xs[0]) / 2, (ys[1] - ys[0]) / 2
    if not (xs[0] - half_x <= xy[0] <= xs[-1] + half_x and ys[0] - half_y <= xy[1] <= ys[-1] + half_y):
        return None
    return int(np.argmin(np.abs(xs - xy[0]))), int(np.argmin(np.abs(ys - xy[1])))


def grasp_name(turn, flip, tilt=None) -> str:
    """A grasp as the reasons print it: the hand's turn, whether it is half-turned, and its lean."""
    return f"hand turned {turn:+.0f} deg" + (", half-turned" if flip else "") + (
        "" if tilt is None else f", leaning {tilt:+.0f} deg")


def grasp_maps(yaw_sim, reach_file=YAW_REACH_FILE, by_tilt=False):
    """The reach band of each grasp for a cube turned yaw_sim (degrees, sim): (xs, ys, [(hand yaw in
    degrees, flip, feasible[i, j])]), least turn first and the plain hand before the half-turned one
    (flip: the half-turned hand, env.HAND_FLIP). The grasps are the cube's three faces (the env tries
    all of them on the real setup, env._grasp_turns) on the map's nearest yaw (90-degree period). A map
    built before the grasps were kept apart (no feasible_by_flip) gives flip None for either.
    by_tilt: one entry per grasp the map was solved for, (hand yaw, flip, lean in degrees, feasible), the
    lean being part of the grasp (env.HAND_TILTS); a map with one lean gives lean None. Without it a flip's
    entry is the cells some lean of it serves (the view before the leans)."""
    with np.load(reach_file) as reach:
        yaws, offsets, xs, ys = (reach[k] for k in ("yaws", "offsets", "xs", "ys"))
        if by_tilt and "feasible_by_grasp" in reach.files:
            flips = list(zip((bool(f) for f in reach["grasp_flips"]), (float(t) for t in reach["grasp_tilts"])))
            feasible = reach["feasible_by_grasp"]
        elif by_tilt and "feasible_by_flip" in reach.files:
            flips, feasible = [(bool(f), None) for f in reach["flips"]], reach["feasible_by_flip"]
        elif by_tilt:
            flips, feasible = [(None, None)], reach["feasible"][:, :, None]
        elif "feasible_by_flip" in reach.files:
            flips, feasible = [bool(f) for f in reach["flips"]], reach["feasible_by_flip"]
        else:
            flips, feasible = [None], reach["feasible"][:, :, None]
    k = int(np.argmin(np.abs([fold_yaw(v - yaw_sim) for v in yaws])))
    # The cube's yaw on the chosen map yaw's side of the +-45 fold. Measured from fold_yaw(yaw_sim)
    # instead, a cube at +37.5 to +45 (nearest map yaw -45) had each hand turn looked up 85-90 degrees
    # from itself: +40 was answered with the map's hand at -45.
    rep = yaws[k] + fold_yaw(yaw_sim - yaws[k])
    y = fold_yaw(yaw_sim)
    out = []
    for turn in sorted((y, y + 90.0, y - 90.0), key=abs):
        o = int(np.argmin(np.abs(offsets - (turn - rep))))
        if abs(offsets[o] - (turn - rep)) > 1.0:
            continue   # a half turn from the map's yaw: not in the map
        if by_tilt:
            out += [(float(turn), flip, tilt, feasible[k, o, f]) for f, (flip, tilt) in enumerate(flips)]
        else:
            out += [(float(turn), flip, feasible[k, o, f]) for f, flip in enumerate(flips)]
    return xs, ys, out


def grasps_for(spots, yaw_sim, reach_file=YAW_REACH_FILE):
    """Every grasp that serves all of `spots` (sim xy) for a cube turned yaw_sim (degrees, sim), as
    {"hand_yaw": degrees, "flip": half-turned hand or not, "tilt": its lean} (grasp_maps' order), or None when a spot is
    off the grid. One delivery keeps its grasp from the pick to the release (env._servable), so a cube
    and its spot need one grasp that serves both. Nearest cells, no interpolation; the env checks the
    exact poses again, and also grasps up to GRASP_MISALIGN off the faces, which the map does not hold."""
    xs, ys, maps = grasp_maps(yaw_sim, reach_file, by_tilt=True)
    cells = [_cell(xs, ys, s) for s in spots]
    if any(c is None for c in cells):
        return None
    return [{"hand_yaw": turn, "flip": flip, "tilt": tilt} for turn, flip, tilt, m in maps if all(m[c] for c in cells)]


def cube_grasps(cube, d: float = MAP_D) -> dict:
    """The grasps that can pick a cube = (x, y, yaw) in real coordinates (m, degrees) and set it down
    again (grasps_for), with the sim yaw: a target serves that cube only where one of them serves it."""
    first = real_to_sim(cube[0], cube[1], d, yaw=cube[2])
    if abs(d - MAP_D) > D_TOLERANCE:
        return {"yaw": first["yaw"], "grasps": [], "reason": first["reason"]}
    g = grasps_for([first["cube"][:2]], first["yaw"])
    return {"yaw": first["yaw"], "grasps": g or [],
            "reason": "outside the reach map's grid" if g is None else
            ("no grasp holds the cube there" if not g else "in the reach band")}


def reachable_turned(spots, yaw_sim, reach_file=YAW_REACH_FILE):
    """Can the arm grip a cube turned yaw_sim (degrees, sim) and serve every spot in `spots` (sim
    xy) with one grasp (grasps_for)? Returns (ok, hand yaw in degrees or None, reason)."""
    g = grasps_for(spots, yaw_sim, reach_file)
    if g is None:
        return False, None, "outside the reach map's grid"
    if g:
        return True, g[0]["hand_yaw"], f"in the reach band with the {grasp_name(g[0]['hand_yaw'], g[0]['flip'], g[0]['tilt'])}"
    return False, None, f"no grasp serves it at cube yaw {fold_yaw(yaw_sim):+.0f} deg"


def plan(cube, target, d: float) -> dict:
    """A delivery on the real table: cube = (x, y, yaw) and target = (x, y) in real coordinates (m,
    degrees), the table top d [m] below the shoulder axis. Returns the sim cube centre, target spot,
    cube yaw and hand yaw, the approach target, and whether the arm can do it, with the reason; also
    the grasp (flip: the half-turned hand) and the grasps each spot alone allows (grasps_for).
    env.reset(options={"objects": [[*cube_sim[:2], yaw_sim]], "targets": [target_sim]}) runs it."""
    x, y, yaw = cube
    first = real_to_sim(x, y, d, yaw=yaw)
    target_sim = to_sim((target[0], target[1], 0.0))[:2]
    out = {"cube": first["cube"], "approach": first["approach"], "target": target_sim,
           "yaw": first["yaw"], "hand_yaw": None}
    if abs(d - MAP_D) > D_TOLERANCE:
        return {**out, "reachable": False, "reason": first["reason"]}
    # Each spot alone first, then the grasps both share: a cube only the half-turned hand grips and a
    # spot only the plain hand serves is no delivery (the env refuses it, 2026-11-20).
    cube_g = grasps_for([first["cube"][:2]], first["yaw"])
    target_g = grasps_for([target_sim], first["yaw"])
    both = grasps_for([first["cube"][:2], target_sim], first["yaw"])
    out.update(cube_grasps=cube_g, target_grasps=target_g, grasps=both, flip=None, tilt=None)
    if both is None:
        return {**out, "reachable": False, "reason": "outside the reach map's grid"}
    if both:
        g = both[0]
        return {**out, "hand_yaw": g["hand_yaw"], "flip": g["flip"], "tilt": g["tilt"], "reachable": True,
                "reason": f"in the reach band with the {grasp_name(g['hand_yaw'], g['flip'], g['tilt'])}"}
    if not cube_g:
        why = f"no grasp holds the cube there at cube yaw {first['yaw']:+.0f} deg"
    elif not target_g:
        why = "no grasp sets a cube down on the target"
    else:
        names = lambda gs: " or ".join(grasp_name(g["hand_yaw"], g["flip"], g["tilt"]) for g in gs)
        why = (f"the cube only with the {names(cube_g)}, the target only with the {names(target_g)}: "
               "no one grasp serves both")
    return {**out, "reachable": False, "reason": why}


def real_to_sim(x: float, y: float, d: float, reach_file=REACH_FILE, yaw=None) -> dict:
    """A cube on the real table at (x, y) [m], the table top `d` [m] below the shoulder axis.

    Returns the cube centre and the approach target (grasp centre APPROACH above the cube's top
    face) in the sim world, and whether the arm can serve the spot, with the reason. Only a table
    at the D the map was solved for can be judged (D_TOLERANCE). With `yaw` (real, degrees) the
    result also has the sim yaw, and the spot is judged for a cube turned that way (both picking
    and setting down there, YAW_REACH_FILE)."""
    cube = to_sim((x, y, -d + CUBE_SIZE / 2))
    approach = cube + np.array([0.0, 0.0, CUBE_SIZE / 2 + APPROACH])
    out = {"cube": cube, "approach": approach}
    if yaw is not None:
        out["yaw"] = fold_yaw(yaw + SIM_FROM_REAL_YAW)
    if abs(d - MAP_D) > D_TOLERANCE:
        ok, why = False, f"no reach map for D = {d:.4f} m (the map is for {MAP_D} m)"
    elif yaw is not None and out["yaw"] != 0.0:
        ok, _, why = reachable_turned([cube[:2]], out["yaw"])
    else:
        ok, why = reachable_spot(cube[:2], reach_file)
    return {**out, "reachable": ok, "reason": why}
