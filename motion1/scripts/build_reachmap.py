"""Solve the poses the palm controller actually commands, and save the feasible band.

A cell of the table is feasible when the arm can hold the grasp centre, fingers straight down,
at every pose a delivery there asks for: the grasp on a cube standing in it and the clear-air
pose above, and the release at table height and the carry pose above that. The arm loses
forward reach as it comes down, so the band is not a circle around the shoulder and cannot be
written down; it is solved, at 5 mm and about 1 degree (REACH_TILT), from the home pose, with the same IK the
controller uses.

    venv/bin/python motion1/scripts/build_reachmap.py      # rewrites src/humanoid_grasp_rl/reachmap.npz
    JAX_SETUP=real venv/bin/python motion1/scripts/build_reachmap.py   # the real setup's reachmap_real.npz
    JAX_SETUP=real venv/bin/python motion1/scripts/build_reachmap.py --yaws -45 -30 -15 0 15 30
        # maps per cube yaw and per hand turn (the cube's yaw, or a quarter more either way:
        # env.hand_turns), in reachmap_real_yaw.npz, which realcoords reads. A cube repeats every
        # 90 degrees, so -45..30 in 15 covers it; the yaw-0 map files stay as they are.

Rebuild it whenever a constant in env.py that the signature covers changes.
"""
import _paths  # noqa: F401  (puts src/ on the path)
import argparse

import numpy as np

import humanoid_grasp_rl.env as E
from humanoid_grasp_rl import JaxDeliveryEnv
from humanoid_grasp_rl.demonstration import TRAVEL_CLEAR

parser = argparse.ArgumentParser()
parser.add_argument("--yaws", type=float, nargs="+", default=None,
                    help="cube yaws in degrees: build the per-yaw map instead of the yaw-0 one (real setup)")
args = parser.parse_args()
if args.yaws and E.SETUP != "real":
    # Only realcoords reads a per-yaw map, the real setup's; in the sim setup this wrote an
    # untracked reachmap_yaw.npz that nothing used.
    parser.error("--yaws builds the real setup's per-yaw map (reachmap_real_yaw.npz): run it with JAX_SETUP=real")

env = JaxDeliveryEnv(stage="place")
env.reset(seed=0)

xs = np.round(np.arange(-0.08, 0.281, 0.02), 3)
ys = np.round(np.arange(-0.30, 0.301, 0.025), 3)


def reachable(position):
    """Inside the workspace first: the env clips every commanded pose to it, so a pose the IK
    could solve outside it is one the controller can never command."""
    if np.any(position < E.WORKSPACE_LOW) or np.any(position > E.WORKSPACE_HIGH):
        return False
    return env.reachable(position)


FLIPS = (False, True) if E.HAND_FLIP else (False,)


def solve(hand_yaw_deg):
    """grasp, place and feasible over the grid, for the hand turned hand_yaw_deg, per grasp: [flip, i, j],
    flip being the half-turned hand (FLIPS: plain only, or also half-turned where the env may take that
    grasp, HAND_FLIP). One delivery keeps one grasp from the pick to the release (env._servable), so a
    cube and its spot are served only where the same grasp serves both: realcoords.plan reads these.
    Saved as grasp_by_flip, place_by_flip, feasible_by_flip; grasp, place and feasible stay the cells either
    grasp serves (np.any over flip), which the env's sampler and every older reader take. Before
    2026-11-20 only that OR was saved, and a cube only the half-turned hand grips was "reachable" with a
    spot only the plain hand serves: the env refused the layout.
    With hand_roll held to its servo's +-88 (env.HAND_ROLL_LIMIT) the real setup's leaning hand holds its
    plain grasp at yaw 0 over 2 cells only, plain or half-turned over 36 (2026-11-20; 60 at +-120)."""
    return tuple(np.array(m) for m in zip(*[solve_grasp(hand_yaw_deg, flip) for flip in FLIPS]))


def merged(by_flip):
    """The cells any grasp serves: what the map held before the grasps were kept apart."""
    return np.any(by_flip, axis=-3)


def solve_grasp(hand_yaw_deg, flip):
    """solve() for one grasp: half-turned or not."""
    env._hand_yaws[env.moving] = np.radians(hand_yaw_deg)
    # Set every time: env.reset(seed=0) above chose a grasp for its own layout (HAND_FLIP), which is
    # drawn from the map loaded, so a half-turned one left there made the map depend on the last map
    # (2026-11-12: 60 cells, then 69 over the same band edge).
    env._hand_flips[env.moving] = flip
    grasp = np.zeros((len(xs), len(ys)), dtype=bool)
    place = np.zeros((len(xs), len(ys)), dtype=bool)
    clear_z = env.clear_height(E.OBJECT_Z)
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            spot = np.array([x, y])
            pick = env.pick_pose(spot)
            drop = env.drop_pose(spot, E.OBJECT_Z)
            over_pick = np.array([pick[0], pick[1], max(pick[2] + TRAVEL_CLEAR, clear_z)])
            over_drop = np.array([drop[0], drop[1], max(drop[2] + TRAVEL_CLEAR, clear_z)])
            grasp[i, j] = reachable(pick) and reachable(over_pick)
            # Setting a cube down is the harder half: the hand is lowest here with a cube under it.
            place[i, j] = reachable(drop) and reachable(over_drop)
    both = grasp & place
    # Drop the rim of the band: a cell whose neighbour the arm cannot reach is one where the arm is
    # at a joint stop, and the hand there can no longer hold the fingers straight down while it
    # moves, so a carried cube slips out. Keeping only cells with all four neighbours feasible
    # leaves the controller a pose's worth of margin in every direction.
    padded = np.pad(both, 1, constant_values=False)
    feasible = (both & padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:])
    # And a cube there must stand on the table.
    feasible &= (xs <= E.CUBE_X_MAX)[:, None]
    return grasp, place, feasible


if args.yaws:
    yaws = np.array(sorted({E.fold_yaw(y) for y in args.yaws}))
    offsets = np.array([0.0, 90.0, -90.0])   # hand yaw = cube yaw + offset (env.hand_turns)
    maps = [[solve(y + o) for o in offsets] for y in yaws]
    by = [np.array([[m[k] for m in row] for row in maps]) for k in range(3)]   # [yaw, offset, flip, i, j]
    out = E.REACH_FILE.with_name(E.REACH_FILE.stem + "_yaw.npz")
    np.savez_compressed(out, yaws=yaws, offsets=offsets, xs=xs, ys=ys, flips=np.array(FLIPS),
                        grasp=merged(by[0]), place=merged(by[1]), feasible=merged(by[2]),
                        grasp_by_flip=by[0], place_by_flip=by[1], feasible_by_flip=by[2],
                        signature=E.map_signature())
    print(f"{out.name}: cells both grasp and place, per cube yaw and hand turn (offset from the cube yaw), "
          "plain / half-turned hand")
    for y, row in zip(yaws, maps):
        counts = "  ".join(f"{o:+4.0f}: " + "/".join(f"{f.sum():2d}" for f in m[2]) for o, m in zip(offsets, row))
        usable = E.hand_turns(y)
        union = np.any([merged(m[2]) for o, m in zip(offsets, row) if y + o in usable], axis=0)
        print(f"  yaw {y:+5.1f}  {counts}   any usable turn: {union.sum():3d}")
    env.close()
    raise SystemExit(0)

by = solve(0.0)
grasp, place, feasible = (merged(m) for m in by)
extra = {} if len(FLIPS) == 1 else dict(   # the sim setup has the plain grasp only: its map stays as it was
    flips=np.array(FLIPS), grasp_by_flip=by[0], place_by_flip=by[1], feasible_by_flip=by[2])
np.savez_compressed(E.REACH_FILE,   # the map of the setup JAX_SETUP picks (env.py)
                    xs=xs, ys=ys, grasp=grasp, place=place, feasible=feasible,
                    signature=E.map_signature(), **extra)
if len(FLIPS) > 1:
    print("손 그대로/반 바퀴 둘다: " + " / ".join(f"{int(f.sum())}" for f in by[2])
          + f",  같은 칸 둘 다 {int(np.all(by[2], axis=0).sum())}")
print(f"격자 {grasp.size},  파지가능 {grasp.sum()},  놓기가능 {place.sum()},  "
      f"둘다 {feasible.sum()} ({100 * feasible.mean():.0f}%)")
idx = np.argwhere(feasible)
if len(idx):
    print(f"둘다 가능 범위: x {xs[idx[:, 0]].min():.2f}~{xs[idx[:, 0]].max():.2f}  "
          f"y {ys[idx[:, 1]].min():+.2f}~{ys[idx[:, 1]].max():+.2f}")
    # The hand's orientation (env._hand_rotation, so every pose solved above) depends on HAND_HOLD_X,
    # which env.py takes from the band's near edge in the map it loaded: a new edge solves differently.
    # 2026-11-12: built over the last setup's map 58 cells, built again 51, and stays 51.
    if abs(float(xs[idx[:, 0]].max()) - E.WORK_X_MAX) > 1e-9:
        print(f"!! the band's near edge moved ({E.WORK_X_MAX:.2f} -> {xs[idx[:, 0]].max():.2f}): "
              "HAND_HOLD_X moves with it, so run this again until it stays")
print("\n" + "  " + " ".join(f"{y:+.2f}"[:5] for y in ys[::2]))
for i, x in enumerate(xs):
    row = " ".join(("#" if feasible[i, j] else ("g" if grasp[i, j] else
                    ("p" if place[i, j] else "."))).center(5) for j in range(0, len(ys), 2))
    print(f"{x:+.2f} {row}")
print("\n# both, g grasp only, p release only, . neither")
env.close()
