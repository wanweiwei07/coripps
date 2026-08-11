"""RRT-Connect motion planning for machine 1 (the COBOTTA at 192.168.0.101).

The start configuration is READ from the real controller, the goal is a random
collision-free configuration, and the path is planned with RRT-Connect against
the mobile system (machine 2 and the ground are static obstacles).

The plan is previewed as a looping animation first. Press ``g`` to send it to
the controller: the arm is taken and the motors powered on the first ``g``, and
the waypoints are played. The arm then STAYS live -- powering the motors costs
a brake release on both edges -- until the window closes or a move fails.
Press ``space`` to throw the plan away and plan a fresh
one to a new random goal, starting from wherever the arm is now -- so ``space``,
``g``, ``space``, ``g`` walks the arm from pose to pose. Press ``q``/``esc``
(or close the window) to quit.
"""

import os
import sys
import time

import numpy as np
import pyglet.window.key as pkey

# Allow running as `python coripps/machine1_planning.py` from the project root.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import one.utils.constant as ouc
import one.scene.scene_object_primitive as ossop
import one.viewer.world as ovw
import one.collider.mj_collider as ocm
import one.motion.core.planning_context as omppc
import one.motion.probabilistic.rrt as ompr
from one.drivers.cobotta import BCapError, CobottaMotion

from coripps.robots.mobisys import mobisys


# This PC is 192.168.0.100; machine 1 is the leaf sampler, machine 2 the tube
# handler.
_MACHINE1_HOST = '192.168.0.101'
_EXEC_SPEED = 100.0                      # external speed/accel/decel, percent
_CD_STEP_SIZE = np.pi / 180             # edge collision-check resolution
_EXTEND_STEP_SIZE = np.pi / 36          # RRT extension step
_MAX_ITERS = 3000
_SHORTCUT_ITERS = 200
_PREVIEW_STEP = np.pi / 60              # preview animation resolution
_PREVIEW_INTERVAL = 0.03
_MIN_GOAL_DISTANCE = np.pi / 2          # keep the random goal a real motion away
_GOAL_TRIES = 200
_START_TOLERANCE = np.deg2rad(2.0)      # real-vs-planned start mismatch guard


def connect_machine1(host=_MACHINE1_HOST):
    """Connect to machine 1; return None (and say so) when unreachable."""
    try:
        cobotta = CobottaMotion(host, speed=_EXEC_SPEED).connect()
    except (OSError, BCapError) as exc:
        print(f'machine1: cannot reach {host} ({exc})')
        return None
    print(f'machine1: connected to {host}')
    return cobotta


def build_scene():
    """Return the world plus the mobile system bodies."""
    base = ovw.World(cam_pos=(1.25, 0.75, 0.65),
                     cam_lookat_pos=(0.45, -0.10, 0.12))
    ossop.frame().attach_to(base.scene)
    mobisys_base, machine2, machine1 = mobisys()
    mobisys_base.attach_to(base.scene)
    machine2.attach_to(base.scene)
    machine2.gripper.attach_to(base.scene)
    machine1.attach_to(base.scene)
    return base, mobisys_base, machine2, machine1


def build_planning_context(machine1, machine2):
    """Machine 1 is the actor; machine 2 and the cart top are obstacles."""
    ground = ossop.plane()
    collider = ocm.MJCollider()
    collider.append(machine1)
    collider.append(machine2)
    collider.append(ground)
    collider.actors = [machine1]
    collider.compile()
    return omppc.PlanningContext(collider=collider,
                                 cd_step_size=_CD_STEP_SIZE), ground


def sample_goal(pln_ctx, start):
    """Draw a random collision-free configuration well away from ``start``."""
    for _ in range(_GOAL_TRIES):
        goal = pln_ctx.sample_uniform()
        if pln_ctx.distance(start, goal) < _MIN_GOAL_DISTANCE:
            continue
        if pln_ctx.is_state_valid(goal):
            return goal
    return None


def plan(pln_ctx, start, goal):
    """Plan with RRT-Connect and shortcut the result. None when it fails."""
    planner = ompr.RRTConnectPlanner(pln_ctx=pln_ctx,
                                     extend_step_size=_EXTEND_STEP_SIZE)
    t0 = time.time()
    path = planner.solve(start=start, goal=goal, max_iters=_MAX_ITERS)
    elapsed = time.time() - t0
    if not path:
        print(f'planning: no path after {elapsed:.3f}s '
              f'(RRT is probabilistic -- rerun the script)')
        return None
    path = ompr.shortcut_path(path, pln_ctx, n_iter=_SHORTCUT_ITERS)
    print(f'planning: path found in {elapsed:.3f}s, '
          f'{len(path)} waypoints after shortcutting')
    return path


def main():
    base, mobisys_base, machine2, machine1 = build_scene()
    pln_ctx, ground = build_planning_context(machine1, machine2)

    cobotta = connect_machine1()
    if cobotta is None:
        print('machine1: no controller -- planning from the simulated pose, '
              "'g' will do nothing")

    # 'start'/'path'/'preview' are swapped wholesale by replan(), so they live
    # in the state dict rather than in closure variables.
    state = {'mode': 'preview', 'i': 0, 'start': None, 'path': None,
             'preview': None}
    ghosts = {}

    def current_pose():
        """Where the next plan has to start from."""
        if cobotta is not None:
            return cobotta.joint_angles(ndof=machine1.ndof)
        if state['path'] is not None:
            # no controller: pretend the last plan ran, otherwise the start
            # would be wherever the preview animation happens to have stopped
            return state['path'][-1]
        # the active-joint slice of the sim pose, i.e. what the planner plans
        active = machine1.structure.compiled.active_jnt_ids_mask
        return np.asarray(machine1.qs, dtype=np.float32)[active]

    def show_ghosts(start, goal):
        """Translucent copies of the arm at both ends of the current plan."""
        if not ghosts:
            for name, rgba in (('start', (1, 0, 0, 0.3)),
                               ('goal', (0, 0, 1, 0.3))):
                ghost = machine1.clone()
                ghost.rgba = rgba
                ghost.attach_to(base.scene)
                ghosts[name] = ghost
        ghosts['start'].fk(qs=start)
        ghosts['goal'].fk(qs=goal)

    def replan():
        """Plan a fresh motion to a new random goal from where the arm is now.
        False when it fails -- the previous plan is then left untouched."""
        try:
            start = pln_ctx.enforce_bounds(current_pose())
        except (OSError, BCapError) as exc:
            print(f'plan: lost {cobotta.host} ({exc})')
            return False
        print(f'start: qs_deg={np.round(np.degrees(start), 2)}')
        if not pln_ctx.is_state_valid(start):
            print('start: the current pose is in collision -- cannot plan '
                  'from it')
            return False
        goal = sample_goal(pln_ctx, start)
        if goal is None:
            print(f'goal: no valid random pose in {_GOAL_TRIES} tries')
            return False
        print(f'goal:  qs_deg={np.round(np.degrees(goal), 2)}')
        path = plan(pln_ctx, start, goal)
        if path is None:
            return False
        show_ghosts(start, goal)
        state['start'] = start
        state['path'] = path
        state['preview'] = ompr.densify_path(path, pln_ctx,
                                             max_step=_PREVIEW_STEP)
        state['mode'] = 'preview'
        state['i'] = 0
        return True

    if not replan():
        if cobotta is not None:
            cobotta.close()
        return
    print("preview: looping the plan -- 'g' runs it, 'space' replans")

    def start_execution():
        if state['mode'] != 'preview':
            return
        if cobotta is None:
            print("execute: no controller connected -- nothing to send")
            return
        try:
            real_qs = cobotta.joint_angles(ndof=machine1.ndof)
        except (OSError, BCapError) as exc:
            print(f'execute: lost {cobotta.host} ({exc})')
            state['mode'] = 'done'
            return
        drift = np.max(np.abs(real_qs - state['start']))
        if drift > _START_TOLERANCE:
            print(f'execute: the arm moved since planning '
                  f'({np.degrees(drift):.2f} deg off the planned start) -- '
                  f"refusing to run the path ('space' replans from here)")
            return
        try:
            cobotta.enable()
        except (OSError, BCapError) as exc:
            print(f'execute: cannot take the arm ({exc})')
            state['mode'] = 'done'
            return
        state['mode'] = 'execute'
        state['i'] = 0
        print(f"execute: running {len(state['path'])} waypoints "
              f'at {_EXEC_SPEED:.0f}%')

    def step_execution():
        path = state['path']
        i = state['i']
        if i >= len(path):
            # the arm stays taken and powered: a motor cycle costs a brake
            # release, and 'space'/'g' is meant to be pressed over and over.
            # cobotta.close() in __main__ releases it when the window closes.
            state['mode'] = 'done'
            print("execute: done, arm still live -- 'space' plans the next "
                  "move")
            return
        try:
            # one blocking Move per tick; the last one stops precisely
            cobotta.move_qs(path[i], pass_motion=(i < len(path) - 1))
            machine1.fk(qs=cobotta.joint_angles(ndof=machine1.ndof))
        except (OSError, BCapError) as exc:
            # release on failure, so the next enable() runs ClearError
            print(f'execute: move {i} failed ({exc}) -- stopping')
            cobotta.disable()
            state['mode'] = 'done'
            return
        state['i'] = i + 1

    def tick(dt):
        # never replan or re-trigger while the arm is actually moving
        if state['mode'] == 'execute':
            step_execution()
            return
        if base.input_manager.is_key_pressed_edge(pkey.G):
            start_execution()
            return
        if base.input_manager.is_key_pressed_edge(pkey.SPACE):
            print('replan: new random goal from the current pose')
            if replan():
                print("preview: looping the plan -- 'g' runs it, "
                      "'space' replans")
            else:
                print('replan: failed -- keeping the previous plan')
            return
        if state['mode'] == 'preview':
            preview = state['preview']
            machine1.fk(qs=preview[state['i']])
            state['i'] = (state['i'] + 1) % len(preview)

    base.schedule_interval(tick, interval=_PREVIEW_INTERVAL)
    return base, cobotta, machine1, machine2, mobisys_base, state


if __name__ == '__main__':
    import builtins

    result = main()
    if result is not None:
        base, cobotta, machine1, machine2, mobisys_base, state = result
        builtins.base = base
        builtins.cobotta = cobotta
        builtins.machine1 = machine1
        builtins.machine2 = machine2
        builtins.state = state
        try:
            base.run()
        finally:
            if cobotta is not None:
                cobotta.close()
