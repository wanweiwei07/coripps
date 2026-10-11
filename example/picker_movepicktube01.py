"""The picker picks the FIRST precomputed tube, gripper included.

One tube, one run, arm and gripper together::

    transfer (RRT) -> pre_pick_high -> [OPEN gripper] -> pre_pick
                   -> pick -> [CLOSE gripper] -> lift

The gripper opens at ``pre_pick_high``, above the tube and clear of it, and
closes at ``pick`` -- the descent ``pre_pick -> pick`` therefore happens with
the blades already open, and the lift happens with the tube already held.

Where everything comes from:

* the four arm poses -- ``config/pick_poses.json`` via ``pick_poses_io``,
  entry ``TUBE_INDEX``; nothing is re-solved here and the solver is never
  imported,
* the transfer to the first pose -- RRT-Connect, because the arm starts
  wherever it was left,
* the arm plumbing (connect / freeze the sampler / collision model /
  preview) -- ``picker_motion``,
* the gripper -- ``grippers.xh430_gripper.exec`` on COM5.

The motion is PREVIEWED as a loop first and only runs on the real hardware
when ``g`` is pressed. A plan is only valid from the pose it was made at, and
the pick leaves the arm at ``lift`` -- so running it a second time means
pressing ``r`` first to plan from where the arm now is. Missing hardware is
not an error: without the COBOTTA or without the servo the preview still
loops, and ``g`` says what it cannot drive.

Keys:  r = plan from where the arm is    g = run that plan
       o = open the gripper (drop the tube)    t = gripper torque off
       esc = quit

    py -3.12 coripps/example/picker_movepicktube01.py
"""

import os
import sys

import numpy as np
import pyglet.window.key as pkey

# Allow running as `py -3.12 coripps/example/picker_movepicktube01.py`.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import one.utils.constant as ouc
import one.scene.scene_object as osso
import one.motion.probabilistic.rrt as ompr
from one.drivers.cobotta import BCapError

import coripps.pick_layout as layout
import coripps.pick_poses_io as ppio
import coripps.picker_motion as pkm
from coripps.grippers.xh430_gripper import OPEN_TRAVEL
from coripps.grippers.xh430_gripper.exec import RotaryGripper, DynamixelError


TUBE_INDEX = 0                   # which precomputed pick to run
_PREVIEW_INTERVAL = pkm._PREVIEW_INTERVAL
_PREVIEW_STEP = pkm._PREVIEW_STEP


def densify_with_marks(path, pln_ctx, max_step=_PREVIEW_STEP):
    """Densify ``path`` for the animation, remembering where the ORIGINAL
    waypoints ended up.

    ``marks[i]`` is the index in the dense path of original waypoint ``i`` --
    that is what lets the preview open and close the gripper at exactly the
    poses the real run will.
    """
    dense = [np.asarray(path[0], dtype=np.float32)]
    marks = [0]
    for start, goal in zip(path[:-1], path[1:]):
        segment = ompr.densify_path([start, goal], pln_ctx, max_step=max_step)
        dense.extend(segment[1:])
        marks.append(len(dense) - 1)
    return dense, marks


def build_pick_program(pln_ctx, start, seq):
    """The whole single-tube run as a typed step list.

    Arm steps carry a joint config, gripper steps carry a verb. Keeping them
    in ONE ordered list is what makes the interleaving obvious: the gripper
    opens between arriving at ``pre_pick_high`` and descending, and closes
    between arriving at ``pick`` and lifting.
    """
    transfer = pkm.build_transfer(pln_ctx, start, seq[0], force_plan=True)
    if transfer is None:
        return None, None
    steps = [('arm', q) for q in transfer]           # ... ends at pre_pick_high
    steps.append(('gripper', 'open'))
    steps.append(('arm', seq[1]))                    # pre_pick
    steps.append(('arm', seq[2]))                    # pick
    steps.append(('gripper', 'close'))
    steps.append(('arm', seq[3]))                    # lift
    return steps, transfer


def main():
    base, mobisys_base, picker, sampler = pkm.build_scene()
    pkm.freeze_sampler(sampler)
    cobotta = pkm.connect(pkm._PICKER_HOST, 'picker', motion=True)
    try:
        gripper = RotaryGripper().connect()
    except DynamixelError as exc:
        gripper = None
        print(f'{exc}\n         the gripper steps will be simulated only')

    points, sequences, _rotmats, meta = ppio.load_checked(
        ndof=picker.ndof, pick_z=layout.PICK_Z)
    if TUBE_INDEX >= len(points):
        raise IndexError(f'tube {TUBE_INDEX} of {len(points)} precomputed picks')
    point, seq = points[TUBE_INDEX], sequences[TUBE_INDEX]
    print(f'tube {TUBE_INDEX}: point={np.round(point, 4)}')

    tube = osso.SceneObject.from_file(pkm._TUBE_PATH, collision_type=None,
                                      rgb=ouc.ExtendedColor.SILVER, alpha=0.5)
    tube.set_pos_rotmat(pos=point + ppio.TUBE_POS_OFFSET,
                        rotmat=ppio.TUBE_ROTMAT)
    tube.attach_to(base.scene)

    pln_ctx, ground = pkm.build_planning_context(picker, sampler)
    if not pkm.local_sequence_valid(pln_ctx, seq):
        raise RuntimeError(f'tube {TUBE_INDEX}: the approach is in collision')

    if cobotta is not None:
        picker.fk(qs=cobotta.joint_angles(ndof=picker.ndof))
    else:
        # the sim rest pose (all zeros) is itself in collision, so there would
        # be nothing to plan from -- park the arm at the first pose
        picker.fk(qs=seq[0])
        print('picker: no controller -- animation only, nothing is sent')

    # the sim gripper is shown at the angle the REAL one is configured for, so
    # preview and execution agree even if the working open is retuned
    open_deg = (gripper.cfg.open_deg if gripper is not None
                else float(np.degrees(OPEN_TRAVEL)))
    open_width = picker.gripper.opening_at(np.deg2rad(open_deg))
    shut_width = float(picker.gripper.jaw_range[0])

    # everything the run is made of is re-derived by replan(), so it lives in
    # the state rather than in closure variables
    state = {'mode': 'preview', 'i': 0, 'start': None, 'steps': None,
             'dense': None, 'open_frame': 0, 'grip_frame': 0, 'stale': False}

    def current_pose():
        """Where a plan has to start."""
        if cobotta is not None:
            return cobotta.joint_angles(ndof=picker.ndof)
        if state['steps'] is not None:
            # no controller: pretend the last plan ran, otherwise the start
            # would be whatever frame the preview animation is showing
            return state['steps'][-1][1]
        return np.asarray(picker.qs, dtype=np.float32)

    def replan():
        """Rebuild the whole run from where the arm is NOW.

        A plan is only good from the pose it was made at, and the pick leaves
        the arm at ``lift`` -- so running it a second time means planning a
        second time. That is what 'r' is for; 'g' then executes what 'r'
        produced.
        """
        try:
            start = pln_ctx.enforce_bounds(current_pose())
        except (OSError, BCapError) as exc:
            print(f'plan: lost {cobotta.host} ({exc})')
            return False
        if not pln_ctx.is_state_valid(start):
            print('plan: the current pose is in collision -- cannot plan '
                  'from it')
            return False
        steps, transfer = build_pick_program(pln_ctx, start, seq)
        if steps is None:
            print(f'plan: tube {TUBE_INDEX} has no transfer from here')
            return False
        # the preview animates the ARM waypoints; the marks say which dense
        # frame each landed on, so the sim gripper acts at the same poses
        arm_path = [q for kind, q in steps if kind == 'arm']
        dense, marks = densify_with_marks(arm_path, pln_ctx)
        state['start'] = start
        state['steps'] = steps
        state['dense'] = dense
        state['open_frame'] = marks[len(transfer) - 1]   # at pre_pick_high
        state['grip_frame'] = marks[len(transfer) + 1]   # at pick
        state['i'] = 0
        state['stale'] = False
        print(f"plan: {len(steps)} steps, previewing "
              f"{len(dense) * _PREVIEW_INTERVAL:.1f}s -- 'g' runs it")
        return True

    def show_gripper(width):
        picker.gripper.set_opening(width)
        base.scene.dirty = True

    def sync_gripper_from_real():
        """Mirror the servo's own angle onto the sim gripper."""
        if gripper is None:
            return
        width = np.clip(gripper.opening_mm() / 1000.0,
                        picker.gripper.jaw_range[0],
                        picker.gripper.jaw_range[1])
        show_gripper(float(width))

    def step_preview():
        dense = state['dense']
        if not dense:
            return
        i = state['i'] % len(dense)
        picker.fk(qs=dense[i])
        # region, not edge: the width is right no matter where the loop starts
        show_gripper(open_width if state['open_frame'] <= i < state['grip_frame']
                     else shut_width)
        state['i'] = (i + 1) % len(dense)

    def start_execution():
        if state['steps'] is None:
            print("no plan -- press 'r' first")
            return
        if state['stale']:
            print("the plan is from before the last run -- press 'r' to "
                  'plan from here')
            return
        if cobotta is None:
            print('no controller -- nothing to send')
            return
        try:
            real_qs = cobotta.joint_angles(ndof=picker.ndof)
        except (OSError, BCapError) as exc:
            print(f'execute: lost {cobotta.host} ({exc})')
            return
        drift = np.max(np.abs(real_qs - state['start']))
        if drift > pkm._START_TOLERANCE:
            print(f'execute: the arm moved since planning '
                  f'({np.degrees(drift):.2f} deg off the planned start) -- '
                  f"refusing to run; press 'r' to plan from here")
            return
        try:
            cobotta.enable()
        except (OSError, BCapError) as exc:
            print(f'execute: cannot take the arm ({exc})')
            return
        state['mode'] = 'execute'
        state['i'] = 0
        print(f'executing at {pkm._EXEC_SPEED:.0f}%')

    def step_execution():
        steps = state['steps']
        i = state['i']
        if i >= len(steps):
            state['mode'] = 'preview'
            state['i'] = 0
            # the arm now stands at 'lift', so the plan that got it there no
            # longer starts where it is: the preview is a replay until 'r'
            state['stale'] = True
            print("tube picked, arm still live -- 'o' opens the gripper, "
                  "'r' plans from here, then 'g' picks again")
            return
        kind, payload = steps[i]
        if kind == 'gripper':
            if gripper is None:
                print(f'  step {i + 1}: {payload} (no servo -- skipped)')
                show_gripper(open_width if payload == 'open' else shut_width)
            else:
                if payload == 'open':
                    gripper.open()
                else:
                    gripper.close_jaws()
                sync_gripper_from_real()
                print(f'  step {i + 1}: gripper {payload} -> '
                      f'{gripper.angle_deg():5.2f} deg / '
                      f'{gripper.opening_mm():5.2f} mm @ '
                      f'{gripper.dxl.present_current_ma():4.0f} mA')
            state['i'] = i + 1
            return
        try:
            # one Move per tick; the last one stops precisely (@0), the rest
            # pass through (@P) -- but never pass through into a gripper step,
            # the arm has to be standing still while the blades move
            nxt = steps[i + 1][0] if i + 1 < len(steps) else None
            cobotta.move_qs(payload, pass_motion=(nxt == 'arm'))
            picker.fk(qs=cobotta.joint_angles(ndof=picker.ndof))
            base.scene.dirty = True
        except (OSError, BCapError) as exc:
            print(f'execute: move {i} failed ({exc}) -- stopping')
            cobotta.disable()
            state['mode'] = 'preview'
            state['i'] = 0
            return
        state['i'] = i + 1

    def tick(dt):
        if state['mode'] == 'execute':
            step_execution()
            return
        if base.input_manager.is_key_pressed_edge(pkey.G):
            start_execution()
            return
        if base.input_manager.is_key_pressed_edge(pkey.R):
            replan()
            return
        if base.input_manager.is_key_pressed_edge(pkey.O):
            # let go of the tube on purpose, so the next pick starts empty
            if gripper is not None:
                gripper.open()
                sync_gripper_from_real()
            else:
                show_gripper(open_width)
            return
        if base.input_manager.is_key_pressed_edge(pkey.T):
            if gripper is not None:
                gripper.release()
            return
        step_preview()

    if not replan():
        raise RuntimeError(f'tube {TUBE_INDEX}: cannot plan from the '
                           f'current pose')
    base.schedule_interval(tick, interval=_PREVIEW_INTERVAL)
    base.set_caption(f'the picker picking tube {TUBE_INDEX} -- '
                     f'r: plan, g: run')
    print("ready: 'g' runs the plan, 'r' plans again from where the arm is, "
          "'o' opens the gripper, 't' releases it")
    return base, cobotta, gripper, sampler, picker, state


if __name__ == '__main__':
    import builtins

    (base, cobotta, gripper, sampler, picker, state) = main()
    builtins.base = base
    builtins.cobotta = cobotta
    builtins.gripper = gripper
    builtins.sampler = sampler
    builtins.picker = picker
    builtins.state = state
    try:
        base.run()
    finally:
        if cobotta is not None:
            cobotta.close()
        if gripper is not None:
            gripper.release()
            gripper.close()
