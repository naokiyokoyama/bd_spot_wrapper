"""
Controlling the robot using a presentation tool ("pt").
Up and down will start moving the robot forward or backwards if in linear mode.
In angular mode, it will turn left or right (respectively).
Tab key will switch between linear and angular mode.
If the same key is pressed again while the robot is moving, it will stop.
"""
import os
import time

import numpy as np
from bosdyn.client.estop import EstopClient

from spot_wrapper.estop import EstopNoGui
from spot_wrapper.spot import Spot
from spot_wrapper.utils.headless import KeyboardListener

UPDATE_PERIOD = 0.2
LINEAR_VEL = 1.0
ANGULAR_VEL = np.deg2rad(50)
DOCK_ID = int(os.environ.get("SPOT_DOCK_ID", 520))


class FSM_ID:
    r"""Finite state machine IDs."""
    IDLE = 0
    ESTOP = 1
    FORWARD = 2
    BACK = 3
    LEFT = 4
    RIGHT = 5


class KEY_ID:
    r"""Keyboard id codes."""
    UP = 103
    DOWN = 108
    TAB = 15
    ENTER = 28


class SpotHeadlessTeleop(KeyboardListener):
    silent = True

    def __init__(self):
        self.mode = "linear"
        self.fsm_state = FSM_ID.ESTOP
        self.spot = Spot("HeadlessTeleop")
        self.lease = None
        estop_client = self.spot.robot.ensure_client(EstopClient.default_service_name)
        self.estop_nogui = EstopNoGui(estop_client, 5, "Estop NoGUI")
        self.last_up = 0
        self.last_down = 0
        super().__init__()

    def process_pressed_key(self, pressed_key):
        if pressed_key is None:
            return

        if pressed_key == KEY_ID.DOWN:
            self.last_down = time.time()
        elif pressed_key == KEY_ID.UP:
            self.last_up = time.time()

        if self.fsm_state == FSM_ID.ESTOP and pressed_key == KEY_ID.UP:
            self.fsm_state = FSM_ID.IDLE
            print("Hijacking robot.")
            self.hijack_robot()
        elif self.fsm_state != FSM_ID.ESTOP:
            if pressed_key == KEY_ID.ENTER:
                self.estop()
                self.fsm_state = FSM_ID.ESTOP
                print("E-Stop")
            elif (
                pressed_key in [KEY_ID.UP, KEY_ID.DOWN]
                and abs(self.last_up - self.last_down) < 0.2
            ):
                print("Halting and docking robot...")
                self.last_up = 0
                self.halt_robot()
                try:
                    self.spot.dock(DOCK_ID)
                    self.spot.home_robot()
                except:
                    print("Dock was not found!")
            elif (
                pressed_key == KEY_ID.UP
                and self.fsm_state in [FSM_ID.FORWARD, FSM_ID.LEFT]
            ) or (
                pressed_key == KEY_ID.DOWN
                and self.fsm_state in [FSM_ID.BACK, FSM_ID.RIGHT]
            ):
                self.halt_robot()
                self.fsm_state = FSM_ID.IDLE
                print("Halting robot.")
            elif self.fsm_state == FSM_ID.IDLE and pressed_key == KEY_ID.TAB:
                if self.mode == "linear":
                    self.mode = "angular"
                    print("Switching to angular mode.")
                else:
                    self.mode = "linear"
                    print("Switching to linear mode.")
            elif pressed_key == KEY_ID.UP:
                if self.mode == "linear":
                    self.move_forward()
                    self.fsm_state = FSM_ID.FORWARD
                    print("Moving forwards.")
                else:
                    self.turn_left()
                    self.fsm_state = FSM_ID.LEFT
                    print("Turning left.")
            elif pressed_key == KEY_ID.DOWN:
                if self.mode == "linear":
                    self.move_backwards()
                    self.fsm_state = FSM_ID.BACK
                    print("Moving backwards.")
                else:
                    self.turn_right()
                    self.fsm_state = FSM_ID.RIGHT
                    print("Turning right.")

    def estop(self):
        self.estop_nogui.settle_then_cut()
        self.lease.return_lease()

    def hijack_robot(self):
        self.lease = self.spot.get_lease(hijack=True)
        self.spot.power_on()
        self.spot.blocking_stand()

    def halt_robot(self, x_vel=0.0, ang_vel=0.0):
        self.spot.set_base_velocity(x_vel, 0.0, ang_vel, vel_time=UPDATE_PERIOD * 2)

    def move_forward(self):
        self.halt_robot(x_vel=LINEAR_VEL)

    def move_backwards(self):
        self.halt_robot(x_vel=-LINEAR_VEL)

    def turn_left(self):
        self.halt_robot(ang_vel=ANGULAR_VEL)

    def turn_right(self):
        self.halt_robot(ang_vel=-ANGULAR_VEL)


if __name__ == "__main__":
    SpotHeadlessTeleop()
