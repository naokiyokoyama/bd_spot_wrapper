"""
Controlling the robot using a presentation tool ("pt").
Up and down will start moving the robot forward or backwards if in linear mode.
In angular mode, it will turn left or right (respectively).
Tab key will switch between linear and angular mode.
If the same key is pressed again while the robot is moving, it will stop.
"""
import numpy as np
from bosdyn.client.estop import EstopClient

from spot_wrapper.estop import EstopNoGui
from spot_wrapper.spot import Spot
from spot_wrapper.utils.headless import KeyboardListener

UPDATE_PERIOD = 0.2
LINEAR_VEL = 1.0
ANGULAR_VEL = np.deg2rad(50)


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
    UP = "103"
    DOWN = "108"
    TAB = "15"
    ENTER = "28"


class SpotHeadlessTeleop(KeyboardListener):
    def __init__(self):
        super().__init__()
        self.mode = "linear"
        self.fsm_state = FSM_ID.ESTOP
        self.spot = Spot("HeadlessTeleop")
        self.lease = None
        estop_client = self.spot.robot.ensure_client(EstopClient.default_service_name)
        self.estop_nogui = EstopNoGui(estop_client, 5, "Estop NoGUI")

    def process_pressed_key(self, pressed_key):
        if pressed_key is None:
            return

        if pressed_key == KEY_ID.ENTER and self.fsm_state != FSM_ID.ESTOP:
            self.estop()
            self.fsm_state = FSM_ID.ESTOP
        elif pressed_key == KEY_ID.UP and self.fsm_state == FSM_ID.ESTOP:
            self.hijack_robot()
            self.fsm_state = FSM_ID.IDLE
        elif self.fsm_state == FSM_ID.IDLE:
            if pressed_key == KEY_ID.TAB:
                self.mode = "angular" if self.mode == "linear" else "linear"
            elif (
                pressed_key == KEY_ID.UP
                and self.fsm_state in [FSM_ID.FORWARD, FSM_ID.LEFT]
            ) or (
                pressed_key == KEY_ID.DOWN
                and self.fsm_state in [FSM_ID.BACK, FSM_ID.RIGHT]
            ):
                self.halt_robot()
                self.fsm_state = FSM_ID.IDLE
            elif pressed_key == KEY_ID.UP:
                if self.mode == "linear":
                    print("Linear up")
                    self.move_forward()
                    self.fsm_state = FSM_ID.FORWARD
                else:
                    print("Angular left")
                    self.turn_left()
                    self.fsm_state = FSM_ID.LEFT
            elif pressed_key == KEY_ID.DOWN:
                if self.mode == "linear":
                    print("Linear down")
                    self.move_backwards()
                    self.fsm_state = FSM_ID.BACK
                else:
                    print("Angular right")
                    self.turn_right()
                    self.fsm_state = FSM_ID.RIGHT

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
