from spot import Spot, HAND_RGB_UUID
from pynput import keyboard
import numpy as np

MOVE_INCREMENT = 0.02
TILT_INCREMENT = 5.0

KEY2MOVEMENT = {
    'w': np.array([0.0, 0.0, MOVE_INCREMENT, 0.0, 0.0, 0.0]),  # move up
    's': np.array([0.0, 0.0, -MOVE_INCREMENT, 0.0, 0.0, 0.0]),  # move down
    'a': np.array([0.0, MOVE_INCREMENT, 0.0, 0.0, 0.0, 0.0]),  # move left
    'd': np.array([0.0, -MOVE_INCREMENT, 0.0, 0.0, 0.0, 0.0]),  # move right
    'q': np.array([MOVE_INCREMENT, 0.0, 0.0, 0.0, 0.0, 0.0]),  # move forward
    'e': np.array([-MOVE_INCREMENT, 0.0, 0.0, 0.0, 0.0, 0.0]),  # move backward
    'i': np.deg2rad([0.0, 0.0, 0.0, 0.0, -TILT_INCREMENT, 0.0]),  # pitch up
    'k': np.deg2rad([0.0, 0.0, 0.0, 0.0, TILT_INCREMENT, 0.0]),  # pitch down
    'j': np.deg2rad([0.0, 0.0, 0.0, 0.0, 0.0, TILT_INCREMENT]),  # pan left
    'l': np.deg2rad([0.0, 0.0, 0.0, 0.0, 0.0, -TILT_INCREMENT]),  # pan right
}


def move_to_initial(spot):
    point = np.array([0.75, 0.0, 0.65])
    rpy = np.deg2rad([0.0, 60.0, 0.0])
    cmd_id = spot.move_gripper_to_point(point, rpy)
    spot.block_until_arm_arrives(cmd_id, timeout_sec=0.8)

    return point, rpy

def main(spot: Spot):
    """Uses IK to move the arm by setting hand poses"""
    spot.power_on()
    spot.blocking_stand()

    # Open the gripper
    spot.open_gripper()

    # Move arm to initial configuration
    point, rpy = move_to_initial(spot)
    try:
        while True:
            point_rpy = np.concatenate([point, rpy])

            with keyboard.Events() as events:
                # Wait for as many seconds as possible
                try:
                    pressed_key = events.get(int(1e6)).key.char
                except AttributeError:
                    # User might press Cmd or something not a char
                    pressed_key = None

            if pressed_key == 'z':
                # Quit
                break
            elif pressed_key in KEY2MOVEMENT:
                # Move
                point_rpy += KEY2MOVEMENT[pressed_key]
                point = point_rpy[:3]
                rpy = point_rpy[-3:]
                cmd_id = spot.move_gripper_to_point(point, rpy)
                spot.block_until_arm_arrives(cmd_id, timeout_sec=0.5)
            elif pressed_key == 'g':
                # Grab whatever object is at the center of hand RGB camera image
                image_responses = spot.get_image_responses([HAND_RGB_UUID])
                hand_image_response = image_responses[0]  # only expecting one image
                spot.grasp_point_in_image(hand_image_response)

                # Move arm to initial configuration
                point, rpy = move_to_initial(spot)
            elif pressed_key == 'r':
                # Open gripper
                spot.open_gripper()
    finally:
        spot.power_off()


if __name__ == "__main__":
    spot = Spot("ArmKeyboardTeleop")
    with spot.get_lease() as lease:
        main(spot)
