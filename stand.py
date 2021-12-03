from spot import Spot
import time
from bosdyn.client.robot_command import RobotCommandClient, blocking_stand


def run(spot):
    """Make Spot stand"""
    spot.robot.power_on(timeout_sec=20)
    assert spot.robot.is_powered_on(), "Robot power on failed."
    spot.loginfo("Robot powered on.")

    spot.loginfo("Commanding robot to stand...")
    command_client = spot.get_command_client()
    blocking_stand(command_client, timeout_sec=10)
    spot.loginfo("Robot standing.")
    time.sleep(3)

    spot.robot.power_off(cut_immediately=False, timeout_sec=20)
    assert not spot.robot.is_powered_on(), "Robot power off failed."
    spot.loginfo("Robot safely powered off.")


def main():
    spot = Spot()
    with spot.get_lease():
        run(spot)


if __name__ == "__main__":
    main()
