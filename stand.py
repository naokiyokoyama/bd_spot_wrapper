from spot import Spot
import time

def run(spot):
    """Make Spot stand"""
    spot.power_on()
    spot.blocking_stand()

    # Wait 3 seconds to before powering down...
    time.sleep(3)
    spot.power_off(cut_immediately=False, timeout_sec=20)


def main():
    spot = Spot("BasicStandingClient")
    with spot.get_lease():
        run(spot)


if __name__ == "__main__":
    main()
