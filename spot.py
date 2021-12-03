# Copyright (c) 2021 Boston Dynamics, Inc.  All rights reserved.
#
# Downloading, reproducing, distributing or otherwise using the SDK Software
# is subject to the terms and conditions of the Boston Dynamics Software
# Development Kit License (20191101-BDSDK-SL).

""" Easy-to-use wrapper for properly controlling Spot """
import os
import bosdyn.client
import bosdyn.client.lease
import bosdyn.client.util
import bosdyn.geometry
from bosdyn.client.robot_command import RobotCommandClient

try:
    SPOT_ADMIN_PW = os.environ["SPOT_ADMIN_PW"]
except KeyError:
    raise RuntimeError(
        "\nSPOT_ADMIN_PW not found as an environment variable!\n"
        "Please run:\n"
        "echo 'export SPOT_ADMIN_PW=<YOUR_SPOT_ADMIN_PW>' >> ~/.bashrc\nor for MacOS,\n"
        "echo 'export SPOT_ADMIN_PW=<YOUR_SPOT_ADMIN_PW>' >> ~/.bash_profile\n"
        "Then:\nsource ~/.bashrc\nor\nsource ~/.bash_profile"
    )


class SpotLease:
    """
    A class that supports execution with Python's "with" statement for safe return of
    the lease upon exit.
    """

    def __init__(self, lease_client, spot):
        self.lease = lease_client.acquire()
        self.lease_client = lease_client
        self.lease_keep_alive = bosdyn.client.lease.LeaseKeepAlive(lease_client)
        self.spot = spot

    def __enter__(self):
        return self.lease

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Exit the LeaseKeepAlive object
        self.lease_keep_alive.__exit__(exc_type, exc_val, exc_tb)

        # Return the lease
        self.lease_client.return_lease(self.lease)

        # Clear lease from Spot object
        self.spot.spot_lease = None


class Spot:
    def __init__(self, client_name_prefix="default_client"):
        bosdyn.client.util.setup_logging()
        sdk = bosdyn.client.create_standard_sdk(client_name_prefix)
        robot = sdk.create_robot("192.168.80.3")
        robot.authenticate("admin", SPOT_ADMIN_PW)
        robot.time_sync.wait_for_sync()
        self.robot = robot
        self.spot_lease = None
        self.command_client = None

    def is_estopped(self):
        return self.robot.is_estopped()

    def get_lease(self):
        # Make sure a lease for this client isn't already active
        assert self.spot_lease is None
        lease_client = self.robot.ensure_client(
            bosdyn.client.lease.LeaseClient.default_service_name
        )
        self.spot_lease = SpotLease(lease_client, self)
        return self.spot_lease

    def loginfo(self, *args, **kwargs):
        self.robot.logger.info(*args, **kwargs)

    def get_command_client(self):
        self.command_client = self.robot.ensure_client(
            RobotCommandClient.default_service_name
        )
        return self.command_client

