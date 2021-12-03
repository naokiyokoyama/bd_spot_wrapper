# Copyright (c) 2021 Boston Dynamics, Inc.  All rights reserved.
#
# Downloading, reproducing, distributing or otherwise using the SDK Software
# is subject to the terms and conditions of the Boston Dynamics Software
# Development Kit License (20191101-BDSDK-SL).

""" Easy-to-use wrapper for properly controlling Spot """
import cv2
import os
import bosdyn.client
import bosdyn.client.lease
import bosdyn.client.util
from bosdyn import geometry
from bosdyn.client.robot_command import (RobotCommandClient, blocking_stand, block_until_arm_arrives)
from bosdyn.client import math_helpers
from bosdyn.client.image import ImageClient
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.api import (
    image_pb2,
    geometry_pb2,
    manipulation_api_pb2,
    trajectory_pb2,
    robot_command_pb2,
    arm_command_pb2,
    synchronized_command_pb2,
)
from bosdyn.client.robot_command import RobotCommandBuilder
from bosdyn.client.frame_helpers import (
    GRAV_ALIGNED_BODY_FRAME_NAME,
    get_vision_tform_body,
    HAND_FRAME_NAME,
)

from bosdyn.api import robot_command_pb2
from bosdyn.api import arm_command_pb2
from bosdyn.api import synchronized_command_pb2
import time
import numpy as np

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

HAND_RGB_UUID = "hand_color_image"
IMG_UUIDS = [HAND_RGB_UUID]


class Spot:
    def __init__(self, client_name_prefix):
        bosdyn.client.util.setup_logging()
        sdk = bosdyn.client.create_standard_sdk(client_name_prefix)
        robot = sdk.create_robot("192.168.80.3")
        robot.authenticate("admin", SPOT_ADMIN_PW)
        robot.time_sync.wait_for_sync()
        self.robot = robot
        self.command_client = None
        self.spot_lease = None

    def get_lease(self, hijack=False):
        # Make sure a lease for this client isn't already active
        assert self.spot_lease is None
        self.spot_lease = SpotLease(self, hijack=hijack)
        return self.spot_lease

    def is_estopped(self):
        return self.robot.is_estopped()

    def power_on(
        self, timeout_sec=20, service_name=RobotCommandClient.default_service_name
    ):
        self.robot.power_on(timeout_sec=timeout_sec)
        assert self.robot.is_powered_on(), "Robot power on failed."
        self.loginfo("Robot powered on.")
        # Just assign a command client by default when the robot is turned on
        self.get_command_client(service_name=service_name)

    def power_off(self, cut_immediately=False, timeout_sec=20):
        self.loginfo("Powering robot off...")
        self.robot.power_off(cut_immediately=cut_immediately, timeout_sec=timeout_sec)
        assert not self.robot.is_powered_on(), "Robot power off failed."
        self.loginfo("Robot safely powered off.")

    def blocking_stand(self, timeout_sec=10):
        self.loginfo("Commanding robot to stand (blocking)...")
        blocking_stand(self.command_client, timeout_sec=timeout_sec)
        self.loginfo("Robot standing.")

    def loginfo(self, *args, **kwargs):
        self.robot.logger.info(*args, **kwargs)

    def get_command_client(self, service_name=RobotCommandClient.default_service_name):
        self.command_client = self.robot.ensure_client(service_name)
        return self.command_client

    def open_gripper(self):
        """Does not block, be careful!"""
        gripper_command = RobotCommandBuilder.claw_gripper_open_command()
        self.command_client.robot_command(gripper_command)

    def move_gripper_to_point(self, point, rotation):
        """
        Moves EE to a point relative to body frame
        :param point: XYZ location
        :param rotation: Euler roll-pitch-yaw or WXYZ quaternion
        :return: cmd_id
        """
        if len(rotation) == 3:  # roll pitch yaw Euler angles
            roll, pitch, yaw = rotation
            quat = geometry.EulerZXY(yaw=yaw, roll=roll, pitch=pitch).to_quaternion()
        elif len(rotation) == 4:  # w, x, y, z quaternion
            w, x, y, z = rotation
            quat = math_helpers.Quat(w=w, x=x, y=y, z=z)
        else:
            raise RuntimeError(
                "rotation needs to have length 3 (euler) or 4 (quaternion),"
                f"got {len(rotation)}"
            )

        hand_pose = math_helpers.SE3Pose(*point, quat)
        hand_trajectory = trajectory_pb2.SE3Trajectory(
            points=[trajectory_pb2.SE3TrajectoryPoint(pose=hand_pose.to_proto())]
        )
        arm_cartesian_command = arm_command_pb2.ArmCartesianCommand.Request(
            pose_trajectory_in_task=hand_trajectory,
            root_frame_name=GRAV_ALIGNED_BODY_FRAME_NAME,
        )

        # Pack everything up in protos.
        arm_command = arm_command_pb2.ArmCommand.Request(
            arm_cartesian_command=arm_cartesian_command
        )
        synchronized_command = synchronized_command_pb2.SynchronizedCommand.Request(
            arm_command=arm_command
        )
        command = robot_command_pb2.RobotCommand(
            synchronized_command=synchronized_command
        )
        cmd_id = self.command_client.robot_command(command)

        return cmd_id

    def block_until_arm_arrives(self, cmd_id, timeout_sec=5):
        block_until_arm_arrives(self.command_client, cmd_id, timeout_sec=timeout_sec)

    def get_image_responses(self, sources):
        """Retrieve images from Spot's cameras

        :param sources: list containing camera uuids
        :return: list containing bosdyn image response objects
        """
        assert all(
            [src in IMG_UUIDS for src in sources]
        ), "An invalid camera uuid was provided!"
        image_client = self.robot.ensure_client(ImageClient.default_service_name)
        image_responses = image_client.get_image_from_sources(sources)
        return image_responses

    def grasp_point_in_image(self, image_response, pixel_xy=None):
        # If pixel location not provided, select the center pixel
        if pixel_xy is None:
            height = image_response.shot.image.rows
            width = image_response.shot.image.cols
            pixel_xy = [width // 2, height // 2]

        manipulation_api_client = self.robot.ensure_client(
            ManipulationApiClient.default_service_name
        )

        pick_vec = geometry_pb2.Vec2(x=pixel_xy[0], y=pixel_xy[1])
        grasp = manipulation_api_pb2.PickObjectInImage(
            pixel_xy=pick_vec,
            transforms_snapshot_for_camera=image_response.shot.transforms_snapshot,
            frame_name_image_sensor=image_response.shot.frame_name_image_sensor,
            camera_model=image_response.source.pinhole,
            walk_gaze_mode=3,  # PICK_NO_AUTO_WALK_OR_GAZE
        )

        # Ask the robot to pick up the object
        grasp_request = manipulation_api_pb2.ManipulationApiRequest(
            pick_object_in_image=grasp
        )
        # Send the request
        cmd_response = manipulation_api_client.manipulation_api_command(
            manipulation_api_request=grasp_request
        )

        # Get feedback from the robot
        while True:
            feedback_request = manipulation_api_pb2.ManipulationApiFeedbackRequest(
                manipulation_cmd_id=cmd_response.manipulation_cmd_id
            )

            # Send the request
            response = manipulation_api_client.manipulation_api_feedback_command(
                manipulation_api_feedback_request=feedback_request
            )

            print(
                "Current grasp_point_in_image state: ",
                manipulation_api_pb2.ManipulationFeedbackState.Name(
                    response.current_state
                ),
            )

            if response.current_state in [
                manipulation_api_pb2.MANIP_STATE_GRASP_SUCCEEDED,
                manipulation_api_pb2.MANIP_STATE_GRASP_FAILED,
            ]:
                break

            time.sleep(0.25)

class SpotLease:
    """
    A class that supports execution with Python's "with" statement for safe return of
    the lease upon exit. Grants control of the Spot's motor.
    """

    def __init__(self, spot, hijack=False):
        self.lease_client = spot.robot.ensure_client(
            bosdyn.client.lease.LeaseClient.default_service_name
        )
        if hijack:
            self.lease = self.lease_client.take()
        else:
            self.lease = self.lease_client.acquire()
        self.lease_keep_alive = bosdyn.client.lease.LeaseKeepAlive(self.lease_client)
        self.spot = spot

    def __enter__(self):
        return self.lease

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Exit the LeaseKeepAlive object
        self.lease_keep_alive.__exit__(exc_type, exc_val, exc_tb)
        # Return the lease
        self.lease_client.return_lease(self.lease)
        self.spot.loginfo("Returned the lease.")
        # Clear lease from Spot object
        self.spot.spot_lease = None


def image_response_to_cv2(image_response):
    if image_response.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_DEPTH_U16:
        dtype = np.uint16
    else:
        dtype = np.uint8
    img = np.fromstring(image_response.shot.image.data, dtype=dtype)
    if image_response.shot.image.format == image_pb2.Image.FORMAT_RAW:
        img = img.reshape(image_response.shot.image.rows, image_response.shot.image.cols)
    else:
        img = cv2.imdecode(img, -1)

    return img