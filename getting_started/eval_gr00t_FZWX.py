# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# SO100 Real Robot
import time
from contextlib import contextmanager
import sys
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.robot_devices.cameras.configs import OpenCVCameraConfig
from lerobot.common.robot_devices.motors.dynamixel import TorqueMode
from lerobot.common.robot_devices.robots.configs import So100RobotConfig
from lerobot.common.robot_devices.robots.utils import make_robot_from_config
from lerobot.common.robot_devices.utils import RobotDeviceAlreadyConnectedError

# NOTE:
# Sometimes we would like to abstract different env, or run this on a separate machine
# User can just move this single python class method gr00t/eval/service.py
# to their code or do the following line below
sys.path.append(os.path.expanduser("/home/Isaac-GR00T/gr00t/eval"))
from service import ExternalRobotInferenceClient

# Import tqdm for progress bar
from tqdm import tqdm
import requests
import base64
from datetime import datetime
import json

#################################################################################


class SO100Robot:
    def __init__(self, calibrate=False, enable_camera=False, camera_index=9):
        self.config = So100RobotConfig()
        self.calibrate = calibrate
        self.enable_camera = enable_camera
        self.camera_index = camera_index
        if not enable_camera:
            self.config.cameras = {}
        else:
            self.config.cameras = {"webcam": OpenCVCameraConfig(camera_index, 30, 640, 480, "bgr")}
        self.config.leader_arms = {}

        # remove the .cache/calibration/so100 folder
        if self.calibrate:
            import os
            import shutil

            calibration_folder = os.path.join(os.getcwd(), ".cache", "calibration", "so100")
            print("========> Deleting calibration_folder:", calibration_folder)
            if os.path.exists(calibration_folder):
                shutil.rmtree(calibration_folder)

        # Create the robot
        self.robot = make_robot_from_config(self.config)
        self.motor_bus = self.robot.follower_arms["main"]

    # @contextmanager 装饰器将一个生成器函数转换为上下文管理器。该生成器函数应该：
    # 执行进入上下文时需要的设置代码
    # 使用 yield 语句暂停执行并将控制权交给 with 块内的代码
    # 在 with 块完成后，继续执行清理代码
    @contextmanager
    def activate(self):
        try:
            self.connect()       # 设置：连接到机器人
            self.move_to_initial_pose()   # 设置：移动到初始位置
            yield                # 暂停并执行 with 块内的代码
        finally:
            self.disconnect()    # 清理：断开连接（无论 with 块是否成功）

    def connect(self):
        if self.robot.is_connected:
            raise RobotDeviceAlreadyConnectedError(
                "ManipulatorRobot is already connected. Do not run `robot.connect()` twice."
            )

        # Connect the arms
        self.motor_bus.connect()

        # We assume that at connection time, arms are in a rest position, and torque can
        # be safely disabled to run calibration and/or set robot preset configurations.
        self.motor_bus.write("Torque_Enable", TorqueMode.DISABLED.value)

        # Calibrate the robot
        self.robot.activate_calibration()

        self.set_so100_robot_preset()

        # Enable torque on all motors of the follower arms
        self.motor_bus.write("Torque_Enable", TorqueMode.ENABLED.value)
        print("robot present position:", self.motor_bus.read("Present_Position"))
        self.robot.is_connected = True

        self.camera = self.robot.cameras["webcam"] if self.enable_camera else None
        if self.camera is not None:
            self.camera.connect()
        print("================> SO100 Robot is fully connected =================")

    def set_so100_robot_preset(self):
        # Mode=0 for Position Control
        self.motor_bus.write("Mode", 0)
        # Set P_Coefficient to lower value to avoid shakiness (Default is 32)
        # self.motor_bus.write("P_Coefficient", 16)
        self.motor_bus.write("P_Coefficient", 10)
        # Set I_Coefficient and D_Coefficient to default value 0 and 32
        self.motor_bus.write("I_Coefficient", 0)
        self.motor_bus.write("D_Coefficient", 32)
        # Close the write lock so that Maximum_Acceleration gets written to EPROM address,
        # which is mandatory for Maximum_Acceleration to take effect after rebooting.
        self.motor_bus.write("Lock", 0)
        # Set Maximum_Acceleration to 254 to speedup acceleration and deceleration of
        # the motors. Note: this configuration is not in the official STS3215 Memory Table
        self.motor_bus.write("Maximum_Acceleration", 254)
        self.motor_bus.write("Acceleration", 254)

    def move_to_initial_pose(self):
        current_state = self.robot.capture_observation()["observation.state"]
        print("current_state", current_state)
        # print all keys of the observation
        print("observation keys:", self.robot.capture_observation().keys())

        current_state[0] = 90
        current_state[2] = 90
        current_state[3] = 90
        self.robot.send_action(current_state)
        time.sleep(2)

        current_state[4] = -70
        current_state[5] = 30
        current_state[1] = 90
        self.robot.send_action(current_state)
        time.sleep(2)

        print("----------------> SO100 Robot moved to initial pose")

    def go_home(self):
        # [ 88.0664, 156.7090, 135.6152,  83.7598, -89.1211,  16.5107]
        print("----------------> SO100 Robot moved to home pose")
        home_state = torch.tensor([88.0664, 156.7090, 135.6152, 83.7598, -89.1211, 16.5107])
        self.set_target_state(home_state)
        time.sleep(2)

    def get_observation(self):
        return self.robot.capture_observation()

    def get_current_state(self):
        return self.get_observation()["observation.state"].data.numpy()

    def get_current_img(self):
        img = self.get_observation()["observation.images.webcam"].data.numpy()
        # convert bgr to rgb
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def set_target_state(self, target_state: torch.Tensor):
        self.robot.send_action(target_state)

    def enable(self):
        self.motor_bus.write("Torque_Enable", TorqueMode.ENABLED.value)

    def disable(self):
        self.motor_bus.write("Torque_Enable", TorqueMode.DISABLED.value)

    def disconnect(self):
        self.disable()
        self.robot.disconnect()
        self.robot.is_connected = False
        print("================> SO100 Robot disconnected")

    def __del__(self):
        self.disconnect()

#################################################################################

# 机器人API配置
ROBOT_IP = "192.168.0.211"  # 机器人IP地址
API_BASE_URL = f"http://{ROBOT_IP}:8000"  # FastAPI默认端口

class FZWXRobot:
    def __init__(self, calibrate=False, enable_camera=False, camera_index=9):
        # self.config = So100RobotConfig()
        # self.calibrate = calibrate
        # self.enable_camera = enable_camera
        # self.camera_index = camera_index
        # if not enable_camera:
        #     self.config.cameras = {}
        # else:
        #     self.config.cameras = {"webcam": OpenCVCameraConfig(camera_index, 30, 640, 480, "bgr")}
        # self.config.leader_arms = {}

        # remove the .cache/calibration/so100 folder
        # if self.calibrate:
        #     import os
        #     import shutil

        #     calibration_folder = os.path.join(os.getcwd(), ".cache", "calibration", "so100")
        #     print("========> Deleting calibration_folder:", calibration_folder)
        #     if os.path.exists(calibration_folder):
        #         shutil.rmtree(calibration_folder)

        # Create the robot
        # self.robot = make_robot_from_config(self.config)
        # self.motor_bus = self.robot.follower_arms["main"]
        self.is_connected = False
        self.cam_is_connected = False

    # @contextmanager 装饰器将一个生成器函数转换为上下文管理器。该生成器函数应该：
    # 执行进入上下文时需要的设置代码
    # 使用 yield 语句暂停执行并将控制权交给 with 块内的代码
    # 在 with 块完成后，继续执行清理代码
    @contextmanager
    def activate(self):
        try:
            self.connect()       # 设置：连接到机器人
            self.move_to_initial_pose()   # 设置：移动到初始位置
            yield                # 暂停并执行 with 块内的代码
        finally:
            self.disconnect()    # 清理：断开连接（无论 with 块是否成功）

    def connect(self):
        if self.is_connected:
            print('机器人连接上了！！')
            # raise RobotDeviceAlreadyConnectedError(
            #     "ManipulatorRobot is already connected. Do not run `robot.connect()` twice."
            # )

        # Connect the arms
        # self.motor_bus.connect()

        # We assume that at connection time, arms are in a rest position, and torque can
        # be safely disabled to run calibration and/or set robot preset configurations.
        # self.motor_bus.write("Torque_Enable", TorqueMode.DISABLED.value)

        # Calibrate the robot
        # self.robot.activate_calibration()

        # self.set_so100_robot_preset()

        # Enable torque on all motors of the follower arms
        # self.motor_bus.write("Torque_Enable", TorqueMode.ENABLED.value)
        # print("robot present position:", self.motor_bus.read("Present_Position"))
        self.is_connected = True

        # check if camera connected:
        self.cam_is_connected = True
        # self.camera = self.robot.cameras["webcam"] if self.enable_camera else None
        # if self.camera is not None:
        #     self.camera.connect()
        print("================> SO100 Robot is fully connected =================")

    # 用于最大关节限制
    def set_so100_robot_preset(self):
        # Mode=0 for Position Control
        self.motor_bus.write("Mode", 0)
        # Set P_Coefficient to lower value to avoid shakiness (Default is 32)
        # self.motor_bus.write("P_Coefficient", 16)
        self.motor_bus.write("P_Coefficient", 10)
        # Set I_Coefficient and D_Coefficient to default value 0 and 32
        self.motor_bus.write("I_Coefficient", 0)
        self.motor_bus.write("D_Coefficient", 32)
        # Close the write lock so that Maximum_Acceleration gets written to EPROM address,
        # which is mandatory for Maximum_Acceleration to take effect after rebooting.
        self.motor_bus.write("Lock", 0)
        # Set Maximum_Acceleration to 254 to speedup acceleration and deceleration of
        # the motors. Note: this configuration is not in the official STS3215 Memory Table
        self.motor_bus.write("Maximum_Acceleration", 254)
        self.motor_bus.write("Acceleration", 254)

    def move_to_initial_pose(self):
        # current_state = self.capture_observation()["observation.state"]
        # print("current_state", current_state)
        # # print all keys of the observation
        # print("observation keys:", self.capture_observation().keys())

        print("Move to initial state: all 0 !!")
        init_state = [0.0]*16
        time.sleep(2)
        self.send_action(init_state)
        

        print("----------------> SO100 Robot moved to initial pose")

    # ziqi: this function is not used in real deployment
    def go_home(self):
        # [ 88.0664, 156.7090, 135.6152,  83.7598, -89.1211,  16.5107]
        print("----------------> SO100 Robot moved to home pose")
        home_state = torch.tensor([88.0664, 156.7090, 135.6152, 83.7598, -89.1211, 16.5107])
        self.set_target_state(home_state)
        time.sleep(2)
    
    def capture_observation(self):
        response = requests.get(f"{API_BASE_URL}/observation")
        data = response.json()

        assert "rgb_image" in data, "缺少图像数据"
        assert "state.left_arm" in data, "缺少左臂状态"
        assert "state.right_arm" in data, "缺少右臂状态"
        assert "state.left_gripper" in data, "缺少左夹爪状态"
        assert "state.right_gripper" in data, "缺少右夹爪状态"
        assert "timestamp" in data, "缺少时间戳"
        
        # 保存图像
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = f"remote_images/observation_{timestamp}.jpg"
        image_data = base64.b64decode(data["rgb_image"])
        with open(image_path, 'wb') as f:
            f.write(image_data)
        
        # 保存观测数据
        data_path = f"remote_data/observation_{timestamp}.json"
        with open(data_path, 'w') as f:
            json.dump(data, f, indent=4)
        
        return data
    
    def send_action(self, target_state):
        print('='*50)
        print(target_state)
        response = requests.post(
            f"{API_BASE_URL}/action",
            json={
                "actions": {
                    "left_arm": target_state[:8],
                    "right_arm": target_state[8:16]
                }
            }
        )
        assert response.status_code == 200, "双臂+gripper动作执行失败"

    def get_observation(self):
        return self.capture_observation()

    def get_current_state(self, modality_keys):
        data = self.get_observation()
        joint_state = []
        for k in modality_keys:
            # if k != 'rgb_image' and k != 'timestamp' and ('state.' + k) in data.keys():
            joint_state += data['state.' + k]
        return np.array(joint_state)

    def get_current_img(self):
        # 解码 base64 数据
        img = base64.b64decode(self.get_observation()["rgb_image"])
        # 将字节转换为 numpy 数组
        nparr = np.frombuffer(img, np.uint8)
        # 解码为图像
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        # convert bgr to rgb  OpenCV 默认使用 BGR 顺序，如果需要 RGB，需要转换
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # ziqi comment out
        cv2.imwrite('/home/Isaac-GR00T/getting_started/output_image.jpg', img)  # ziqi add
        ##################
        return img

    def set_target_state(self, target_state: torch.Tensor):
        self.send_action(target_state)

    # def enable(self):
    #     self.motor_bus.write("Torque_Enable", TorqueMode.ENABLED.value)

    # def disable(self):
    #     self.motor_bus.write("Torque_Enable", TorqueMode.DISABLED.value)

    def disconnect(self):
        # self.disable()
        # self.disconnect()
        self.is_connected = False
        print("================> FZWX Robot disconnected")

    def __del__(self):
        self.disconnect()

#################################################################################


class Gr00tRobotInferenceClient:
    def __init__(
        self,
        host="localhost",
        port=8088,
        language_instruction="Pick up the water bottle and place it in the basket.",
    ):
        self.language_instruction = language_instruction
        # 480, 640
        self.img_size = (480, 640)
        self.policy = ExternalRobotInferenceClient(host=host, port=port)

    def get_action(self, img, state):
        '''
        state: 这些state要串起来，然后在下面代码用np.newaxis升维
        '''
        obs_dict = {
            "video.webcam": img[np.newaxis, :, :, :],
            "state.left_arm": state[:7][np.newaxis, :].astype(np.float64),
            "state.left_gripper": state[7:8][np.newaxis, :].astype(np.float64),
            "state.right_arm": state[8:15][np.newaxis, :].astype(np.float64),
            "state.right_gripper": state[15:16][np.newaxis, :].astype(np.float64),
            "annotation.human.action.task_description": [self.language_instruction],
        }
        start_time = time.time()
        res = self.policy.get_action(obs_dict)
        print("Inference query time taken", time.time() - start_time)
        return res

    def sample_action(self):
        obs_dict = {
            "video.webcam": np.zeros((1, self.img_size[0], self.img_size[1], 3), dtype=np.uint8),
            "state.single_arm": np.zeros((1, 5)),
            "state.gripper": np.zeros((1, 1)),
            "annotation.human.action.task_description": [self.language_instruction],
        }
        return self.policy.get_action(obs_dict)


#################################################################################


def view_img(img, img2=None):
    """
    This is a matplotlib viewer since cv2.imshow can be flaky in lerobot env
    also able to overlay the image to ensure camera view is alligned to training settings
    """
    plt.imshow(img)
    if img2 is not None:
        plt.imshow(img2, alpha=0.5)
    plt.axis("off")
    plt.pause(0.001)  # Non-blocking show
    plt.clf()  # Clear the figure for the next frame


#################################################################################

if __name__ == "__main__":
    import argparse
    import os

    default_dataset_path = os.path.expanduser("~/datasets/so100_strawberry_grape")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_policy", action="store_true"
    )  # default is to playback the provided dataset
    parser.add_argument("--dataset_path", type=str, default=default_dataset_path)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8088)
    parser.add_argument("--action_horizon", type=int, default=12)
    parser.add_argument("--actions_to_execute", type=int, default=350)
    parser.add_argument("--camera_index", type=int, default=9)
    parser.add_argument("--lang_instruction", type=str, default="Pick up the water bottle and place it in the basket.")
    args = parser.parse_args()

    ACTIONS_TO_EXECUTE = args.actions_to_execute
    USE_POLICY = args.use_policy
    ACTION_HORIZON = (
        args.action_horizon
    )  # we will execute only some actions from the action_chunk of 16
    # ziqi：必须按照这个顺序写：
    MODALITY_KEYS = ["left_arm", "left_gripper", "right_arm", "right_gripper"]

    # ziqi: in our own demo, we always use use_policy
    if USE_POLICY:
        client = Gr00tRobotInferenceClient(
            host=args.host,
            port=args.port,
            language_instruction=args.lang_instruction,
        )

        robot = FZWXRobot(calibrate=False, enable_camera=True, camera_index=args.camera_index)
        with robot.activate():
            for i in tqdm(range(ACTIONS_TO_EXECUTE), desc="Executing actions"):
                img = robot.get_current_img()
                view_img(img)
                state = robot.get_current_state(MODALITY_KEYS)
                # time.sleep(2.00)
                action = client.get_action(img, state)
                start_time = time.time()
                for j in range(ACTION_HORIZON):
                    concat_action = np.concatenate(
                        [np.atleast_1d(action[f"action.{key}"][j]) for key in MODALITY_KEYS],
                        axis=0,
                    )
                    # ziqi watch out, why is this 6? b/c so100 has 6 actions and states?
                    assert concat_action.shape == (16,), concat_action.shape 
                    print(f'concat_action: {concat_action[11:12]}')
                    robot.set_target_state(list(concat_action.data))
                    time.sleep(0.05)

                    # get the realtime image
                    img = robot.get_current_img()
                    view_img(img)

                    # 0.05*16 = 0.8 seconds
                    print("executing action", j, "time taken", time.time() - start_time)
                print("Action chunk execution time taken", time.time() - start_time)
                time.sleep(0.1)  # ziqi add
    else:
        pass
# ziqi:
# python eval_gr00t_FZWX.py --use_policy --actions_to_execute 600 --action_horizon 12