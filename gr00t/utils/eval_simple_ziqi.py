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

import matplotlib.pyplot as plt
import numpy as np

from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.model.policy import BasePolicy
from gr00t.eval.robot import RobotInferenceClient

import requests
import time
import base64
import cv2

# # numpy print precision settings 3, dont use exponential notation
# np.set_printoptions(precision=3, suppress=True)


# def download_from_hg(repo_id: str, repo_type: str) -> str:
#     """
#     Download the model/dataset from the hugging face hub.
#     return the path to the downloaded
#     """
#     from huggingface_hub import snapshot_download

#     repo_path = snapshot_download(repo_id, repo_type=repo_type)
#     return repo_path

# 机器人API配置
ROBOT_IP = "192.168.0.211"  # 机器人IP地址
API_BASE_URL = f"http://{ROBOT_IP}:8000"  # FastAPI默认端口
# ziqi add:
def send_action(target_state=[0.0]*16, fix_gripper=True):
        print('='*50)
        print('target_state: ', target_state)
        if fix_gripper:
            target_state[7] = 5.0
            target_state[15] = 5.0

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

def calc_mse_for_single_trajectory(
    policy: BasePolicy,
    modality_keys: list,
    steps=300,
    action_horizon=16,
    plot=False,
):
    state_joints_across_time = []
    gt_action_joints_across_time = []
    pred_action_joints_across_time = []
    
    for step_count in range(steps):
        response = requests.get(f"{API_BASE_URL}/observation")
        assert response.status_code == 200, "获取观测数据失败"
        response = response.json()

        ############## 头顶摄像头：
        img = base64.b64decode(response["rgb_image"])
        # 将字节转换为 numpy 数组
        nparr = np.frombuffer(img, np.uint8)
        # 解码为图像
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        # convert bgr to rgb  OpenCV 默认使用 BGR 顺序，如果需要 RGB，需要转换
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite('/home/Isaac-GR00T/getting_started/output_image.jpg', img)  # ziqi add

        ############## 右臂摄像头：
        right_img = base64.b64decode(response["right_image"])
        # 将字节转换为 numpy 数组
        right_nparr = np.frombuffer(right_img, np.uint8)
        # 解码为图像
        right_img = cv2.imdecode(right_nparr, cv2.IMREAD_COLOR)
        # convert bgr to rgb  OpenCV 默认使用 BGR 顺序，如果需要 RGB，需要转换
        right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)
        cv2.imwrite('/home/Isaac-GR00T/getting_started/output_right_image.jpg', right_img)  # ziqi add



        data_point = dict()
        data_point['video.webcam'] = img[np.newaxis, :, :, :]
        data_point['video.cam_right_wrist'] = right_img[np.newaxis, :, :, :]
        data_point["state.left_arm"] = np.array(response['state.left_arm'])[np.newaxis, :].astype(np.float64)
        data_point["state.left_gripper"] = np.array(response['state.left_gripper'])[np.newaxis, :].astype(np.float64)
        data_point["state.right_arm"] = np.array(response['state.right_arm'])[np.newaxis, :].astype(np.float64)
        data_point["state.right_gripper"] = np.array(response['state.right_gripper'])[np.newaxis, :].astype(np.float64)
        data_point["annotation.human.task_description"] = ['Pick up the water bottle and place it in the basket.']

        # NOTE this is to get all modality keys concatenated
        # concat_state = data_point[f"state.{modality_keys[0]}"][0]
        # concat_gt_action = data_point[f"action.{modality_keys[0]}"][0]
        # concat_state = np.concatenate(
        #     [data_point[f"state.{key}"][0] for key in modality_keys], axis=0
        # )
        # concat_gt_action = np.concatenate(
        #     [data_point[f"action.{key}"][0] for key in modality_keys], axis=0
        # )

        # state_joints_across_time.append(concat_state)
        # gt_action_joints_across_time.append(concat_gt_action)

        if step_count % action_horizon == 0:
            print('='*100)
            print("inferencing at step: ", step_count)
            print(f'data_point at step {step_count}: ', data_point, type(data_point))
            action_chunk = policy.get_action(data_point)
            for j in range(action_horizon):
                # NOTE: concat_pred_action = action[f"action.{modality_keys[0]}"][j]
                # the np.atleast_1d is to ensure the action is a 1D array, handle where single value is returned
                concat_pred_action = np.concatenate(
                    [np.atleast_1d(action_chunk[f"action.{key}"][j]) for key in modality_keys],
                    axis=0,
                )
                assert concat_pred_action.shape == (16,), concat_pred_action.shape   # 16个关节
                ######### ziqi add
                send_action(list(concat_pred_action.data), fix_gripper=False)
                time.sleep(0.5)
                print(f"joint 11 pred_action at step_count {step_count}, iter {j}: ", concat_pred_action[11:12])
                #########
                pred_action_joints_across_time.append(concat_pred_action)

    # plot the joints
    # state_joints_across_time = np.array(state_joints_across_time)
    # gt_action_joints_across_time = np.array(gt_action_joints_across_time)
    # pred_action_joints_across_time = np.array(pred_action_joints_across_time)[:steps]
    # assert (
    #     state_joints_across_time.shape
    #     == gt_action_joints_across_time.shape
    #     == pred_action_joints_across_time.shape
    # )

    # calc MSE across time
    # mse = np.mean((gt_action_joints_across_time - pred_action_joints_across_time) ** 2)
    # print("Unnormalized Action MSE across single traj:", mse)

    # num_of_joints = state_joints_across_time.shape[1]

    if plot:
        fig, axes = plt.subplots(nrows=num_of_joints, ncols=1, figsize=(8, 4 * num_of_joints))

        # Add a global title showing the modality keys
        fig.suptitle(
            f"Trajectory {traj_id} - Modalities: {', '.join(modality_keys)}",
            fontsize=16,
            color="blue",
        )

        for i, ax in enumerate(axes):
            ax.plot(state_joints_across_time[:, i], label="state joints")
            ax.plot(gt_action_joints_across_time[:, i], label="gt action joints")
            ax.plot(pred_action_joints_across_time[:, i], label="pred action joints")

            # put a dot every ACTION_HORIZON
            for j in range(0, steps, action_horizon):
                if j == 0:
                    ax.plot(j, gt_action_joints_across_time[j, i], "ro", label="inference point")
                else:
                    ax.plot(j, gt_action_joints_across_time[j, i], "ro")

            ax.set_title(f"Joint {i}")
            ax.legend()

        plt.tight_layout()
        ################## was:
        # plt.show()
        ################## ziqi add:
        plt.savefig("/home/tmp/output.png")
        # 可选：关闭当前图形，防止内存泄露
        plt.close()

    ################################# ziqi add: 直接执行真实指令：
    # print('='*100)
    # print('直接执行真实指令')
    # for i in gt_action_joints_across_time[::-1]:
    #     print(i.shape)
    #     print('右臂joint 11:')
    #     print(list(i.data)[11:12])
    #     send_action(list(i.data))
    #     time.sleep(0.1)
    # print('='*100)
    # print('真实指令shape：')
    # print(gt_action_joints_across_time.shape, type(gt_action_joints_across_time))
    ################################
    # return mse


if __name__ == '__main__':
    policy: BasePolicy = RobotInferenceClient(host='localhost', port=8088)
    modality_keys = ["left_arm", "left_gripper", "right_arm", "right_gripper"]
    print('='*100)
    print('上来先默认0位')
    send_action()  # 上来先默认0位
    calc_mse_for_single_trajectory(
        policy,
        modality_keys,
        steps=800,
        action_horizon=16,
        plot=False
    )
