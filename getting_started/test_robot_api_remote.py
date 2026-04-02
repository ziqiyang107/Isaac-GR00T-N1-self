import os
import time
import json
import cv2
import numpy as np
import base64
import requests
import logging
from datetime import datetime
import socket

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_robot_api_remote.log'),
        logging.StreamHandler()
    ]
)

# 机器人API配置
ROBOT_IP = "192.168.0.211"  # 机器人IP地址
API_BASE_URL = f"http://{ROBOT_IP}:8000"  # FastAPI默认端口

def setup_directories():
    """创建必要的目录"""
    directories = ['remote_images', 'remote_data', 'test_results']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logging.info(f"创建目录: {directory}")

def test_connection():
    """测试API连接"""
    logging.info("开始测试API连接...")
    try:
        # 测试网络连接
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)  # 设置超时时间为5秒
        result = sock.connect_ex((ROBOT_IP, 8000))
        sock.close()
        
        if result != 0:
            logging.error(f"无法连接到机器人IP: {ROBOT_IP}:8000")
            logging.error("请检查：")
            logging.error("1. 机器人API服务是否已启动")
            logging.error("2. 网络连接是否正常")
            logging.error("3. 防火墙设置是否正确")
            raise ConnectionError("网络连接失败")
        
        # 测试API连接
        response = requests.get(f"{API_BASE_URL}/", timeout=5)
        if response.status_code != 200:
            logging.error(f"API连接失败，状态码: {response.status_code}")
            logging.error(f"响应内容: {response.text}")
            raise ConnectionError(f"API连接失败，状态码: {response.status_code}")
        
        logging.info("API连接测试通过")
        logging.info(f"服务器响应: {response.json()}")
    except requests.exceptions.ConnectionError as e:
        logging.error(f"连接错误: {str(e)}")
        logging.error("可能的原因：")
        logging.error("1. 机器人API服务未启动")
        logging.error("2. 网络连接问题")
        logging.error("3. 防火墙阻止了连接")
        raise
    except requests.exceptions.Timeout:
        logging.error("连接超时")
        logging.error("请检查网络连接和服务器响应时间")
        raise
    except Exception as e:
        logging.error(f"测试连接时发生错误: {str(e)}")
        raise

def test_get_observation():
    """测试获取观测数据"""
    logging.info("开始测试获取观测数据...")
    try:
        response = requests.get(f"{API_BASE_URL}/observation")
        assert response.status_code == 200, "获取观测数据失败"
        
        data = response.json()
        print('='*100)
        print(type(data), data.keys(), type(data['rgb_image']))
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
        
        logging.info(f"观测数据已保存到: {data_path}")
        logging.info(f"图像已保存到: {image_path}")
        logging.info("获取观测数据测试通过")
        
    except Exception as e:
        logging.error(f"获取观测数据时发生错误: {str(e)}")
        raise

def test_execute_action():
    """测试执行动作"""
    logging.info("开始测试执行动作...")
    try:
        left_arm_action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1]
        right_arm_action = [0.1, 0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 3]
        # # 测试左臂动作
        # response = requests.post(
        #     f"{API_BASE_URL}/action",
        #     json={"actions": {"left_arm": left_arm_action}}
        # )
        # assert response.status_code == 200, "左臂动作执行失败"
        
        # # 测试右臂动作
        # response = requests.post(
        #     f"{API_BASE_URL}/action",
        #     json={"actions": {"right_arm": right_arm_action}}
        # )
        # assert response.status_code == 200, "右臂动作执行失败"
        
        # 测试双臂动作
        response = requests.post(
            f"{API_BASE_URL}/action",
            json={
                "actions": {
                    "left_arm": left_arm_action,
                    "right_arm": [0.0, 0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 3]
                }
            }
        )
        assert response.status_code == 200, "双臂动作执行失败"
        time.sleep(2)
        response = requests.post(
            f"{API_BASE_URL}/action",
            json={
                "actions": {
                    "left_arm": left_arm_action,
                    "right_arm": [0.1, 0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 3]
                }
            }
        )
        assert response.status_code == 200, "双臂动作执行失败"
        time.sleep(2)
        response = requests.post(
            f"{API_BASE_URL}/action",
            json={
                "actions": {
                    "left_arm": left_arm_action,
                    "right_arm": [0.2, 0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 3]
                }
            }
        )
        assert response.status_code == 200, "双臂动作执行失败"

        
        # 保存动作数据
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        data_path = f"remote_data/action_{timestamp}.json"
        with open(data_path, 'w') as f:
            json.dump({
                "timestamp": timestamp,
                "left_arm": left_arm_action,
                "right_arm": right_arm_action
            }, f, indent=4)
        
        logging.info(f"动作数据已保存到: {data_path}")
        logging.info("执行动作测试通过")
        
    except Exception as e:
        logging.error(f"执行动作时发生错误: {str(e)}")
        raise

def run_all_tests():
    """运行所有测试"""
    logging.info("开始运行远程API测试...")
    setup_directories()
    
    try:
        test_connection()
        test_get_observation()
        test_execute_action()
        logging.info("所有远程API测试通过！")
    except AssertionError as e:
        logging.error(f"测试失败: {str(e)}")
    except Exception as e:
        logging.error(f"测试过程中发生错误: {str(e)}")

if __name__ == "__main__":
    run_all_tests() 