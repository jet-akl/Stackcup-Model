from model.baxter.base_move_baxter_lock import BaseMoveBaxter
from model.baxter.base_xbox import BaseXbox
import os
import cv2
import rospy
import math
import time
import threading
from typing import Dict, List, Optional, Tuple

import numpy as np
import pygame
import pyrealsense2 as rs
from geometry_msgs.msg import PoseStamped


class D455Capture:
    def __init__(self, width=1280, height=720, fps=30):
        # D455 使用独立后台线程持续缓存最新一组彩色图和深度图。
        self.width = width
        self.height = height
        self.fps = fps

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        self.align = rs.align(rs.stream.color)

        self._lock = threading.Lock()
        self._latest = None
        self._running = False
        self._thread = None

    def start(self):
        if self._running:
            return
        self.pipeline.start(self.config)
        self._running = True
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    def _reader_loop(self):
        while self._running:
            try:
                # 对齐深度到彩色流，保证后续保存的数据在同一视角下可直接使用。
                frames = self.pipeline.wait_for_frames()
                frames = self.align.process(frames)

                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                if not color_frame or not depth_frame:
                    continue

                color_image = np.asanyarray(color_frame.get_data()).copy()
                depth_image = np.asanyarray(depth_frame.get_data()).copy()
                timestamp = color_frame.get_timestamp() / 1000.0

                with self._lock:
                    self._latest = {
                        "rs_image": color_image,
                        "rs_depth": depth_image,
                        "timestamp": timestamp,
                    }
            except Exception as exc:
                if self._running:
                    rospy.logwarn(f"D455 capture failed: {exc}")
                time.sleep(0.05)

    def get_latest(self):
        with self._lock:
            if self._latest is None:
                return None
            latest = {}
            for key, value in self._latest.items():
                latest[key] = value.copy() if isinstance(value, np.ndarray) else value
            return latest

    def stop(self):
        if not self._running:
            return
        self._running = False
        try:
            self.pipeline.stop()
        except Exception:
            pass
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None

class XboxSamplerContinue(BaseXbox):
    def __init__(
            self,
            joy_id=0,
            axis_deadzone=0.3,
            fps=15,
            timeout=0.3,
            lr_peak = 0.004,
            adaptive_lr = False,
            restrict_lr = False,
            data_root_path = './data_path',
            use_realsense = True,
            start_position_noise_type = 'Uniform',
            start_position_restrict_lr = 0.02,
            sync_tolerance = 0.8,
            rs_width = 1280,
            rs_height = 720,
            rs_fps = 30,
            show_preview = False,
            preview_scale = 0.5,
            startPosition = [0.6662041027959035, 0.008202369321740299, 0.3464959378579765],
            endPosition = [0.6637042176202286, 0.5231769257932641, 0.13266077885413383],
            ):
        super().__init__(joy_id, axis_deadzone, fps)

        self.timeout = timeout
        self.lr_peak = lr_peak
        self.lr = lr_peak
        self.adaptive_lr = adaptive_lr
        self.restrict_lr = restrict_lr
        self.data_root_path = data_root_path
        self.use_realsense = use_realsense
        self.start_position_noise_type = start_position_noise_type
        self.sync_tolerance = sync_tolerance
        self.rs_width = rs_width
        self.rs_height = rs_height
        self.rs_fps = rs_fps
        self.show_preview = show_preview
        self.preview_scale = preview_scale
        self.preview_window_name = 'save_preview'
        self.startPosition_base = np.array(startPosition, dtype=float)
        self.startPosition_restrict_lr = start_position_restrict_lr
        self.startPosition = self._sample_start_position()
        self.endPosition = endPosition

        # baxter
        self.baxter = BaseMoveBaxter(self.timeout)
        self.baxter.init_camera()
        # 只有开启开关时才启动 D455 采集。
        self.rs_capture = D455Capture(self.rs_width, self.rs_height, self.rs_fps) if self.use_realsense else None

        # dir iter
        self.dir_iter = 0
        self.dir_iter_path = os.path.join(self.data_root_path,'iter.txt')
        with open(self.dir_iter_path, 'r') as f:
            for line in f:
                line = line.strip()
                self.dir_iter = int(line)
                print('read dir_iter',self.dir_iter)
        
        # data iter
        self.data_iter = 0
        self.prev_fake_state = None
        self.prev_fake_action = None
        self.prev_fake_joint_action = None

    def _sample_start_position(self):
        if self.start_position_noise_type == 'Gaussian':
            startPosition_std = self.startPosition_restrict_lr / 2.0
            noise = np.random.normal(0.0, startPosition_std, size=3)
        elif self.start_position_noise_type == 'Uniform':
            noise = np.random.uniform(-self.startPosition_restrict_lr, self.startPosition_restrict_lr, size=3)
        else:
            noise = np.zeros(3)
        noise = np.clip(noise, -self.startPosition_restrict_lr, self.startPosition_restrict_lr)
        print('noise',noise)
        return (self.startPosition_base + noise).tolist()

    def _joint_vector(self, joint_item):
        return [
            joint_item['left_e0'],
            joint_item['left_e1'],
            joint_item['left_s0'],
            joint_item['left_s1'],
            joint_item['left_w0'],
            joint_item['left_w1'],
            joint_item['left_w2'],
        ]

    def _write_vector(self, file_path, values, type='w'):
        with open(file_path, type) as f:
            f.write(' '.join(str(value) for value in values))

    def _resize_for_preview(self, image):
        if image is None:
            return None
        scale = float(self.preview_scale)
        if scale <= 0 or abs(scale - 1.0) < 1e-6:
            return image
        width = max(1, int(image.shape[1] * scale))
        height = max(1, int(image.shape[0] * scale))
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    def _make_preview_canvas(self, observation):
        image = observation['image']
        preview_image = self._resize_for_preview(image)
        if preview_image is None:
            return None

        if not self.use_realsense or observation['rs_image'] is None:
            return preview_image

        rs_image = self._resize_for_preview(observation['rs_image'])
        if rs_image is None:
            return preview_image

        if rs_image.shape[:2] != preview_image.shape[:2]:
            rs_image = cv2.resize(rs_image, (preview_image.shape[1], preview_image.shape[0]), interpolation=cv2.INTER_AREA)

        depth = observation['rs_depth']
        if depth is not None:
            if depth.dtype != np.uint16:
                depth = depth.astype(np.uint16)
            depth_vis = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
            depth_vis = depth_vis.astype(np.uint8)
            depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
            depth_vis = self._resize_for_preview(depth_vis)
            if depth_vis.shape[:2] != preview_image.shape[:2]:
                depth_vis = cv2.resize(depth_vis, (preview_image.shape[1], preview_image.shape[0]), interpolation=cv2.INTER_AREA)
            top_row = np.hstack([preview_image, rs_image])
            bottom_row = np.hstack([depth_vis, np.zeros_like(depth_vis)])
            return np.vstack([top_row, bottom_row])
        return np.hstack([preview_image, rs_image])

    def _show_preview(self, observation):
        if not self.show_preview:
            return

        preview = self._make_preview_canvas(observation)
        if preview is None:
            return
        
        cv2.imshow(self.preview_window_name, preview)
        cv2.waitKey(1)

    def _get_synced_observation(self, timeout=0.3):
        # 这里用 Baxter 左相机时间戳与 D455 时间戳做近似对齐。
        deadline = time.time() + timeout
        while time.time() < deadline:
            baxter_image = self.baxter.left_latest_image
            baxter_stamp = self.baxter.left_timestamp
            rs_bundle = self.rs_capture.get_latest() if self.rs_capture is not None else None

            if baxter_image is None or baxter_stamp is None:
                time.sleep(0.0001)
                continue

            if self.use_realsense:
                if rs_bundle is None:
                    time.sleep(0.0001)
                    continue
                # 时间戳差超过阈值就继续等，避免把不同步的数据写入同一条样本。
                stamp_gap = abs(float(baxter_stamp) - float(rs_bundle['timestamp']))
                if stamp_gap > self.sync_tolerance:
                    time.sleep(0.0001)
                    continue
                return {
                    'image': np.copy(baxter_image),
                    'image_stamp': float(baxter_stamp),
                    'rs_image': rs_bundle['rs_image'],
                    'rs_depth': rs_bundle['rs_depth'],
                    'rs_stamp': float(rs_bundle['timestamp']),
                    'stamp_gap': stamp_gap,
                }
            return {
                'image': np.copy(baxter_image),
                'image_stamp': float(baxter_stamp),
                'rs_image': None,
                'rs_depth': None,
                'rs_stamp': None,
                'stamp_gap': None,
            }

        # raise RuntimeError("failed to get synchronized observation within timeout")

    def _save_data(self, observation, fake_state, fake_action, fake_joint_state, fake_joint_action, fake_pure_action):
        sample_root = os.path.join(self.data_root_path, f"{self.dir_iter:0{3}d}")
        sample_name = f"{self.data_iter:0{3}d}"

        # image 代表 Baxter 左相机图像；rs_image / rs_depth 只有在开关打开时才会保存。
        cv2.imwrite(os.path.join(sample_root, 'image', f'{sample_name}.png'), observation['image'])
        if self.use_realsense and observation['rs_image'] is not None and observation['rs_depth'] is not None:
            cv2.imwrite(os.path.join(sample_root, 'rs_image', f'{sample_name}.png'), observation['rs_image'])

            rs_depth = observation['rs_depth']
            if rs_depth.dtype != np.uint16:
                rs_depth = rs_depth.astype(np.uint16)
            cv2.imwrite(os.path.join(sample_root, 'rs_depth', f'{sample_name}.png'), rs_depth)

        self._write_vector(os.path.join(sample_root, 'fake_state', f'{sample_name}.txt'), fake_state)
        self._write_vector(os.path.join(sample_root, 'fake_action', f'{sample_name}.txt'), fake_action)
        self._write_vector(os.path.join(sample_root, 'fake_pure_action', f'{sample_name}.txt'), fake_pure_action)
        self._write_vector(os.path.join(sample_root, 'fake_joint_state', f'{sample_name}.txt'), fake_joint_state)
        self._write_vector(os.path.join(sample_root, 'fake_joint_action', f'{sample_name}.txt'), fake_joint_action)

        with open(os.path.join(sample_root, 'timestamp', f'{sample_name}.txt'), 'w') as f:
            # 保存用于检查同步性的时间戳信息。
            f.write(f"baxter_image_stamp {observation['image_stamp']}\n")
            if self.use_realsense:
                f.write(f"rs_stamp {observation['rs_stamp']}\n")
                f.write(f"stamp_gap {observation['stamp_gap']}\n")

    def _reset_epoch_state(self):
        # fake_state / fake_joint_state 保存的是“上一次”的目标状态，和当前动作形成监督对。
        self.prev_fake_state = list(self.startPosition)
        self.prev_fake_action = list(self.startPosition)
        joint_state = self.baxter.get_joint_angles('left')
        self.prev_fake_joint_action = self._joint_vector(joint_state)

    def sampling_pipline(self):
        if self.joy is None:
            raise ValueError("joystick has not be inited.Please use 'connect' first.")
        self.connect()
        if self.rs_capture is not None:
            self.rs_capture.start()
        # clock = pygame.time.Clock()

        # init
        # 起始位姿加入随机噪声，避免每次采样都从完全相同的姿态开始。
        self.startPosition = self._sample_start_position()
        print('[INFO] noisy start position:', self.startPosition)
        if not self.baxter.ik_move_limb('left', self.startPosition, timeout=5.0):
            if self.rs_capture is not None:
                self.rs_capture.stop()
            raise RuntimeError('failed to move baxter to the start position')
        rospy.sleep(0.2)
        self.baxter.left_gripper.open()
        rospy.sleep(0.2)
        self.init_path()
        self._reset_epoch_state()

        try:
            while True:
                # #cal_fps
                #self.start_time = time.time()
                pygame.event.pump()

                ## out detect
                if self.joy.get_button(8)==1:
                    if self.rs_capture is not None:
                        self.rs_capture.stop()
                    self.end()
                    print("\nGet Home Button: Exit.")
                    break
                
                ## epoch finish
                ## B夹取
                if self.joy.get_button(1)==1:
                    ## update iter and init path
                    self.data_iter = 0
                    self.dir_iter += 1
                    with open(self.dir_iter_path, 'w') as f:
                        f.write(f'{self.dir_iter}')
                    self.init_path()

                    # 重新回到起点时也采一次噪声，避免每个 epoch 的起始位姿完全一致。
                    self.startPosition = self._sample_start_position()
                    
                    ## 恢复状态
                    self.baxter.left_gripper.close()
                    rospy.sleep(0.2)
                    self.baxter.ik_move_limb('left', self.endPosition,timeout=5.0)
                    rospy.sleep(0.2)
                    self.baxter.left_gripper.open()
                    rospy.sleep(0.2)
                    self.baxter.ik_move_limb('left', self.startPosition,timeout=5.0)
                    rospy.sleep(0.2)
                    self._reset_epoch_state()

                    print("[INFO] Finish epoch")
                    # clock.tick(self.fps)
                    continue
                
                ## get position
                # 读取手柄轴输入，换算成当前步的末端偏移。
                xbox_position = self.moving_policy_axis()
                if not xbox_position:
                    # clock.tick(self.fps)
                    continue

                # 等待 Baxter 图像和可选 D455 数据进入同一时间窗。
                observation = self._get_synced_observation()
                if observation is None:
                    print("[WARN] Could not get synchronized observation")
                    # clock.tick(self.fps)
                    continue

                ## update lr
                current_state, _ = self.baxter.get_pose('left')
                if self.adaptive_lr:
                    # 根据当前末端高度自适应调整步长。
                    # current_state, _ = self.baxter.get_pose('left')
                    self.adaptive_lr_by_depth(current_state[2])
                
                ## cal moving dis
                # fake_* 表示“采样时用于监督的目标状态”，不是机械臂真实执行后的状态。
                fake_pure_action = self.cal_moving_dis(xbox_position)
                fake_state = list(self.prev_fake_action)
                fake_action = [fake_state[i] + fake_pure_action[i] for i in range(3)]

                # # 回环检测:
                # print(f'【LOOP】 current_state : {current_state}')
                # print(f'【LOOP】 fake_state : {fake_state}')
                loss = np.array(current_state)-np.array(fake_state)
                print(f'【LOOP】 loss : {loss}')
                if np.any( np.abs(loss) > 0.02 ):
                    self._write_vector(os.path.join(self.data_root_path, 'warning.txt'), f'{self.dir_iter}\n', type='a')
                    print(f'【WARNING】 loop detected! loss: {loss}')    

                ## ik request
                poseStamp = PoseStamped()
                poseStamp.header.frame_id = "base"
                poseStamp.pose.position.x = fake_action[0]
                poseStamp.pose.position.y = fake_action[1]
                poseStamp.pose.position.z = fake_action[2]
                poseStamp.pose.orientation.x = 0.0
                poseStamp.pose.orientation.y = 1.0
                poseStamp.pose.orientation.z = 0.0
                poseStamp.pose.orientation.w = 0.0
                joint_action_item = self.baxter.ik_request('left',poseStamp,timeout=self.timeout)

                if joint_action_item is None:
                    print("[ERROR] Could not solve IK")
                    # clock.tick(self.fps)
                    continue
                    # raise RuntimeError("failed to solve IK for the requested pose")

                ## get joint state and action
                fake_joint_state = list(self.prev_fake_joint_action)
                fake_joint_action = self._joint_vector(joint_action_item)
                
                ## save data
                # 先写样本，再执行动作，保证数据对应的是“当前决策时刻”。
                self._save_data(
                    observation,
                    fake_state,
                    fake_action,
                    fake_joint_state,
                    fake_joint_action,
                    fake_pure_action,
                )
                self._show_preview(observation)

                ## move
                try:
                    self.baxter.left_limb.move_to_joint_positions(joint_action_item, self.timeout)
                except Exception as exc:
                    raise RuntimeError(f"failed to move Baxter to target joint positions: {exc}") from exc
                
                ## data update
                # 当前动作成为下一轮的“上一轮状态”。
                self.prev_fake_state = fake_state
                self.prev_fake_action = fake_action
                self.prev_fake_joint_action = fake_joint_action
                self.data_iter += 1
                
                time.sleep(0.01)
                
                # # 计算FPS
                # self.end_time = time.time()
                # cal_fps = 1.0 / (self.end_time - self.start_time)
                # print(f'【FPS】 {cal_fps:.2f}')

                # clock.tick(self.fps)
            
        except KeyboardInterrupt:
            if self.rs_capture is not None:
                self.rs_capture.stop()
            self.end()
            print("\nGet Ctrl-c: Exit.")
        finally:
            if self.rs_capture is not None:
                self.rs_capture.stop()
            cv2.destroyAllWindows()
            self.end()

    def init_path(self):
        # 每个 epoch 单独建目录，便于按批次整理数据。
        path_names = ['image', 'fake_joint_state', 'fake_joint_action', 'fake_state', 'fake_action', 'fake_pure_action', 'timestamp']
        if self.use_realsense:
            path_names.extend(['rs_image', 'rs_depth'])
        for name in path_names:
            os.makedirs(os.path.join(self.data_root_path,f"{self.dir_iter:0{3}d}",name),exist_ok=True)

    def moving_policy_button(self):
        # hat:前后左右
        # button:x上，y下, home结束
        x,y,z = 0,0,0
        y,x = self.joy.get_hat(0)
        x = 1.0 if x > 0 else -1.0 if x < 0 else 0.0
        y = 1.0 if y > 0 else -1.0 if y < 0 else 0.0
        button_x = self.joy.get_button(2)
        button_y = self.joy.get_button(3)

        if button_x == button_y:
            z = 0.0
        elif button_x:
            z = 1.0
        elif button_y:
            z = -1.0
        
        if x == 0.0 and y == 0.0 and z == 0.0:
            return False
        
        return [x,y,z]

    def moving_policy_axis(self):
        y = self.joy.get_axis(0) # LX
        x = -self.joy.get_axis(1) # LY
        z = -self.joy.get_axis(4) # RY
        
        # 消抖
        if abs(x) <= self.axis_deadzone:
            x = 0.0
        if abs(y) <= self.axis_deadzone:
            y = 0.0
        if abs(z) <= self.axis_deadzone:
            z = 0.0
        if x == 0.0 and y == 0.0 and z == 0.0:
            return False
        return [x,y,z]
    
    def cal_moving_dis(self, xbox_position):
        x, y, z = xbox_position
        if self.restrict_lr:
            length = math.sqrt(x**2 + y**2 + z**2)
            position = [x*self.lr/length,y*self.lr/length,z*self.lr/length]
        else:
            position = [x*self.lr,y*self.lr,z*self.lr]
        return position
    
    def adaptive_lr_by_depth(self, current_depth):
            lr_min_ratio = 0.5   # 最小 lr = lr_peak * 0.2
            gamma = 0.2         # >1: 前期降得慢，后期降得快（末端更精细）
            # depth_range = (0.3264959378579765,-0.05)
            depth_range = (0.2264959378579765,-0.05)

            depth_start, depth_end = depth_range

            total_depth = abs(depth_start - depth_end)
            
            if current_depth > depth_start:
                self.lr =  self.lr_peak
            else:
                progress = abs(current_depth - depth_start) / total_depth
                progress = max(0.0, min(1.0, progress))
                scale = lr_min_ratio + (1.0 - lr_min_ratio) * ((1.0 - progress) ** gamma)
                self.lr =  self.lr_peak * scale


if __name__=='__main__':
    datapath = './data/dataset13'

    xbox = XboxSamplerContinue(data_root_path=datapath)
    xbox.sampling_pipline()