#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
import rosbag
import threading
import time
from pynput import keyboard
from message_filters import ApproximateTimeSynchronizer, Subscriber
# 导入你需要订阅的话题消息类型（示例：Image和String，可根据实际需求替换）
from sensor_msgs.msg import Image
from std_msgs.msg import String

class RosbagRecorderWithSync:
    def __init__(self):
        # 1. 初始化核心参数（可根据需求修改）
        self.recording = False  # 录制状态标记（False：未录制，True：录制中）
        self.bag = None  # rosbag对象
        self.bag_filename = rospy.get_param("~bag_filename", "recorded_sync.bag")  # bag文件保存名称
        self.target_topics = rospy.get_param("~target_topics", ["/camera/image_raw", "/chatter"])  # 待订阅话题列表
        self.sync_tolerance = rospy.get_param("~sync_tolerance", 0.01)  # 多话题时间同步容忍度（秒），越小同步精度越高
        self.record_frequency = rospy.get_param("~record_frequency", 10.0)  # 录制频率（Hz），可调
        self.last_record_time = 0.0  # 上一次录制写入时间（用于控制频率）

        # 2. 校验话题列表
        if not self.target_topics:
            rospy.logerr("错误：未指定需要订阅的话题，请配置target_topics参数！")
            rospy.signal_shutdown("话题列表为空")
            return

        # 3. 初始化多话题订阅器（基于message_filters，保证同时性）
        self.topic_subscribers = []
        for topic in self.target_topics:
            # 注意：此处消息类型默认用Image/String，需根据实际话题修改对应消息类型
            if "image" in topic:
                sub = Subscriber(topic, Image)
            else:
                sub = Subscriber(topic, String)
            self.topic_subscribers.append(sub)
            rospy.loginfo(f"已订阅话题：{topic}")

        # 4. 初始化时间同步器（核心：保证多话题消息同时性）
        # 选用ApproximateTimeSynchronizer（近似时间同步），适合大多数场景（允许微小时间偏差）
        # 若需严格时间同步，可替换为TimeSynchronizer（要求所有话题消息时间戳完全一致，场景有限）
        self.sync = ApproximateTimeSynchronizer(self.topic_subscribers,
                                                queue_size=100,
                                                slop=self.sync_tolerance)
        self.sync.registerCallback(self.sync_callback)  # 注册同步回调函数

        # 5. 启动键盘监听线程（实现启停开关，不阻塞ROS主循环）
        self.key_listener_thread = threading.Thread(target=self.keyboard_listener, daemon=True)
        self.key_listener_thread.start()

        rospy.loginfo("初始化完成！请按以下按键操作：")
        rospy.loginfo("  S键：开始录制rosbag")
        rospy.loginfo("  E键：结束录制rosbag")
        rospy.loginfo("  Q键：退出整个程序")

    def sync_callback(self, *msgs):
        """
        多话题同步回调函数（所有话题消息达到时间同步后触发）
        :param msgs: 对应话题列表的消息对象（顺序与target_topics一致）
        """
        # 非录制状态，直接返回
        if not self.recording:
            return

        # 控制录制频率（按设定Hz写入，避免高频消息导致bag过大）
        current_time = rospy.get_time()
        time_interval = 1.0 / self.record_frequency  # 理论时间间隔
        if current_time - self.last_record_time < time_interval:
            return

        # 写入rosbag（保留消息原始时间戳，保证数据有效性）
        try:
            for topic, msg in zip(self.target_topics, msgs):
                self.bag.write(topic, msg, t=msg.header.stamp if hasattr(msg, "header") else rospy.Time.now())
            self.last_record_time = current_time  # 更新上一次写入时间
        except Exception as e:
            rospy.logerr(f"写入rosbag失败：{str(e)}")

    def keyboard_listener(self):
        """键盘监听函数（独立线程运行）"""
        def on_press(key):
            """按键按下回调"""
            try:
                # 开始录制（S键）
                if key.char == 's' or key.char == 'S':
                    if not self.recording:
                        self.start_recording()
                    else:
                        rospy.logwarn("当前已处于录制状态，无需重复开始！")

                # 结束录制（E键）
                elif key.char == 'e' or key.char == 'E':
                    if self.recording:
                        self.stop_recording()
                    else:
                        rospy.logwarn("当前未处于录制状态，无需重复结束！")

                # 退出程序（Q键）
                elif key.char == 'q' or key.char == 'Q':
                    rospy.loginfo("收到退出指令，正在收尾...")
                    if self.recording:
                        self.stop_recording()
                    rospy.signal_shutdown("用户主动退出")
            except AttributeError:
                pass  # 忽略特殊按键（如回车、空格等）

        # 启动键盘监听
        with keyboard.Listener(on_press=on_press) as listener:
            listener.join()

    def start_recording(self):
        """开始录制rosbag"""
        try:
            # 初始化bag文件（写入模式：'w'覆盖，'a'追加）
            self.bag = rosbag.Bag(self.bag_filename, 'w')
            self.recording = True
            self.last_record_time = rospy.get_time()
            rospy.loginfo(f"开始录制rosbag！文件保存为：{self.bag_filename}")
            rospy.loginfo(f"当前录制频率：{self.record_frequency} Hz，同步容忍度：{self.sync_tolerance} 秒")
        except Exception as e:
            rospy.logerr(f"启动录制失败：{str(e)}")
            self.recording = False

    def stop_recording(self):
        """结束录制rosbag"""
        try:
            self.recording = False
            if self.bag:
                self.bag.close()
                rospy.loginfo(f"结束录制rosbag！文件已保存：{self.bag_filename}")
                self.bag = None
        except Exception as e:
            rospy.logerr(f"停止录制失败：{str(e)}")

    def run(self):
        """运行节点主循环"""
        rospy.spin()

if __name__ == "__main__":
    try:
        # 初始化ROS节点（带参数服务器，支持外部配置参数）
        rospy.init_node("sync_rosbag_recorder", anonymous=True)
        # 实例化录制类并运行
        recorder = RosbagRecorderWithSync()
        recorder.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("节点正常退出")
    except Exception as e:
        rospy.logerr(f"节点运行异常：{str(e)}")
