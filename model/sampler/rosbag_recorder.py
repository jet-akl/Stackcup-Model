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

class RosbagRecorder:
    def __init__(self,
                 topic_list,
                 tolerance=0.01,
                 fps=10):
        self.topic_list = topic_list
        self.tolerance = tolerance
        self.fps = fps

        self.bag = None
        self.last_record_time = 0.0  # 上一次录制写入时间（用于控制频率）
        self._recording = False

        # 校验话题列表
        if self.topic_list is None:
            raise ValueError('Topic list should not be None')

        # 初始化多话题订阅器（基于message_filters，保证同时性）
        self._subscribers = []
        for topic in self.topic_list:
            # 注意：此处消息类型默认用Image/String，需根据实际话题修改对应消息类型
            sub = Subscriber(topic[0], topic[1])
            self._subscribers.append(sub)
            rospy.loginfo(f"ROS has subscribed to the topic:{topic[0]}")

        # 初始化时间同步器（核心：保证多话题消息同时性）
        self.sync = ApproximateTimeSynchronizer(self._subscribers,
                                                queue_size=100,
                                                slop=self.tolerance)
        # 注册同步回调函数
        self.sync.registerCallback(self.sync_callback)

    def sync_callback(self, *msgs):
        """
        多话题同步回调函数（所有话题消息达到时间同步后触发）
        :param msgs: 对应话题列表的消息对象（顺序与topic_list一致）
        """
        # 非录制状态,或未初始化包,直接返回
        if not self._recording or self.bag is None:
            return

        # 控制录制频率（按设定Hz写入，避免高频消息导致bag过大）
        current_time = rospy.get_time()
        thread_time = 1.0 / self.fps
        if current_time - self.last_record_time < thread_time:
            return

        # 写入rosbag（保留消息原始时间戳，保证数据有效性）
        try:
            for topic, msg in zip(self.topic_list, msgs):
                self.bag.write(topic[0], msg, t=msg.header.stamp if hasattr(msg, "header") else rospy.Time.now())
            self.last_record_time = current_time  # 更新上一次写入时间
        except Exception as e:
            rospy.logerr(f"写入rosbag失败：{str(e)}")

    def init_record(self,bag_filename):
        if self.bag is None:
            self._recording = False
            self.bag = rosbag.Bag(bag_filename,'w')
            self.last_record_time = 0.0
            rospy.loginfo('Init record!')
            rospy.loginfo(f"Rosbag:{bag_filename} init success!")
        else:
            rospy.loginfo(f"Warnning! Rosbag:{bag_filename} is already init")
            # rospy.logerr(f"Fail init : Rosbag is already init")
        
    def start_record(self):
        if self.bag is not None:
            self._recording=True
            rospy.loginfo('Start record!')
        else:
            rospy.logerr(f"Fail start : Rosbag is None")
    
    def pause_record(self):
        if self.bag is not None:
            self._recording=False
            rospy.loginfo('Pause record!')
        else:
            rospy.logerr(f"Fail pause : Rosbag is None")

    def end_record(self):
        if self.bag is not None:
            self.bag.close()
            self.bag = None
            self._recording=False
            rospy.loginfo('End record!')
            rospy.loginfo(f"Rosbag close success!")
        # else:
        #     rospy.logerr(f"Fail close : Rosbag is None")

    def run(self):
        """运行节点主循环"""
        rospy.spin()