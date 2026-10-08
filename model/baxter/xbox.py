
import pygame
import time
import rospy
from baxter_interface import Limb, Gripper
from baxter_core_msgs.srv import SolvePositionIK, SolvePositionIKRequest
from geometry_msgs.msg import PoseStamped
import threading

pygame.init()
rospy.init_node('xbox_baxter_control', anonymous=True)

left_limb = Limb('left')
right_limb = Limb('right')
left_gripper = Gripper('left')
right_gripper = Gripper('right')

left_gripper_open = True
right_gripper_open = True
prev_left_stick_btn = 0
prev_right_stick_btn = 0

joystick = pygame.joystick.Joystick(0)
joystick.init()

service_name_left  = "/ExternalTools/left/PositionKinematicsNode/IKService"
service_name_right = "/ExternalTools/right/PositionKinematicsNode/IKService"

rospy.wait_for_service(service_name_left,  timeout=5.0)
rospy.wait_for_service(service_name_right, timeout=5.0)

ik_left_srv  = rospy.ServiceProxy(service_name_left,  SolvePositionIK)
ik_right_srv = rospy.ServiceProxy(service_name_right, SolvePositionIK)

def ik_request(limb, pose, timeout=0.3):
    """
    在超时 timeout 秒内尝试调用 IK 服务，超时或出错则返回 None，
    否则返回关节字典。
    """
    srv = ik_left_srv if limb == "left" else ik_right_srv
    req = SolvePositionIKRequest()
    req.pose_stamp.append(pose)
    req.seed_mode = req.SEED_CURRENT

    result = {"resp": None}
    def call_srv():
        try:
            result["resp"] = srv(req)
        except rospy.ServiceException as e:
            rospy.logwarn(f"[{limb}] IK ServiceException: {e}")

    # 2) 启动后台线程
    th = threading.Thread(target=call_srv, daemon=True)
    th.start()
    th.join(timeout)    # 等待 timeout 秒

    # 3) 超时或服务内部报错 result["resp"] 仍为 None
    resp = result["resp"]
    if resp is None:
        rospy.logwarn(f"[{limb}] IK call timed out after {timeout:.3f}s")
        return None

    # 4) 检查 IK 解算是否有效
    if resp.isValid and resp.isValid[0]:
        return dict(zip(resp.joints[0].name,
                        resp.joints[0].position))
    else:
        return None

def move_limb(arm, x, y, z):
    """
    移动左臂到指定位置 (x, y, z)。
    """
    pose = PoseStamped()
    pose.header.frame_id = "base"
    pose.pose.position.x = x
    pose.pose.position.y = y
    pose.pose.position.z = z
    pose.pose.orientation.x = 0.0  # 默认朝向
    pose.pose.orientation.y = 1.0  # 默认朝向
    pose.pose.orientation.z = 0.0  # 默认朝向
    pose.pose.orientation.w = 0.0  # 默认朝向
    if arm == "right":
        joints = ik_request("right", pose)
        if joints:
            right_limb.move_to_joint_positions(joints, timeout=0.3)
    else:
        joints = ik_request("left", pose)
        if joints:
            left_limb.move_to_joint_positions(joints, timeout=0.3)

while not rospy.is_shutdown():
    left_pose = left_limb.endpoint_pose()
    right_pose = right_limb.endpoint_pose()
    left_x, left_y, left_z = left_pose['position'].x, left_pose['position'].y, left_pose['position'].z
    right_x, right_y, right_z = right_pose['position'].x, right_pose['position'].y, right_pose['position'].z

    pygame.event.pump()
    if joystick.get_button(11) == 1:
        break
    
    lx = joystick.get_axis(0)
    ly = joystick.get_axis(1)
    rx = joystick.get_axis(3)
    ry = joystick.get_axis(4)
    l1 = joystick.get_button(4)
    l2 = joystick.get_axis(2)
    if l2 < 0:
        l2 = 0
    else :
        l2 = 1

    r1 = joystick.get_button(5)
    r2 = joystick.get_axis(5)
    if r2 < 0:
        r2 = 0
    else :
        r2 = 1
    # left_stick_btn = joystick.get_button(2)
    # right_stick_btn = joystick.get_button(1)
    left_stick_btn = joystick.get_button(9)
    right_stick_btn = joystick.get_button(10)
    step_xy = 0.1
    step_z = 0.1


    if abs(lx) > 0.1 or abs(ly) > 0.1 or l1 or l2:
    # 左臂目标
        left_x += -ly * step_xy
        left_y += -lx * step_xy
        left_z += (l1 - l2) * step_z
        move_limb("left", left_x, left_y, left_z)

    # 右臂目标
    if abs(rx) > 0.1 or abs(ry) > 0.1 or r1 or r2:
        # 右臂目标
        right_x += -ry * step_xy
        right_y += -rx * step_xy
        right_z += (r1 - r2) * step_z
        move_limb("right", right_x, right_y, right_z)


    if left_stick_btn and not prev_left_stick_btn:
        if left_gripper_open:
            left_gripper.close()
        else:
            left_gripper.open()
        left_gripper_open = not left_gripper_open
        # time.sleep(0.1)
    prev_left_stick_btn = left_stick_btn

    # 右夹爪
    if right_stick_btn and not prev_right_stick_btn:
        if right_gripper_open:
            right_gripper.close()
        else:
            right_gripper.open()
        right_gripper_open = not right_gripper_open
    prev_right_stick_btn = right_stick_btn
    
    time.sleep(0.1)

    # time.sleep(0.05)



'''
0 : 左摇杆水平轴
1 : 左摇杆垂直轴
2 : LT按钮
3 : 右摇杆水平轴
4 : 右摇杆垂直轴

'''