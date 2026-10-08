import rospy
import baxter_interface

rospy.init_node('usssr')

left_gripper = baxter_interface.Gripper('left')
right_gripper = baxter_interface.Gripper('right')

left_limb = baxter_interface.Limb('left')
right_limb = baxter_interface.Limb('right')

init_left = left_limb.joint_angles()
init_right = right_limb.joint_angles()

w_2 = 0

#----------------S0------S1------E0------E1------W0------W1---
demo_test  = [-0.7969,-0.3877,-0.7965, 1.2379, 0.7831, 1.0569,  # cloth1 left
               0.2324,-0.7904, 0.8245, 2.2822,-1.1309, 0.6343,  # cloth2 right
               1.3764,-0.5377, 0.2174, 1.1233,-0.1645, 0.9610,  # cloth3 right
               1.0688,-0.7206, 0.5246, 1.7411,-0.5400, 0.7344,  # cloth4 right
               0.5852,-0.6600, 0.4564, 1.3323,-0.4222, 0.9660]  # cloth5 right

current_demo = demo_test

init_left['left_s0']=0.30
init_left['left_s1']=-0.50
init_left['left_e0']=-0.80
init_left['left_e1']=0.50
init_left['left_w0']=0.80
init_left['left_w1']=1.80
init_left['left_w2']=w_2

init_right['right_s0']=-0.30
init_right['right_s1']=-0.50
init_right['right_e0']=0.80
init_right['right_e1']=0.50
init_right['right_w0']=-0.80
init_right['right_w1']=1.80
init_right['right_w2']=w_2

left_up = left_limb.joint_angles()
right_up = right_limb.joint_angles()

left_up['left_s0']=-0.36
left_up['left_s1']=-1.00
left_up['left_e0']=-1.28
left_up['left_e1']=0.80
left_up['left_w0']=0.60
left_up['left_w1']=2.08
left_up['left_w2']=w_2

right_up['right_s0']=0.36
right_up['right_s1']=-1.00
right_up['right_e0']=1.28
right_up['right_e1']=0.80
right_up['right_w0']=-0.60
right_up['right_w1']=2.08
right_up['right_w2']=w_2

target_c = left_limb.joint_angles()
target_b = right_limb.joint_angles()
target_a = right_limb.joint_angles()

target_c['left_s0']=0.015
target_c['left_s1']=-0.30
target_c['left_e0']=-0.70
target_c['left_e1']=0.85
target_c['left_w0']=0.75
target_c['left_w1']=1.24
target_c['left_w2']=w_2

target_b['right_s0']=-0.90
target_b['right_s1']=-0.20
target_b['right_e0']=1.40
target_b['right_e1']=1.60
target_b['right_w0']=-1.36
target_b['right_w1']=1.42
target_b['right_w2']=w_2

target_a['right_s0']=-0.015
target_a['right_s1']=-0.30
target_a['right_e0']=0.70
target_a['right_e1']=0.85
target_a['right_w0']=-0.75
target_a['right_w1']=1.24
target_a['right_w2']=w_2

cloth1 = left_limb.joint_angles()
cloth2 = right_limb.joint_angles()
cloth3 = right_limb.joint_angles()
cloth4 = right_limb.joint_angles()
cloth5 = right_limb.joint_angles()

cloth1['left_s0']=current_demo[0]
cloth1['left_s1']=current_demo[1]
cloth1['left_e0']=current_demo[2]
cloth1['left_e1']=current_demo[3]
cloth1['left_w0']=current_demo[4]
cloth1['left_w1']=current_demo[5]
cloth1['left_w2']=w_2

cloth2['right_s0']=current_demo[6]
cloth2['right_s1']=current_demo[7]
cloth2['right_e0']=current_demo[8]
cloth2['right_e1']=current_demo[9]
cloth2['right_w0']=current_demo[10]
cloth2['right_w1']=current_demo[11]
cloth2['right_w2']=w_2

cloth3['right_s0']=current_demo[12]
cloth3['right_s1']=current_demo[13]
cloth3['right_e0']=current_demo[14]
cloth3['right_e1']=current_demo[15]
cloth3['right_w0']=current_demo[16]
cloth3['right_w1']=current_demo[17]
cloth3['right_w2']=w_2

cloth4['right_s0']=current_demo[18]
cloth4['right_s1']=current_demo[19]
cloth4['right_e0']=current_demo[20]
cloth4['right_e1']=current_demo[21]
cloth4['right_w0']=current_demo[22]
cloth4['right_w1']=current_demo[23]
cloth4['right_w2']=w_2

cloth5['right_s0']=current_demo[24]
cloth5['right_s1']=current_demo[25]
cloth5['right_e0']=current_demo[26]
cloth5['right_e1']=current_demo[27]
cloth5['right_w0']=current_demo[28]
cloth5['right_w1']=current_demo[29]
cloth5['right_w2']=w_2

left_limb.move_to_joint_positions(init_left)
right_limb.move_to_joint_positions(init_right)

'''
left_limb.move_to_joint_positions(left_up)
left_limb.move_to_joint_positions(cloth1)
left_gripper.close(block=True, timeout=1.0)
left_limb.move_to_joint_positions(left_up)
left_limb.move_to_joint_positions(init_left)
left_limb.move_to_joint_positions(target_c)
left_gripper.open(block=True, timeout=1.0)
left_limb.move_to_joint_positions(init_left)
'''

right_limb.move_to_joint_positions(right_up)
right_limb.move_to_joint_positions(cloth5)
right_gripper.close(block=True, timeout=1.0)
right_limb.move_to_joint_positions(right_up)
right_limb.move_to_joint_positions(init_right)
right_limb.move_to_joint_positions(target_a)
right_gripper.open(block=True, timeout=1.0)
right_limb.move_to_joint_positions(init_right)

right_limb.move_to_joint_positions(right_up)
right_limb.move_to_joint_positions(cloth4)
right_gripper.close(block=True, timeout=1.0)
right_limb.move_to_joint_positions(right_up)
right_limb.move_to_joint_positions(init_right)
right_limb.move_to_joint_positions(target_b)
right_gripper.open(block=True, timeout=1.0)
right_limb.move_to_joint_positions(init_right)

right_limb.move_to_joint_positions(right_up)
right_limb.move_to_joint_positions(cloth3)
right_gripper.close(block=True, timeout=1.0)
right_limb.move_to_joint_positions(right_up)
right_limb.move_to_joint_positions(init_right)
right_limb.move_to_joint_positions(target_a)
right_gripper.open(block=True, timeout=1.0)
right_limb.move_to_joint_positions(init_right)
