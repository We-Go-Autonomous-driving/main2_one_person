#! /usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
from scout_msgs.msg import ScoutLightCmd

class scout_pub_basic():
    def __init__(self):
        rospy.init_node("scout_pub_basic_name", anonymous=False)
        self.msg_pub = rospy.Publisher(
            '/cmd_vel',
            Twist,
            queue_size = 10
        )
        self.msg_pub2 = rospy.Publisher(
            '/scout_light_control',
            ScoutLightCmd,
            queue_size = 10
        )

        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.th = 0.0
        self.speed = 0.0
        self.turn = 0.0

    def update(self,x,y,z,th,speed,turn):
        self.x = x
        self.y = y
        self.z = z
        self.th = th
        self.speed = speed
        self.turn = turn

    def sendMsg(self, mode):
        tt = Twist()
        light = ScoutLightCmd()
        light.enable_cmd_light_control = True
        light.front_mode = mode

        tt.linear.x = self.x * self.speed
        tt.linear.y = self.y * self.speed
        tt.linear.z = self.z * self.speed
        tt.angular.x = 0
        tt.angular.y = 0
        tt.angular.z = self.th * self.turn
        self.msg_pub.publish(tt)
        self.msg_pub2.publish(light)

# if __name__ == '__main__':
#     count = 0

#     x = 0
#     y = 0
#     z = 0
#     th = 0
#     speed = 0.5
#     turn = 1
    
#     moveBindings = {
#         'go':(1,0,0,0), # i
#         'go_turn_right':(1,0,0,-1), # o
#         'turn_left':(0,0,0,1), # j
#         'turn_right':(0,0,0,-1), # l
#         'go_turn_left':(1,0,0,1), # u
#         'back':(-1,0,0,0), # ,
#         'back_right':(-1,0,0,1), # .
#         'back_left':(-1,0,0,-1), # m
#         # 'I':(1,0,0,0), # It's same as 'i'
#         'parallel_go_right':(1,-1,0,0), # O
#         'parallel_left':(0,1,0,0), # J
#         'parallel_right':(0,-1,0,0), # L
#         'parallel_go_left':(1,1,0,0), # U
#         # '<':(-1,0,0,0), # It's same as ','
#         'parallel_back_right':(-1,-1,0,0), # >
#         'parallel_back_left':(-1,1,0,0), # M
#         # 't':(0,0,1,0),  # it's for drone
#         # 'b':(0,0,-1,0), # it's for drone
#     }

#     speedBindings={
#         'total_speed_up':(1.1,1.1), # q
#         'total_speed_down':(.9,.9),   # z
#         'linear_speed_up':(1.1,1),   # w
#         'linear_speed_down':(.9,1),    # x
#         'angular_speed_up':(1,1.1), # e
#         'angular_speed_down':(1,.9),  # c
#     }    

    # go = scout_pub_basic()
    # rate = rospy.Rate(10)
    
    # while not rospy.is_shutdown():
    #     go = scout_pub_basic()
    #     rate = rospy.Rate(30)
    #     go.update(x,y,z,th,speed,turn)
    #     go.sendMsg()

    #     if count >= 10:
    #         if str(count)[-2] in ['1','3','5','7','9']:
    #             speed*=speedBindings['total_speed_up'][0]
    #             # turn+=speedBindings['total_speed_up'][1]
    #         # else:
    #         #     speed*=speedBindings['total_speed_down'][0]
    #             # turn-=speedBindings['total_speed_up'][1]
    #     else:
    #         x+=0.1
    #         # y+=0.1
    #         # z+=0.1

    #     print('count: {}, speed: {}, turn : {}'.format(count,speed,turn))
    #     rate.sleep()
    #     count+=1