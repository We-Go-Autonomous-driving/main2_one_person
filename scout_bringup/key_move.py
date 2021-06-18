# key 변수로 모터 제어하는 함수
def key_move(key,x,y,z,th,speed,turn):
    moveBindings = {
        'go':(1,0,0,0), # i
        'go_turn_right':(1,0,0,-1), # o
        'turn_left':(0,0,0,1), # j
        'turn_right':(0,0,0,-1), # l
        'go_turn_left':(1,0,0,1), # u
        'back':(-1,0,0,0), # ,
        'back_right':(-1,0,0,1), # .
        'back_left':(-1,0,0,-1), # m
        # 'I':(1,0,0,0), # It's same as 'i'
        'parallel_go_right':(1,-1,0,0), # O
        'parallel_left':(0,1,0,0), # J
        'parallel_right':(0,-1,0,0), # L
        'parallel_go_left':(1,1,0,0), # U
        # '<':(-1,0,0,0), # It's same as ','
        'parallel_back_right':(-1,-1,0,0), # >
        'parallel_back_left':(-1,1,0,0), # M
        # 't':(0,0,1,0),  # it's for drone
        # 'b':(0,0,-1,0), # it's for drone
    }

    speedBindings={
        'total_speed_up':(1.1,1.1), # q
        'total_speed_down':(.9,.9),   # z
        'linear_speed_up':(1.1,1),   # w
        'linear_speed_down':(.9,1),    # x
        'angular_speed_up':(1,1.1), # e
        'angular_speed_down':(1,.9),  # c
    }    

    if key in moveBindings.keys():
        x = moveBindings[key][0]
        y = moveBindings[key][1]
        z = moveBindings[key][2]
        th = moveBindings[key][3]
    elif key in speedBindings.keys():
        speed = speed * speedBindings[key][0]
        turn = turn * speedBindings[key][1]
    else:
        x = 0
        y = 0
        z = 0
        th = 0

    return x,y,z,th,speed,turn

# if __name__ == "__main__":
#     key_move(key,x,y,z,th,speed,turn)