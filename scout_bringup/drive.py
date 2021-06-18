def drive(cx, left_limit, right_limit, turn, speed):
    # Target의 위치 파악(좌우 회전이 필요한 지)
    if cx <= left_limit: 
        key = 'turn_left' # bbox 중앙점 x좌푯값이 좌측 회전 한곗값(left_limit)보다 작으면 좌회전
        turn = 1
    elif cx >= right_limit: 
        key = 'turn_right' # bbox 중앙점 x좌푯값이 우측 회전 한곗값(right_limit)보다 크면 우회전
        turn = 1
    else: # 좌/우 회전이 아니라면 직진, 거리에 따른 속도 제어
        key = 'go'

    return key, speed, turn

def drive2(cx, left_limit, right_limit, turn, frame, speed, max_speed, max_turn):
    # 회전 속도가 빠른 구간
    speed_up_area = 50
    
    # Target의 위치 파악(좌우 회전이 필요한 지)
    if cx <= left_limit: 
        key = 'go_turn_left' # bbox 중앙점 x좌푯값이 좌측 회전 한곗값(left_limit)보다 작으면 좌회전
    elif cx < speed_up_area:
        key = 'go_turn_left'
        speed = max_speed
        turn = max_turn
    elif cx >= right_limit: 
        key = 'go_turn_right' # bbox 중앙점 x좌푯값이 우측 회전 한곗값(right_limit)보다 크면 우회전
    elif cx > frame.shape[1] - speed_up_area:
        key = 'go_turn_right'
        speed = max_speed
        turn = max_turn
    else: # 좌/우 회전이 아니라면 직진, 거리에 따른 속도 제어
        key = 'go'

    return key, speed, turn

def drive3(cx, left_limit, right_limit, turn, frame, speed, max_speed, min_speed, max_turn, stable_min_dist, stable_max_dist, person_distance, start_speed_down=300):
    # 회전 속도가 빠른 구간
    speed_up_area = 50
    
    # Target의 위치 파악(좌우 회전이 필요한 지)
    if cx <= left_limit: 
        key = 'go_turn_left' # bbox 중앙점 x좌푯값이 좌측 회전 한곗값(left_limit)보다 작으면 좌회전
    elif cx < speed_up_area:
        key = 'go_turn_left'
        speed = max_speed
        turn = max_turn
    elif cx >= right_limit: 
        key = 'go_turn_right' # bbox 중앙점 x좌푯값이 우측 회전 한곗값(right_limit)보다 크면 우회전
    elif cx > frame.shape[1] - speed_up_area:
        key = 'go_turn_right'
        speed = max_speed
        turn = max_turn
    else: # 좌/우 회전이 아니라면 직진, 거리에 따른 속도 제어
        if stable_min_dist <= person_distance <= stable_max_dist:
            key = 'go' # 로봇과 사람과의 거리(person_distance)가 2.0(stable_min_dist) ~ 2.5m(stable_max_dist)라면 전진
        else: # stable_max_dist 초과일 경우, 거리에 따른 속도 증감
            remaining_distance = person_distance - stable_max_dist # 안정 거리 최대값과 사람과의 거리의 차이, 이를 이용해 얼만큼 속도를 증감해야하는 지 정하는 요소
            """
            기본 컨셉은 remaining_distance 300 이상이면 속도 증가, 미만이면 속도 감소(대신 속도 최대값은 0.8, 최소값은 0.5로 설정한다)
            예를 들어 사람과 로봇의 거리가 2500이라면 remaining_distance 500이 된다. 로봇은 speed_fremaining_distanceactor이 200이 될 때까지 속도 증가(최댓값은 2.5로 설정함.)
            remaining_distance 200 미만이 되는 순간 속도 감소(최솟값은 1.2로 설정)
            그리고 remaining_distance 0이 되면 안정 구간 진입이므로 속도는 1로 설정됨.
            
            <아래는 로봇과 사람이 4.0m 떨어진 상황>
                                                                                                           [사람]
            [로봇] <----위험(stop)---->|<---직진(go)--->|<-------------속도 증가------------->|<---속도 감소--->|
                                     2.0m             2.5m                                 3.7m              4.0m
            """
            if remaining_distance >= start_speed_down: # speed up
                if speed < max_speed:
                    key = 'linear_speed_up'
                else:
                    key = 'go'
                    speed = max_speed
            else: # speed down
                if speed > min_speed:
                    key = 'linear_speed_down'
                else:
                    key = 'go'
                    speed = min_speed
    return key, speed, turn

def drive4(cx, left_limit, right_limit, turn, frame, speed, max_speed, min_speed, max_turn, stable_min_dist, stable_max_dist, person_distance, start_speed_down=400):
    # 회전 속도가 빠른 구간
    speed_up_area = 70
    edge_turn = 0.6

    speed_default = 0.85
    turn_default = 0.6

    # Target의 위치 파악(좌우 회전이 필요한 지)
    if cx <= left_limit: 
        key = 'go_turn_left' # bbox 중앙점 x좌푯값이 좌측 회전 한곗값(left_limit)보다 작으면 좌회전
        speed = 0.5
        turn = turn_default
    elif cx < speed_up_area:
        key = 'turn_left'
        turn = edge_turn
    elif cx >= right_limit: 
        key = 'go_turn_right' # bbox 중앙점 x좌푯값이 우측 회전 한곗값(right_limit)보다 크면 우회전
        speed = 0.5
        turn = turn_default
    elif cx > frame.shape[1] - speed_up_area:
        key = 'turn_right'
        turn = edge_turn
    else: # 좌/우 회전이 아니라면 직진, 거리에 따른 속도 제어
        if stable_min_dist <= person_distance <= stable_max_dist:
            key = 'go' # 로봇과 사람과의 거리(person_distance)가 2.0(stable_min_dist) ~ 2.5m(stable_max_dist)라면 전진
            speed = speed_default
        else: # stable_max_dist 초과일 경우, 거리에 따른 속도 증감
            remaining_distance = person_distance - stable_max_dist # 안정 거리 최대값과 사람과의 거리의 차이, 이를 이용해 얼만큼 속도를 증감해야하는 지 정하는 요소
            """
            기본 컨셉은 remaining_distance 300 이상이면 속도 증가, 미만이면 속도 감소(대신 속도 최대값은 0.8, 최소값은 0.5로 설정한다)
            예를 들어 사람과 로봇의 거리가 2500이라면 remaining_distance 500이 된다. 로봇은 speed_fremaining_distanceactor이 200이 될 때까지 속도 증가(최댓값은 2.5로 설정함.)
            remaining_distance 200 미만이 되는 순간 속도 감소(최솟값은 1.2로 설정)
            그리고 remaining_distance 0이 되면 안정 구간 진입이므로 속도는 1로 설정됨.
            
            <아래는 로봇과 사람이 4.0m 떨어진 상황>
                                                                                                           [사람]
            [로봇] <----위험(stop)---->|<---직진(go)--->|<-------------속도 증가------------->|<---속도 감소--->|
                                     2.0m             2.5m                                 3.7m              4.0m
            """
            if remaining_distance >= start_speed_down: # speed up
                if speed < max_speed:
                    key = 'linear_speed_up'
                else:
                    key = 'go'
                    speed = max_speed
            else: # speed down
                if speed > min_speed:
                    key = 'linear_speed_down'
                else:
                    key = 'go'
                    speed = min_speed
    return key, speed, turn