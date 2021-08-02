import airsim
import setup_path
import time
from datetime import datetime

x_diff = 0
x_diff_pre = 0 #上一次的误差
x_Perror = 0
x_Ierror = 0
x_Derror = 0

y_diff = 0
y_diff_pre = 0
y_Perror = 0
y_Ierror = 0
y_Derror = 0

timepause = 1
error = 1
# official position wanted (will not be changed throughout)
x_poswanted = 10
y_poswanted = 10

done = 1

def PIDcontroller(x_pos, y_pos, z_pos):
    global x_Ierror
    global x_Derror
    global x_Perror
    global x_diff
    global x_diff_pre

    global y_Ierror
    global y_Derror
    global y_Perror
    global y_diff
    global y_diff_pre

    global done



    # setts up initially x_diff  and y_diff value(will be change on each loop)
    x_diff = x_poswanted - x_pos #x方向上位置误差
    y_diff = y_poswanted - y_pos #y方向上位置误差

    # initial print of data
    print(f"x_pos is {x_pos} and x_diff is {x_diff}")
    print(f"y_poss is {y_pos} and y_diff is {y_diff}")

    if abs(x_diff) > error or abs(y_diff) > error :
        print("loop entered")
        # gains
        Kp = 1.2#1.2  # 100 这些参数怎么设置 （0.1，0.15，0.1）
        Ki = 0.01
        Kd = 0.001

        #  x_diff is error
        x_Ierror = x_Ierror + x_diff * timepause
        x_Derror = (x_diff - x_diff_pre) / timepause
        print(f"the x_Ierror is {x_Ierror} the x_Derror is {x_Derror} ")
        x_V = Kp * x_diff + Ki * x_Ierror + Kd * x_Derror

        #  y_diff is error
        y_Ierror = y_Ierror + y_diff * timepause
        y_Derror = (y_diff - y_diff_pre) / timepause
        print(f"the y_Ierror is {y_Ierror} the y_Derror is {y_Derror} ")
        y_V = Kp * y_diff + Ki * y_Ierror + Kd * y_Derror

        print(f"the x_V is {x_V}  and the y_V is {y_V}")#pid 输出的是x,y方向上的速度

        # velocity movement
        client.moveByVelocityAsync(x_V, y_V, -0.1, timepause).join()

        # new positioning
        x_diff_pre = x_diff#保存这次的误差作为上一次的误差
        y_diff_pre = y_diff
    else:
        done = 0
    print(f"done in the PIDController is {done}")


if __name__ == '__main__':

    # pivals = np.arange(0,2.01,0.01)
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    client.takeoffAsync().join()
    client.moveToPositionAsync(0, 0, -4, 1).join()
    start_time = datetime.now()
    print("the start time is ",start_time)

    while(done):
        print(f"done in the main is {done}")
        position = client.getMultirotorState().kinematics_estimated.position
        velocity = client.getMultirotorState().kinematics_estimated.linear_velocity
        X_pos = position.x_val
        Y_pos = position.y_val
        Z_pos = position.z_val
        PIDcontroller(X_pos, Y_pos, Z_pos)
        print(f"the velocity is {velocity}")
        time.sleep(2)

    end_time = datetime.now()
    print(f"the end time is {end_time} and all cost {end_time - start_time}", )

    #rospy.spin()
#