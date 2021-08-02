import airsim
import setup_path
import time
from datetime import datetime

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

#位置式PID: u(k) = Kp * e(k) + Ki * ∑e(i) + Kd * e(k) - e(k-1)
#e(k): 用户设定的值（目标值） -  控制对象的当前的状态值
#比例P :    e(k)
#积分I :   ∑e(i)  误差的累加
#微分D :  e(k) - e(k-1) 这次误差-上次误差

Kp = 0.8  # 0.1 最后速度太大（0.7，0.7，0.6）#1.2#1.2  # 100 这些参数怎么设置 （0.1，0.15，0.1）
Ki = 0.001# 0.0001追不上
Kd = 0.1


x_diff_pre = 0 #上一次的误差
x_Perror = 0 #x方向上的Perror
x_Ierror = 0 #x方向上的Ierror
x_Derror = 0 #x方向上的Derror


y_diff_pre = 0
y_Perror = 0
y_Ierror = 0
y_Derror = 0


z_diff_pre = 0
z_Perror = 0
z_Ierror = 0
z_Derror = 0

timepause = 1 #每次飞行的时间
xy_error = 0.5
z_error = 0.5 #控制精度误差无人机大概到0.4m 就算是撞到地面上了
# official position wanted (will not be changed throughout)

# x_poswanted = 0 #降落的目标点是在x方向上速度为 1.2m/s 的移动平台（x,0,0）
# y_poswanted = 0
# z_poswanted = 0

done = 1  #判断结束标志
collision = False

x_record_list = []
y_record_list = []
z_record_list = []



def PIDcontroller(x_diff, y_diff, z_diff):
    #申明全局变量 以供下面修改
    global x_diff_pre
    global x_Ierror
    global x_Derror
    global x_Perror

    global y_diff_pre
    global y_Ierror
    global y_Derror
    global y_Perror

    global z_diff_pre
    global z_Ierror
    global z_Derror
    global z_Perror

    global done
    global collision
    # setts up initially x_diff  and y_diff value(will be change on each loop)


    # initial print of data
    print(f"the diff between uav and target is ({x_diff} , {y_diff} , {z_diff} )")

    if abs(x_diff) > xy_error or abs(y_diff) > xy_error or abs(z_diff) > z_error:
        print("loop entered")
        # gains

        #  x_diff is error
        x_Ierror = x_Ierror + x_diff * timepause
        x_Derror = (x_diff - x_diff_pre) / timepause
        #print(f"the x_Ierror is {x_Ierror} the x_Derror is {x_Derror} ")
        x_V = Kp * x_diff + Ki * x_Ierror + Kd * x_Derror

        #  y_diff is error
        y_Ierror = y_Ierror + y_diff * timepause
        y_Derror = (y_diff - y_diff_pre) / timepause
        #print(f"the y_Ierror is {y_Ierror} the y_Derror is {y_Derror} ")
        y_V = Kp * y_diff + Ki * y_Ierror + Kd * y_Derror

        #  x_diff is error
        z_Ierror = z_Ierror + z_diff * timepause
        z_Derror = (z_diff - z_diff_pre) / timepause
        #print(f"the x_Ierror is {x_Ierror} the x_Derror is {x_Derror} ")
        z_V = Kp * z_diff + Ki * z_Ierror + Kd * z_Derror

        print(f"the output velocity is ({x_V} , {y_V} , {z_V} )")#pid 输出的是x,y,z方向上的速度

        # velocity movement
        client.moveByVelocityAsync(x_V, y_V, z_V, timepause).join()
        state = client.getMultirotorState().kinematics_estimated
        collision = client.simGetCollisionInfo().has_collided
        print(f"after move the velocity of uav is {state.linear_velocity} ")
        print(f"after move the position of uav is {state.position}")
        #print(f"the linear_acceleration is {state.linear_acceleration}  adn the collosion is {collision}")

        # new positioning
        x_diff_pre = x_diff#保存这次的误差作为上一次的误差
        y_diff_pre = y_diff
        z_diff_pre = z_diff
    else:
        done = 0
    print(f"the collision is {collision}")
    if collision : #无人机碰撞到，认为结束
        done = 0
    #print(f"done in the PIDController is {done}")


if __name__ == '__main__':

    # pivals = np.arange(0,2.01,0.01)
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    client.takeoffAsync().join()
    client.moveToPositionAsync(0, 0, -10, 5).join()
    start_time = datetime.now()
    print("the start time is ",start_time)


    while(done):
        #print(f"done in the main is {done}")
        state = client.getMultirotorState().kinematics_estimated
        position = state.position
        velocity = state.linear_velocity
        x_pos = position.x_val
        y_pos = position.y_val
        z_pos = position.z_val

        x_record_list.append(x_pos)
        y_record_list.append(y_pos)
        z_record_list.append(z_pos)
        #print(f"the position of uav now is ({x_pos} , {y_pos} , {z_pos} )")

        pose_moving_platform = client.simGetObjectPose("Mobile_platform_2")
        #print(f"the pose is {pose_moving_platform}")
        x_poswanted = pose_moving_platform.position.x_val
        y_poswanted = pose_moving_platform.position.y_val
        z_poswanted = pose_moving_platform.position.z_val
        print(f"the position of moving_platform now is ({x_poswanted} , {y_poswanted} , {z_poswanted} )")

        x_diff = x_poswanted - x_pos
        y_diff = y_poswanted - y_pos
        z_diff = z_poswanted - z_pos

        #print(f"before the pid controller the velocity is {velocity}")
        PIDcontroller(x_diff, y_diff, z_diff)
        state_after_pid_controller = client.getMultirotorState().kinematics_estimated
        #time.sleep(2)

    end_time = datetime.now()
    cost_time = end_time - start_time
    print(f"the end time is {end_time} and all cost {cost_time}", )

    # #画图
    # fig = plt.figure()
    # ax1 = plt.axes(projection='3d')
    # # ax = fig.add_subplot(111,projection='3d')  #这种方法也可以画多个子图
    # z = np.linspace(-10, 0, 1000)
    # x = np.linspace(0,10,1000)
    # y = np.linspace(0,10,1000)
    # ax1.scatter3D(x_record_list, y_record_list, z_record_list, cmap='Blues')  # 绘制散点图
    # ax1.plot3D(x, y, z, 'gray')  # 绘制空间曲线
    # plt.xlabel('x (m)')
    # plt.ylabel('y (m)')
    # plt.title(f" p = {Kp} i = {Ki} d = {Kd} {len(x_record_list)} point the velocity is {state_after_pid_controller.linear_velocity},the last ({x_record_list[-1]},{y_record_list[-1]},{z_record_list[-1]} cost {cost_time})",
    #           fontdict={'weight': 'normal', 'size': 7})
    # plt.savefig("image/pid_landing_xyz_moving_target/01_00001_01.png")
    # plt.show()

