import airsim
import setup_path
import time

#位置式PID: u(k) = Kp * e(k) + Ki * ∑e(i) + Kd * e(k) - e(k-1)
#e(k): 用户设定的值（目标值） -  控制对象的当前的状态值
#比例P :    e(k)
#积分I :   ∑e(i)  误差的累加
#微分D :  e(k) - e(k-1) 这次误差-上次误差

z_diff = 0
z_diff_pre = 0 #上一次的误差
z_Perror = 0
z_Ierror = 0
z_Derror = 0

timepause = 1 #无人机飞行时间
error = 0.2 #控制的误差允许范围
# official position wanted (will not be changed throughout)
z_Poswanted = 0 #想要到达z的位置

done = 1 #判断程序结束标志

def PIDcontroller(x_pos, y_pos, z_pos):
    global z_diff_pre
    global z_diff
    global z_Ierror
    global z_Derror
    global z_Perror
    global done

    # setts up initially Z_diff value(will be change on each loop)
    z_diff = z_Poswanted - z_pos #位置误差

    # initial print of data
    print("z_pos: ", z_pos)
    print("z_diff: ", z_diff)
    if z_diff > error:
        print("loop entered")
        # gains
        Kp = 1.2 #1.2最后没赚到地面 #0.1最后会撞到地面  # 100
        Ki = 0.01
        Kd = 0.001

        # BTW Z_diff is error
        z_Ierror = z_Ierror + z_diff * timepause
        z_Derror = (z_diff - z_diff_pre) / timepause
        print(f"the z_Ierror is {z_Ierror} the z_Derror is {z_Derror} the ")
        z_V = Kp * z_diff + Ki * z_Ierror + Kd * z_Derror
        print("z_V", z_V)#pid 输出的是z方向上的速度
        # velocity movement
        client.moveByVelocityAsync(0, 0, z_V, timepause).join()

        # new positioning
        z_diff_pre = z_diff#保存这次的误差作为上一次的误差
    else:
        done = 0


if __name__ == '__main__':

    # pivals = np.arange(0,2.01,0.01)
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    client.takeoffAsync().join()
    client.moveToPositionAsync(0, 0, -20, 5).join()

    while(done):
        position = client.getMultirotorState().kinematics_estimated.position
        velocity = client.getMultirotorState().kinematics_estimated.linear_velocity
        collosion = client.simGetCollisionInfo().has_collided
        X_pos = position.x_val
        Y_pos = position.y_val
        Z_pos = position.z_val
        PIDcontroller(X_pos, Y_pos, Z_pos)
        print(f"the velocity is {velocity} and the collosion is {collosion}")
        time.sleep(2)

    #rospy.spin()
#