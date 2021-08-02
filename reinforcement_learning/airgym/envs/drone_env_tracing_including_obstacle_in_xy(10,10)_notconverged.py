import math

import setup_path
import airsim
import numpy as np
import math
import time
from argparse import ArgumentParser

import gym
from gym import spaces
from airgym.envs.airsim_env import AirSimEnv
from datetime import datetime


class AirSimDroneEnv(AirSimEnv):
    def __init__(self, ip_address, step_length, image_shape):
        super().__init__(image_shape)
        self.step_length = step_length
        self.image_shape = image_shape

        self.state = {
            "position": np.zeros(3),
            "collision": False,
            "prev_position": np.zeros(3),
        }
        self.dist_pre = 15
        self.drone = airsim.MultirotorClient(ip=ip_address)
        self.action_space = spaces.Discrete(7)#这里把动作设置成离散的7个动作 应该要改这里
        self._setup_flight()

        self.image_request = airsim.ImageRequest(
            3, airsim.ImageType.DepthPerspective, True, False
        )

    def __del__(self):
        self.drone.reset()

    def _setup_flight(self):
        self.drone.reset()
        self.drone.enableApiControl(True)
        self.drone.armDisarm(True)
        #定义每个回合的开始时间，
        global start_episode_time
        start_episode_time = datetime.now()
        #global dist_pre
        print("the episode start time is on ",start_episode_time)
        #print(f"the dist_pre is {dist_pre}")
        # Set home position and velocity
        self.drone.moveToPositionAsync(0,0,-4,10).join()

    def transform_obs(self, responses):
        img1d = np.array(responses[0].image_data_float, dtype=np.float)
        img1d = 255 / np.maximum(np.ones(img1d.size), img1d)
        img2d = np.reshape(img1d, (responses[0].height, responses[0].width))

        from PIL import Image

        image = Image.fromarray(img2d)
        im_final = np.array(image.resize((84, 84)).convert("L"))

        return im_final.reshape([84, 84, 1])

    def _get_obs(self):
        responses = self.drone.simGetImages([self.image_request])
        image = self.transform_obs(responses)
        self.drone_state = self.drone.getMultirotorState()

        self.state["prev_position"] = self.state["position"]
        self.state["position"] = self.drone_state.kinematics_estimated.position
        self.state["velocity"] = self.drone_state.kinematics_estimated.linear_velocity

        collision = self.drone.simGetCollisionInfo().has_collided
        self.state["collision"] = collision

        return image

    def _do_action(self, action):
        quad_offset = self.interpret_action(action)
        quad_vel = self.drone.getMultirotorState().kinematics_estimated.linear_velocity
        # print("the velocity of vx,vy,vz is ",quad_vel)
        print("the quad_offset is ",quad_offset)
        #在这里可以获取无人机的高度 然后加几个if 来判断 ：
        # 比如 无人机高度=20米，就只在x-y平面飞，这时判断下x-y的位置，比如无人机离目标附近5米了，开启下降阶段，只在z方向输出，
        #判断无人机高度，当无人机高度到达10米左右，关掉z方向上的速度，使无人机在x-y平面上接近目标，
        #这时判断下x - y的位置，比如无人机离目标附近2米了，开启下降阶段 只允许无人机在z轴上的速度进行输出

        #print("the move velocity is ",quad_vel.x_val+quad_offset[0],quad_vel.y_val+quad_offset[1],quad_vel.z_val+quad_offset[2])
        #这里通过加上quad_offset[]来改变无人机在vx,vy,vz上的速度，使其往新的方向飞
        self.drone.moveByVelocityAsync(
            quad_vel.x_val + quad_offset[0],
            quad_vel.y_val + quad_offset[1],
            #quad_vel.z_val + quad_offset[2],
            -0.1,
            3,
            #5,
        ).join() # 这里持续时间是5秒 那速度不应该设计太大，我在想，能不能前期可以把速度弄大或者持续时间变大，后期接近了以后把速度变小，这个想法很重要
        #仔细一想，速度不应该过(无人机的速度每次都是从0开始的，突然加的偏移值不能太大)，会考虑到无人机的漂移问题，因此可以减小速度，使得持续时间变大

    #这个是无人机飞向（10，10）静止目标，加入几个障碍物，仅仅是x-y平面
    @property
    def _compute_reward(self):
        destination = [10, 10,]
        done = 0 #done = 1 即回合结束,两种情况回合结束,无人机碰撞,无人机接近目标位置
        #第一步先算出x,y,z,并计算是否撞击了
        drone_x = self.state["position"].x_val
        drone_y = self.state["position"].y_val
        drone_z = self.state['position'].z_val
        print("the x,y,z is (", drone_x, drone_y, drone_z, " )")
        print("the state of collision is ",self.state["collision"])
        dist_now = math.sqrt(pow(10 - drone_x, 2) + pow(10 - drone_y, 2))
        if (drone_x - 10 >= 3) or (drone_y - 10 >= 3 ) or (drone_x < -10) or (drone_y < -10) or drone_z < -10:
            reward_dist = 300 # x,y超过目标1米,x运动范围[-10,11],y运动范围[-10,11]
        else:
            print(f"the dist_now is {dist_now}  the dist_pre is {self.dist_pre}")
            if dist_now < 2:
                #只要距离小于2m,即认为无人机离以目标点为圆心，2米为半径的地方,是考虑到无人机半径1米，而且被最终物体应该也有一定半径就认为到达目标点
                file_name = 'record/dqn_drone/including_obstacle_in_x_y.txt'
                reward_dist = -10  # 认为已经到达,则给奖励函数很大的值以鼓励去执行更多这样的动作
                with open(file_name, 'a') as file_obj:
                    file_obj.write("the close x,y is:( ")
                    file_obj.write(str(drone_x))
                    file_obj.write(",")
                    file_obj.write(str(drone_y))
                    file_obj.write(",")
                    file_obj.write(str(drone_z))
                    file_obj.write(" ) at ")
                    file_obj.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                    file_obj.write("\n")
                print(" i think the uav is closing to the position ")
                done = 1    #无人机到达目标位置附近,可认为回合结束
            elif self.state["collision"]:
                reward_dist= 300 #碰撞则回合结束
            elif dist_now < self.dist_pre :
                #无人机相比上一步离目标更近了
                reward_dist = dist_now - 5
            else:
                reward_dist = dist_now

        self.dist_pre = dist_now #把这次的距离记录下来
        reward = -10*reward_dist
        #print("the total reward is ",reward)

        # 碰撞(reward_dist = -20 ，或者计算到距离是远离目标的(reward_dist < -14.142(14.142是无人机初始位置离目标的距离))，即认为这回合结束
        #这里不能把无人机往反方向就立马停止，这样无人机到后期会一直往反向飞，从而结束回合使得出现问题，具体为什么目前还不知道？？？？
        if reward <= -282.8:
            done = 1
        if done == 1 :
            self.dist_pre = 15
        return reward, done

    def step(self, action):
        self._do_action(action)
        obs = self._get_obs()
        reward, done = self._compute_reward

        return obs, reward, done, self.state

    def reset(self):
        self._setup_flight()
        return self._get_obs()

    def interpret_action(self, action):
        if action == 0:
            #往vx反向飞
            quad_offset = (self.step_length, 0, 0)
        elif action == 1:
            #往vy方向飞
            quad_offset = (0, self.step_length, 0)
        elif action == 2:
            #往vz方向飞
            #quad_offset = (0, 0, self.step_length)
            quad_offset = (0, 0, 0)
        elif action == 3:
            #往-vx方向飞
            quad_offset = (-self.step_length, 0, 0)
        elif action == 4:
            #往-vy方向飞
            quad_offset = (0, -self.step_length, 0)
        elif action == 5:
            #往-vz方向飞
            #quad_offset = (0, 0, -self.step_length)
            quad_offset = (0, 0, 0)#更改为了避免无人机总是输出（0.3，0.3，0）因为这个动作的概率被放大了
        else:
            #盘旋不动
            quad_offset = (0.3, 0.3, 0) #设置为(0.3,0.3,0.3)的时候，无人机飞向静止目标(10,10)大约3000step可以收敛，飞向移动目标(10,10)vx=0.1,vy = 0.1 大约3000step 可以收敛
            #quad_offset = (0, 0, 0)

        return quad_offset
