import setup_path
import gym
import airgym
import time
import os

from stable_baselines3 import DQN, PPO
from stable_baselines3 import DDPG
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CheckpointCallback1

# Create a DummyVecEnv for main airsim gym env
env = DummyVecEnv(
    [
        lambda: Monitor(
            gym.make(
                "airgym:airsim-drone-sample-v0",
                ip_address="127.0.0.1",
                #step_length=0.25,   #用来控制无人机的各个方向新增的速度
                step_length = 0.3, #不能太大了
                image_shape=(84, 84, 1),
            )
        )
    ]
)

# Wrap env as VecTransposeImage to allow SB to handle frame observations
env = VecTransposeImage(env)

# Initialize RL algorithm type and parameters
dir = "D:/software/airsim/AirSim/PythonClient/reinforcement_learning/model_store/ddpg/ddpg_drone/best_model"

callbacks = []
eval_callback = EvalCallback(
    env,
    callback_on_new_best=None,
    n_eval_episodes=5,
    best_model_save_path="./model_store/ddpg/ddpg_drone/",
    log_path="./model_store/ddpg/ddpg_drone/",
    eval_freq=500,
)
checkpoint_callback = CheckpointCallback(save_freq=500, save_path='./model_store/ddpg/ddpg_drone_step/',
                                         name_prefix='ddpg_drone_model')
checkpoint_callback1 = CheckpointCallback1(save_freq=500, save_path='./model_store/ddpg/ddpg_drone_step/',
                                         name_prefix='ddpg_airsim_drone_buffer')
callbacks.append(eval_callback)
callbacks.append(checkpoint_callback)
callbacks.append(checkpoint_callback1)

kwargs = {}
kwargs["callback"] = callbacks

file_name = 'record/ddpg_drone/recording_closing_position.txt'
with open(file_name, 'a') as file_obj:
    file_obj.write("\nthe new train starts at : ")
    file_obj.write(time.strftime("%Y-%m-%d %H:%M:%S\n", time.localtime()))

if os.path.exists(dir+".zip"):
    print("已有模型，正在加载模型")
    model = DDPG.load("./model_store/dqn_drone/best_model",env,device="cuda")
    #model.load_replay_buffer("./model_store/dqn_drone_step/dqn_airsim_drone_buffer1_2500_steps")
    print(f"the load buffer has {model.replay_buffer.size()} trsansition ")
    print("111")
    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        #env.render()
    # model.learn(
    #     total_timesteps=2000,
    #     tb_log_name="dqn_airsim_drone_run_" + str(time.time()),
    #     **kwargs
    # )
    # # Save policy weights
    print("in the dqn_drone the buffer size is ", model.replay_buffer.size())

else:
    print("没有已有模型，重新开始训练")
    #mode = DDPG()
    model = DDPG(
        "CnnPolicy", #使用图片作为输入
        env,
        verbose=1,
        device="cuda",
        tensorboard_log="./tb_logs/",
    )
    #Train for a certain number of timesteps
    model.learn(
        total_timesteps=3000,#原来设计的是5*10^5
        tb_log_name="ddpg_airsim_drone_run_" + str(time.time()),
        **kwargs
    )
    print("the buffer size is ",model.replay_buffer.size())

with open(file_name, 'a') as file_obj:
    file_obj.write("the new train ends at : ")
    file_obj.write(time.strftime("%Y-%m-%d %H:%M:%S\n", time.localtime()))


