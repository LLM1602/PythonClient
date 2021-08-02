import setup_path
import gym
import airgym
import time
import os

from stable_baselines3 import DQN
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
                step_length = 0.25, #不能太大了
                image_shape=(84, 84, 1),
            )
        )
    ]
)

# Wrap env as VecTransposeImage to allow SB to handle frame observations
env = VecTransposeImage(env)

# Initialize RL algorithm type and parameters
dir = "E:/PyCharmProject/llm_AirSim_github_modified/PythonClient/reinforcement_learning/model_store/dqn_drone/best_model_for_moving_converged"

callbacks = []
eval_callback = EvalCallback(
    env,
    callback_on_new_best=None,
    n_eval_episodes=5,
    best_model_save_path="./model_store/dqn_drone/",
    log_path="./model_store/dqn_drone/",
    eval_freq=500,
)
checkpoint_callback = CheckpointCallback(save_freq=500, save_path='./model_store/dqn_drone_step/',
                                         name_prefix='dqn_tracing_moving_model_in_xy')
checkpoint_callback1 = CheckpointCallback1(save_freq=500, save_path='./model_store/dqn_drone_step/',
                                         name_prefix='dqn_tracing_moving_model_in_xy_buffer')
callbacks.append(eval_callback)
callbacks.append(checkpoint_callback)
callbacks.append(checkpoint_callback1)

kwargs = {}
kwargs["callback"] = callbacks

file_name = 'record/dqn_drone/recording_closing_position_moving.txt'
with open(file_name, 'a') as file_obj:
    file_obj.write("\nthe new train starts at : ")
    file_obj.write(time.strftime("%Y-%m-%d %H:%M:%S\n", time.localtime()))

if os.path.exists(dir+".zip"):
    print("已有模型，正在加载模型")
    model = DQN.load("./model_store/dqn_drone/best_model_for_moving_converged",env,device="cuda")
    #model.load_replay_buffer("./model_store/dqn_drone_step/11dqn_avoid_model_in_xy_buffer_2000_steps")
    #print(f"the load buffer has {model.replay_buffer.size()} trsansition ")
    print("111")
    # model.learn(
    #     total_timesteps=3000,
    #     tb_log_name="dqn_airsim_drone_run_" + str(time.time()),
    #     **kwargs
    # )
    # Save policy weights
    #predict 是直接加载好的模型 演示 不用继续训练了
    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
    env.render()
    print("in the dqn_drone the buffer size is ", model.replay_buffer.size())

else:
    print("没有已有模型，重新开始训练")
    #mode = DDPG()
    model = DQN(
        "CnnPolicy", #使用图片作为输入
        env,
        learning_rate=0.00025,#莫凡代码的学习率是0.01 这个怎么确定最好的呢？
        verbose=1,
        batch_size=32,#每个梯度更新的小批量大小，应该是与tensorflow有关
        train_freq=4,#每train_freq一步更新模型。或者，传递频率和单位的元组
        target_update_interval=500,#200应该效果不如500，下次改一改#10000,#在每个target_update_interval 环境步骤更新目标网络，原来是10000     这个我得改
        learning_starts=500, #2#10000,在学习开始之前收集模型的多少步骤的转换 应该是要先存储10000个元组到经验池中     这个我得改
        buffer_size=500000,#500000,#经验池的大小
        max_grad_norm=5, #梯度裁剪的最大值
        exploration_fraction=0.1,#探索率降低的整个训练期的分数??
        exploration_final_eps=0.1,#0.01,#随机动作概率的最终值0.01
        device="cuda",
        tensorboard_log="./tb_logs/",
    )
    #Train for a certain number of timesteps
    model.learn(
        total_timesteps=3000,#原来设计的是5*10^5
        tb_log_name="dqn_airsim_drone_run_" + str(time.time()),
        **kwargs
    )
    # Save policy weights
    print("the buffer size is ",model.replay_buffer.size())

with open(file_name, 'a') as file_obj:
    file_obj.write("the new train ends at : ")
    file_obj.write(time.strftime("%Y-%m-%d %H:%M:%S\n", time.localtime()))


