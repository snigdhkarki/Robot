import os
import numpy as np
import gym
from gym.wrappers.record_video import RecordVideo

ENV_NAME = 'BipedalWalker-v3'
env = gym.make(ENV_NAME, render_mode = "rgb_array")
def mkdir(base, name):
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path
videos_dir = mkdir('.', 'videos')
monitor_dir = mkdir(videos_dir, ENV_NAME)
if monitor_dir is not None:
    should_record = lambda i: record_video
    env = RecordVideo(env, monitor_dir, episode_trigger=should_record)
        



input_size = env.observation_space.shape[0] 
output_size = env.action_space.shape[0]
theta = np.zeros((output_size, input_size))
record_every=50
seed=1
np.random.seed(seed)
record_video = False

class Normalizer():
    def __init__(self, input_size):
        self.n = np.zeros(input_size)    #to make dynamic mean of all states till now
        self.mean = np.zeros(input_size) #dynamic mean of all state till now
        self.mean_diff = np.zeros(input_size) #needed to find standard dev(also dynamic(wut)) req to normalise
        self.var = np.zeros(input_size)   #needed to find standard dev req to normalise

    def normalise(self,state):
        self.n += 1.0
        last_mean = self.mean.copy() 
        self.mean += (state - self.mean) / self.n
        self.mean_diff += (state - last_mean) * (state - self.mean)
        self.var = (self.mean_diff / self.n).clip(min = 1e-2)
        std = np.sqrt(self.var)
        return (state - self.mean) / std

for game in range(1000):
    arrayofrandomdeltas = []
    arrayofthetaanddelta = []
    noise = 0.03

    for i in range(16):
        delta = np.random.randn(output_size, input_size)
        arrayofrandomdeltas.append([delta, (-1)*delta])
    for i in range(16):
        arrayofthetaanddelta.append([arrayofrandomdeltas[i][0]*noise + theta, arrayofrandomdeltas[i][1]*noise + theta])

    def playgame(thetaanddelta):
    # input: thetaanddelta
    # output: sum of reward when playing a game with that theta and delta
        normalizer = Normalizer(input_size)
        state = env.reset()[0]
        done = False
        num_plays = 0.0
        sum_rewards = 0.0
        episode_length = env.spec.max_episode_steps
        while not done and num_plays < episode_length:
            state = normalizer.normalise(state)        
            output = thetaanddelta.dot(state)
            state, reward, done, _, _ = env.step(output)
            reward = max(min(reward, 1), -1)
            sum_rewards += reward
            num_plays += 1
        return sum_rewards


    arrayofrewardsanddelta = []
    for i in range(16):
        reward_posi = playgame(arrayofthetaanddelta[i][0])
        reward_neg = playgame(arrayofthetaanddelta[i][1])
        arrayofrewardsanddelta.append([reward_posi, reward_neg, arrayofrandomdeltas[i][0]])


    
    num_best_deltas = 16
    learning_rate = 0.02
    positive_rewards = []
    negative_rewards = []
    for r_pos, r_neg, delta in arrayofrewardsanddelta:
        positive_rewards.append(r_pos)
        negative_rewards.append(r_neg)
    sigma_rewards = np.array(positive_rewards + negative_rewards).std()
    step = np.zeros((output_size, input_size))
    for r_pos, r_neg, delta in arrayofrewardsanddelta:
        step += (r_pos - r_neg) * delta
    theta += learning_rate / (num_best_deltas * sigma_rewards) * step
    if game % record_every == 0:
        record_video = True
    else:
        record_video = False


    print("the reward in game", game, "is" , playgame(theta))
    record_video = False
    


    





