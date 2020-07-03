import torch
import gym
import math
import matplotlib.pyplot as plt
import numpy as np
import DQNAgent
import pickle

####################################################################################################
#################################### ENVIRONEMENT CONFIGURATION ####################################
####################################################################################################

# Import the created traffic environment
if 'necTraffic-v11' in gym.envs.registry.env_specs:
    del gym.envs.registry.env_specs['necTraffic-v11']

    import nec_traffic_v11

    gym.register(
        id='necTraffic-v11',
        entry_point='nec_traffic_v11.envs:NECTraffic',
    )
else:
    import nec_traffic_v11

# Defining the environment variable
env = gym.make('necTraffic-v11')

# Determine algorithm seed for proper randomization
seed_value = 91
env.seed(seed_value)
torch.manual_seed(seed_value)

# Plot settings
plt.style.use('ggplot')
plot_dict = {1: 'Rewards', 2: 'Flickers', 3: 'Queue'}

# Weighted average plot settings
beta_init  = 0.85
beta_final = 0.999
beta_decay = 500

plot_freq = 1000 # Frequency of saving plots

####################################################################################################
############################################ PARAMETERS ############################################
####################################################################################################

num_episodes = 100000 # Total number of episodes

# e-greedy configuration
egreedy_init       = 1
egreedy_final      = 0.05
egreedy_decay      = 15000

# Choose traffic control method
RL         = 1
Fixed_time = 0
SOTL       = 0

# If "RL" is chosen, activate desired techniques by changing 0 to 1
double_dqn  = 0 # Double Q-Learning
dueling_dqn = 0 # Dueling Networks
per         = 0 # Prioritized Experience Replay
noisy_dqn   = 0 # Noisy Networks
dist_dqn    = 0 # Distributional RL

options = [double_dqn, dueling_dqn, per, noisy_dqn, dist_dqn]
options_list = ["double", "dueling", "per", "noisy", "dist"]

# Trun on if you wish to continue previously stopped training
resume_previous_train = 0

####################################################################################################
############################################ Functions #############################################
####################################################################################################

def find_plot_avg_param(steps_done):
    """ function to plot the weighted average - beta = decaying parameter """
    beta = beta_final + (beta_init - beta_final) * math.exp(-1 * steps_done / beta_decay)
    return beta

def calculate_epsilon(steps_done):
    """ e-greedy exploration decay equation """
    eps = egreedy_final + (egreedy_init - egreedy_final) * math.exp(-1 * steps_done / egreedy_decay)
    return eps

def save_plot_info():
    """ function to save trajectory data """
    reward_record.append(cum_reward)
    flicker_record.append(info[1])
    queue_record.append(info[2][0] + info[2][1])
    traffic_record.append(info[3][0] + info[3][1])

    r1 = reward_average_plot[-1] * beta
    r1 += (1 - beta) * cum_reward
    reward_average_plot.append(r1)

    r2 = flicker_average_plot[-1] * beta
    r2 += (1 - beta) * info[1]
    flicker_average_plot.append(r2)

    r3 = wait_average_plot[-1] * beta
    r3 += (1 - beta) * (info[2][0] + info[2][1])
    wait_average_plot.append(r3)

    list_to_save = [
                      [reward_record,  reward_average_plot],
                      [flicker_record, flicker_average_plot],
                      [queue_record,   wait_average_plot],
                      [traffic_record, traffic_plot]
                   ]

    if folder == "RL":
        if all(options):
            PATH = "Plots/" + folder + "/all/data_all.txt"
        elif not any(options):
            PATH = "Plots/" + folder + "/vanilla_dqn/data_vanilla_dqn.txt"
        else:
            zero_idx = list(options).index(0)
            PATH = "Plots/" + folder + "/no_" + options_list[zero_idx] + "_dqn/data_no_" + options_list[zero_idx] + "_dqn.txt"
    else:
        PATH = "Plots/" + folder + "/data_{}".format(folder) + ".txt"

    with open(PATH, "wb") as file:
        pickle.dump(list_to_save, file)

def plot_snapshot():
    """ function to draw plots in the corresponding folders """
    plt.figure(figsize=(20, 15))

    plt.subplot(311)
    plt.ylabel('Rewards', fontsize=16)
    plt.plot(reward_record, alpha=0.15, color='red')
    plt.plot(reward_average_plot, alpha=0.9, color='red', linewidth=3.0)
    plt.xlim(-0.5, i_episode)
    plt.ylim(-10, 3)

    plt.subplot(312)
    plt.ylabel('Number of Flickers', fontsize=16)
    plt.plot(flicker_record, alpha=0.15, color='green')
    plt.plot(flicker_average_plot, alpha=0.9, color='green', linewidth=3.0)
    plt.xlim(-0.5, i_episode)
    plt.ylim(-0.2, 25)

    plt.subplot(313)
    plt.ylabel('Wait Time', fontsize=16)
    plt.plot(queue_record, alpha=0.15, color='blue')
    plt.plot(wait_average_plot, alpha=0.9, color='blue', linewidth=3.0)
    plt.xlim(-0.5, i_episode)
    plt.ylim(-0.5, 9000)

    if folder == "RL":
        if all(options):
            PATH = "Plots/" + folder + "/all" + "/{}.png".format('plot_all')
        elif not any(options):
            PATH = "Plots/" + folder + "/vanilla_dqn" + "/{}.png".format('plot_vanilla_dqn')
        else:
            zero_idx = list(options).index(0)
            PATH = "Plots/" + folder + "/no_" + options_list[zero_idx] + "_dqn/plot_no_" + options_list[zero_idx] + "_dqn.png"
    else:
        PATH = "Plots/" + folder + "/plot_{}".format(folder) + ".png"

    plt.savefig(PATH)
    plt.close()

####################################################################################################
############################################ MAIN FILE #############################################
####################################################################################################

if __name__ == "__main__":

    qnet_agent   = DQNAgent.DQN(options, resume_previous_train) # Define the RL agent

    # Initialize the episode
    frames_total = 0
    done  = 0
    action = 0
    counter = 0

    gui   = 0 # Set to 1 if you like to see the episode steps in SUMO graphical interface

    # Determine the folder to save the data and plots
    if RL:
        folder = "RL"
    if Fixed_time:
        folder = "Fixed Time"
    elif SOTL:
        folder = "SOTL"

    if resume_previous_train:
        if RL:
            if all(options):
                PATH = "Plots/" + folder + "/all/data_all.txt"
            elif not any(options):
                PATH = "Plots/" + folder + "/vanilla_dqn/data_vanilla_dqn.txt"
            else:
                zero_idx = list(options).index(0)
                PATH = "Plots/" + folder + "/no_" + options_list[zero_idx] + "_dqn/data_no_" + options_list[zero_idx] + "_dqn.txt"
        else:
            PATH = "Plots/" + folder + "/data_" + folder + ".txt"

        with open(PATH, "rb") as file:
            data = pickle.load(file)

        from_episode = len(data[0][0]) - 1 # If resuming the learning, start from the last episode

        egreedy_init = egreedy_final + (egreedy_init - egreedy_final) \
                            * math.exp(-1 * from_episode / egreedy_decay) # e-greedy update
        beta_init = beta_final + (beta_init - beta_final) \
                            * math.exp(-1 * from_episode / beta_decay) # Weighted average plot update

        reward_record  , reward_average_plot  = data[0][0][0:from_episode], data[0][1][0:from_episode]
        flicker_record , flicker_average_plot = data[1][0][0:from_episode], data[1][1][0:from_episode]
        queue_record   , wait_average_plot    = data[2][0][0:from_episode], data[2][1][0:from_episode]
        traffic_record , traffic_plot         = data[3][0][0:from_episode], data[3][1][0:from_episode]
    else:
        from_episode = 1

        reward_record, reward_average_plot   = [], []
        flicker_record, flicker_average_plot = [], []
        queue_record, wait_average_plot      = [], []
        traffic_record, traffic_plot         = [], []

    # Training Phase
    for i_episode in range(from_episode, num_episodes+1):

        state = env.reset(gui, folder, options)

        frames_total += 1
        cum_reward = 0

        beta = find_plot_avg_param(frames_total)
        epsilon = calculate_epsilon(frames_total)

        while True:
            if RL:
                action = qnet_agent.act(state, epsilon)
                new_state, reward, done, info = env.step(action)
                experience = state, action, new_state, reward, done
                qnet_agent.optimize(experience)
                state = new_state
                cum_reward += reward

            elif Fixed_time:
                if   counter % 6 in [0,1]: action = 0
                elif counter % 6 in [2]:   action = 1
                elif counter % 6 in [3,4]: action = 2
                elif counter % 6 in [5]:   action = 3
                new_state, reward, done, info = env.step(action)
                state = new_state
                cum_reward += reward
                counter += 1

            elif SOTL:
                new_state, reward, done, info = env.step(action)
                state = new_state
                cum_reward += reward
                if   info[4][1][0] + info[4][3][0] > 10:  action = 0
                elif info[4][1][1] + info[4][3][1] >  5:  action = 1
                elif info[4][0][0] + info[4][2][0] >  5:  action = 2
                elif info[4][0][1] + info[4][2][1] >  5:  action = 3

            # Output Print
            if done:
                print("\n==========================")
                print("    Episode {} ".format(i_episode))
                print("--------------------------")
                print("# of Flickers:   {}".format(info[1]))
                print("Queue N-S:       {}".format(info[2][0]))
                print("Queue E-W:       {}".format(info[2][1]))
                print("Average Reward:  {}".format(round(cum_reward, 2)))
                print("--------------------------")
                if not noisy_dqn:
                    print("Epsilon:         {}".format(np.round(epsilon, 2)))
                print("==========================\n\n")

                if i_episode == 1:
                    reward_average_plot.append(cum_reward)
                    flicker_average_plot.append(12)
                    wait_average_plot.append(info[2][0] + info[2][1])

                    reward_record.append(cum_reward)
                    flicker_record.append(12)
                    queue_record.append(info[2][0] + info[2][1])
                    traffic_record.append(info[3][0] + info[3][1])
                else:
                    save_plot_info()
                    if i_episode % plot_freq == 0:
                        plot_snapshot()

                break
