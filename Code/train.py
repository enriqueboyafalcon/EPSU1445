#   Proj:   TFG - Evaluating RL algorithms and policies in gFootbal enviroment
#   File:   train.py
#   Desc:   Trains 30 agents with the specified algorithm and policy in the given scenario
#   Auth:   Enrique Boya Falcón
#   Date:   2021

import os
import sys
import numpy as np

import gfootball.env as football_env

from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import BaseCallback

from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy


log_dir = os.path.join('Logs')

#   Callback called every check_freq steps
#   If the current model has a new best accuracy, it saves the model
#   If it has not achieved a new best mean reward in the last stag_limit checks, it ends the training
class CustomCallback(BaseCallback):

    def __init__(self, check_freq: int, log_dir: str, name: str, checkpoints: bool, verbose=1):
        super(CustomCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = name
        self.best_mean_reward = -np.inf
        self.best_accuracy = -np.inf
        self.stagnation = 0
        self.checkpoints = checkpoints

    def _on_step(self) -> bool:
        stag_limit = 15 # number of oportunities to get a high score before ending training
        episodes = 100  # number of episodes taken into account when calculating mean reward and accuracy

        if self.n_calls % self.check_freq == 0:

            x, y = ts2xy(load_results(self.log_dir), 'timesteps')

            if len(x) > 0:

                list = y[-episodes:]

                mean_reward = np.mean(list)

                if self.checkpoints: accuracy = (list == 2).sum()
                else:                accuracy = (list == 1).sum()
                accuracy = (accuracy/episodes)*100

                # if better acc, save model
                # if better rew, reset stag counter

                if accuracy > self.best_accuracy:
                    self.best_accuracy = accuracy
                    self.model.save(os.path.join("Models", self.save_path))

                if mean_reward > self.best_mean_reward:
                    self.stagnation = 0
                    self.best_mean_reward = mean_reward
                elif self.best_accuracy > 0:    # only start counting down after saving 1 best model
                    self.stagnation += 1
                elif self.num_timesteps > 500000:     # if 0 acc and isnt gonna learn, end training
                    return False
                
                if self.stagnation >= stag_limit:
                    return False

                if self.verbose > 0:
                    agent_id = self.save_path.split(' ')
                    agent_id = agent_id[len(agent_id)-1]
                    print("agent_{} {:7d}ts | mean rew:{:5.2f} | best rew:{:5.2f} | acc:{:6.2f} | best acc:{:6.2f} | stag: {:2d}"
                            .format(agent_id,
                                    self.num_timesteps, 
                                    mean_reward,
                                    self.best_mean_reward,
                                    accuracy,
                                    self.best_accuracy, 
                                    stag_limit - self.stagnation))


# Creates the enviroment and trains an agent with the specified params
# Saves monitor logs in /Logs
# Saves best model in /Models
# Output files naming format: "<scenario> <algorithm> <politic> <agent id>.csv/zip"
def train_agent(agent_id: str, checkpoints: bool, scenario: str, model: str, timesteps: int, check_freq: int):

    output_file_name = '{} {} '.format(scenario, model)
    if checkpoints:
        output_file_name+='scr+chp'
        rewards='scoring,checkpoints'
    else:
        output_file_name+='scoring'
        rewards='scoring'

    output_file_name+= ' {}'.format(agent_id)

    env = football_env.create_environment(env_name=scenario, 
                                        stacked=False, 
                                        representation='simple115v2',
                                        logdir='/tmp/football', 
                                        rewards=rewards,
                                        write_goal_dumps=False,
                                        write_full_episode_dumps=False, 
                                        render=False)

    env = Monitor(env, log_dir)

    if   model == 'PPO': model = PPO('MlpPolicy', env, verbose=0)

    elif model == 'DQN': model = DQN('MlpPolicy', env, verbose=0)

    elif model == 'A2C': model = A2C('MlpPolicy', env, verbose=0)

    else: assert False, "Unhandled model"

    callback = CustomCallback(check_freq=check_freq, log_dir=log_dir, name=output_file_name, checkpoints=checkpoints)

    model.learn(total_timesteps=timesteps, callback=callback)

    env.close()

    os.rename(os.path.join('Logs','monitor.csv'),os.path.join('Logs',output_file_name+'.csv'))


def main(argv):

    scenario = 'MyScenarios_Penalty' # MyScenarios_Penalty/Counterattack/Iniesta
    model = 'PPO'                    # A2C/DQN/PPO
    checkpoints = False              # False/True

    first_agent = 1
    last_agent  = 30

    if checkpoints:
        print('\nTraining {} agents {}..{} on {} with checkpoints\n'.format(model, first_agent, last_agent, scenario))
    else:
        print('\nTraining {} agents {}..{} on {} without checkpoints\n'.format(model, first_agent, last_agent, scenario))

    for x in range(first_agent, last_agent+1):

        train_agent(agent_id = '{:02d}'.format(x),
                    checkpoints = checkpoints,
                    scenario = scenario,
                    model = model,
                    timesteps = 1500000,
                    check_freq =  10000)

        print('agent_{} has been trained.\n'.format(x))

    print('All {} agents have been trained.'.format(last_agent-first_agent+1))



if __name__ == "__main__" :
    main(sys.argv[1:])