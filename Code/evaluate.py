#   Proj:   TFG - Evaluating RL algorithms and policies in gFootbal enviroment
#   File:   evaluate.py
#   Desc:   Loads trained models 1-30 of a specific fase and writes down their accuracies after 1000 episodes
#   Auth:   Enrique Boya Falcón
#   Date:   2021

import os
import sys
import getopt

import csv

import gfootball.env as football_env

from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3 import A2C


def parse_path(path: str):
    #   *path*/<scenario> <model_type> <rewards> <agent_id>.zip
    try:
        parse = path[:-3]           # remove '.zip'
        parse = parse.split('/')
        parse = parse[-1]
        parse = parse.split(' ')
        parse.pop()                 # remove agent id
        scenario, model_type, rewards = parse
    except:
        print('Error: Incorrect path')
        sys.exit(2)
    
    if rewards == 'scr+chp':
        rewards = 'scoring,checkpoints'

    return scenario, model_type, rewards

# ------------------------------------------------------------------------------

#   Loads a model given its path, runs an ammount of episodes, returns accuracy
def test(path: str, episodes: int, render: bool, verbose: bool):

    scenario, model_type, rewards = parse_path(path)

    env = football_env.create_environment(env_name=scenario,
                                        stacked=False, 
                                        representation='simple115v2',
                                        logdir='/tmp/football', 
                                        rewards=rewards,
                                        write_goal_dumps=False,
                                        write_full_episode_dumps=False, 
                                        render=render)

    if   model_type == 'PPO':   model = PPO.load(path, env=env)

    elif model_type == 'DQN':   model = DQN.load(path, env=env)
        
    elif model_type == 'A2C':   model = A2C.load(path, env=env)

    print('Testing: {}'.format(path))

    total=0
    for episode in range(1, episodes+1):
        obs = env.reset()
        done = False
        score = 0

        while not done:
            action, _ = model.predict(obs)
            obs, rew, done, info = env.step(action)
            score += rew

        if rewards == 'scoring,checkpoints':
            if score>=2:
                total+=1
        else:
            if score>=1:
                total+=1

        if verbose:
            print('Episode: {}  Score: {:.2f}'.format(episode, score))
    
    accuracy = total/episodes

    print('Accuracy: {}%'.format(accuracy*100))
    
    env.close()

    return accuracy


# ---------------------------


# accuracy column, algorithm column
def main(argv):
    
    phase = 2   # 0-Penalty 1-CounterAttack 2-Iniesta

    episodes = 1000

    scenario = ['MyScenarios_Penalty','MyScenarios_Counterattack','MyScenarios_Iniesta']
    algorithm = ['PPO','DQN','A2C']
    politic = ['scoring','scr+chp']

    # Penalty
    if phase == 0:
        output = '../Data/Results/Penalty.csv'
        with open(output, 'w', newline='') as csvfile:

            writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

            writer.writerow(algorithm)

            # For every agent_id (row)
            for i in range(1,31):

                accuracy = [0] * len(algorithm)

                # For every politic (column)
                for j in range(len(algorithm)):

                    path = '../Data/Models/{} {} {} {:02d}.zip'.format(scenario[phase], algorithm[j], politic[0], i)

                    accuracy[j] = test(path=path, episodes=episodes, render=False, verbose=False)

                writer.writerow(accuracy)

    # CounterAttack
    elif phase == 1:
        output = '../Data/Results/Counterattack.csv'
        with open(output, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

            writer.writerow(politic)

            # For every agent_id (row)
            for i in range(1,31):   

                accuracy = [0] * len(politic)

                # For every politic (column)
                for j in range(len(politic)):

                    path = '../Data/Models/{} {} {} {:02d}.zip'.format(scenario[phase], algorithm[0], politic[j], i)

                    accuracy[j] = test(path=path, episodes=episodes, render=False, verbose=False)

                writer.writerow(accuracy)

    # Iniesta
    elif phase == 2:
        output = '../Data/Results/Iniesta.csv'
        with open(output, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

            writer.writerow(politic)

            # For every agent_id (row)
            for i in range(1,31):   

                accuracy = [0] * len(politic)

                # For every politic (column)
                for j in range(len(politic)):

                    path = '../Data/Models/{} {} {} {:02d}.zip'.format(scenario[phase], algorithm[0], politic[j], i)

                    accuracy[j] = test(path=path, episodes=episodes, render=False, verbose=False)

                writer.writerow(accuracy)

    else:
        assert False, "unhandled option"


if __name__ == "__main__" :
    main(sys.argv[1:])