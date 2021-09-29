#   Proj:   TFG - Evaluating RL algorithms and policies in gFootbal enviroment
#   File:   test.py
#   Desc:   Loads a given trained model and runs the desired ammount of episodes, with or without rendering
#   Auth:   Enrique Boya Falcón
#   Date:   2021

import os
import sys
import getopt

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

    print('Accuracy: {}%'.format((total/episodes)*100))

    env.close()

# ---------------------------

def main(argv):

    path = ''
    render = False
    verbose = False
    episodes = 100

    try:
        opts, args = getopt.gnu_getopt(
            argv,
            'hp:rve:',
            ['help','path=','render','verbose','episodes='])
    except getopt.GetoptError:
        print('Usage: test.py path=<Model_Path> -opts')
        print('Use -h for option list')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print('Options:')
            print('    Episodes  -e <Num>    --episodes <Num>    default: 1000')
            print('    Render    -r          --render')
            print('    Verbose   -v          --verbose')
            sys.exit()
        elif opt in ('-e', '--episodes'):
            episodes = arg
        elif opt in ('-p', '--path'):
            path = arg
        elif opt in ('-r', '--render'):
            render = True
        elif opt in ('-v', '--verbose'):
            verbose = True
        else:
            assert False, "unhandled option"
    
    if path == '':
        print('Error: Incorrect path')
        print('Usage: test.py path=<Model_Path> -opts')
        print('Use -h for option list')
        sys.exit(2)
    
    try:
        episodes = int(episodes)
    except:
        print('Error: episodes is not an int')
        print('Use -h for option list')
        sys.exit(2)

    test(path=path, episodes=episodes, render=render, verbose=verbose)


if __name__ == "__main__" :
    main(sys.argv[1:])