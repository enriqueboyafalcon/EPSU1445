#   Proj:   TFG - Evaluating RL algorithms and policies in gFootbal enviroment
#   File:   sort.py
#   Desc:   Generates a copy the of output of evaluate.py where the values are ordered, for better readability
#   Auth:   Enrique Boya Falcón
#   Date:   2021

import os
import sys

import csv

from scipy import stats

import numpy as np

def importResults(file: str, type: str):

    if type == 'algorithms':

        PPO = []
        DQN = []
        A2C = []

        with open(file, mode='r') as csvfile:

            reader = csv.DictReader(csvfile)
            for row in reader:
                PPO.append(float(row["PPO"]))
                DQN.append(float(row["DQN"]))
                A2C.append(float(row["A2C"]))

            return PPO, DQN, A2C

    elif type == 'policies':

        scoring = []
        checkpoint = []

        with open(file, mode='r') as csvfile:

            reader = csv.DictReader(csvfile)
            for row in reader:
                scoring.append(float(row["scoring"]))
                checkpoint.append(float(row["scr+chp"]))

        return scoring, checkpoint
        
    else: 
        assert False, "unhandled option"
                


# Creates a sorted version of the evaluation result, for better readability and easier interpretation
def main(argv):

    # PPO, DQN, A2C = importResults('../Data/Results/Counterattack.csv', 'algorithms')
    # PPO.sort(reverse=True)
    # DQN.sort(reverse=True)
    # A2C.sort(reverse=True)
    algorithm = ['A2C','DQN','PPO']

    scoring, checkpoint = importResults('../Data/Results/Iniesta.csv', 'policies')
    scoring.sort(reverse=True)
    checkpoint.sort(reverse=True)
    politic = ['scoring','scr+chp']
    
    output = '../Data/Results/Iniesta_sorted.csv'
    with open(output, 'w', newline='') as csvfile:

        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        writer.writerow(politic)

        # For every agent_id (row)
        for i in range(len(scoring)):

            accuracy = [0] * len(politic)

            # accuracy[0] = A2C[i]
            # accuracy[1] = DQN[i]
            # accuracy[2] = PPO[i]

            accuracy[0] = scoring[i]
            accuracy[1] = checkpoint[i]

            writer.writerow(accuracy)



if __name__ == "__main__" :
    main(sys.argv[1:])