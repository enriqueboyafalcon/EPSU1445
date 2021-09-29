#   Proj:   TFG - Evaluating RL algorithms and policies in gFootbal enviroment
#   File:   plot.py
#   Desc:   Plots the results of other data
#   Auth:   Enrique Boya Falcón
#   Date:   2021

import os
import sys

import csv

import numpy as np

import matplotlib.pyplot as plt

from scipy.ndimage.filters import gaussian_filter1d

n = 0

sample_rate = 500

scenarios = ['MyScenarios_Penalty','MyScenarios_Counterattack','MyScenarios_Iniesta']
algorithms = ['PPO','DQN','A2C']
politics = ['scoring','checkpoint']
types = ['plotAll','plotLimAverage','plotAccuracies']

def readlog(path: str, sample_rate: int, last: bool):
    # check policy to see what reward value counts as a goal (scoring rew=1, scrchp rew=2)
    parse = path.split('/')
    parse = parse[-1]
    parse = parse.split(' ')
    if 'scoring' in parse:
        goal = 1
    else:
        goal = 2

    # fill data[] with average acuracy. Average every 'sample_rate' episodes
    data = []
    data.append(0)
    with open(path, mode='r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader) # skip headers
        next(reader)
        n = 0
        acc = 0
        for row in reader:
            if float(row[0]) == goal:
                acc += 1
            if n == sample_rate:
                n = 0
                data.append(acc/sample_rate)
                acc = 0
            n += 1

        eps = []
        for i in range(0,len(data)):
            eps.append(sample_rate*i)

        if last:
            data.append(acc/n)
            eps.pop
            eps.append(eps[-1]+n)
        
        return eps, data

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

#   -----------------------------------------------------      

def plotAll(scenario: str, algorithm: str, politic: str, sample_rate: int, ax=plt):
    # global n
    # n += 1
    # ax.figure(n)

    if politic == 'checkpoint':
        politic = 'scr+chp'

    x_length = -np.inf

    for i in range(1,31):
        eps, data = readlog('../Data/Logs/{} {} {} {:02d}.csv'
                        .format(scenario, algorithm, politic, i),
                    sample_rate,
                    last=True)
        if len(data) > x_length:
            x_length = len(data)
        data = gaussian_filter1d(data, sigma=2)
        ax.plot(eps, data)

    # ax.suptitle("Precisión a lo largo del entrenamiento")
    # ax.title("{} - {} - {}"
    #             .format(scenario[12:], algorithm, politic),
    #             y=1.01,
    #             fontsize=10)

    # axes = ax.gca()
    # axes.yaxis.grid()
    # ax.xlabel("Episodios")
    # ax.ylabel("Precisión")
    # ax.yticks([0, 0.25, 0.5, 0.75, 1], ['0', '25', '50', '75', '100'])
    ax.axis([0, (x_length+1)*sample_rate, 0, 1])

    # ax.savefig('Graphs/plotAll {} {} {}.pdf'
    #                 .format(scenario[12:], algorithm, politic))


def plotLimAverage(scenario: str, algorithm: str, politic: str, sample_rate: int, ax=plt):
    # global n
    # n += 1
    # ax.figure(n)
    
    if politic == 'checkpoint':
        politic = 'scr+chp'

    x_length = -np.inf

    for i in range(1,31):
        _, data = readlog('../Data/Logs/{} {} {} {:02d}.csv'
                        .format(scenario, algorithm, politic, i),
                    sample_rate,
                    last=False)
        if len(data) > x_length:
            x_length = len(data)

    matrix = [[-np.inf for x in range(30)] for y in range(x_length)] # x duration - y agent

    for agent in range(0,30):
        _,data = readlog('../Data/Logs/{} {} {} {:02d}.csv'
                        .format(scenario, algorithm, politic, agent+1),
                    sample_rate,
                    last=False)
        for i in range(len(data)):
            if data[i] > matrix[i][agent]:
                if i > 0 and data[i] < matrix[i-1][agent]:
                    matrix[i][agent] = matrix[i-1][agent]
                else:
                    matrix[i][agent] = data[i]
            # print("agent={:2d} i={:3d} data[i]={:4.2f} matrix[i][agent]={:4.2f}".format(agent, i, data[i], matrix[i][agent])) # debugging
        for i in range(len(data), x_length):
            matrix[i][agent] = matrix[i-1][agent]

    min = [np.inf] * x_length
    max = [-np.inf] * x_length
    average = [0] * x_length

    for i in range(x_length):
        for agent in range(0,30):
            if matrix[i][agent] > max[i]:
                max[i] = matrix[i][agent]
            if matrix[i][agent] < min[i]:
                min[i] = matrix[i][agent]
            average[i] += matrix[i][agent]
        average[i] = average[i] / 30

    eps = []
    for i in range(0, x_length):
        eps.append(sample_rate*i)

    # ax.suptitle("Precisión mínima, máxima y media a lo largo del entrenamiento")
    # ax.title("{} - {} - {}"
    #             .format(scenario[12:], algorithm, politic),
    #             y=1.01,
    #             fontsize=10)
    
    # axes = ax.gca()
    # axes.yaxis.grid()
    # ax.xlabel("Episodios")
    # ax.ylabel("Precisión")
    # ax.yticks([0, 0.25, 0.5, 0.75, 1], ['0', '25', '50', '75', '100'])
    ax.axis([0, (x_length+1)*sample_rate, 0, 1])

    ax.plot(eps, min, color='tab:red')
    ax.plot(eps, max, color='tab:green')
    ax.fill_between(eps, min, max, facecolor='tab:blue', alpha=0.15)
    ax.plot(eps, average, color='tab:blue')

    # ax.savefig('Graphs/plotMinMaxAvg {} {} {}.pdf'
    #                 .format(scenario[12:], algorithm, politic))


def plotAcc(scenario: str):
    global n
    n += 1
    plt.figure(n)

    scenario = scenario[12:]
    if scenario == 'Penalty':
        PPO, DQN, A2C = importResults('../Data/Results/Penalty.csv', 'algorithms')
        raw = [A2C, DQN, PPO]
        plt.xlabel("Algoritmo")
        plt.xticks([0, 1, 2], ['A2C', 'DQN', 'PPO'])
    else:
        scoring, checkpoint = importResults('../Data/Results/{}.csv'.format(scenario), 'policies')
        raw = [scoring, checkpoint]
        plt.xlabel("Política")
        plt.xticks([0, 1], ['scoring', 'checkpoint'])

    x = []
    y = []
    for list in raw:
        for value in list:
            x.append(raw.index(list))
            y.append(value)

    plt.suptitle("Resultados de la evaluación")
    plt.title("{}"
                .format(scenario),
                y=1.01,
                fontsize=10)

    axes = plt.gca()
    axes.yaxis.grid()
    plt.ylabel("Precisión")
    plt.yticks([0, 0.25, 0.5, 0.75, 1], ['0', '25', '50', '75', '100'])
    plt.axis([-0.5, len(raw)-0.5 , 0, 1])
    plt.plot(x,y, "_", markersize=50)

    plt.savefig('Graphs/plotAcc {}.pdf'
                    .format(scenario))


            
def generateAll():

    # phase 1
    plotAll(scenarios[0], algorithms[0], politics[0], sample_rate)
    plotAll(scenarios[0], algorithms[1], politics[0], sample_rate)
    plotAll(scenarios[0], algorithms[2], politics[0], sample_rate)
    plotLimAverage(scenarios[0], algorithms[0], politics[0], sample_rate)
    plotLimAverage(scenarios[0], algorithms[1], politics[0], sample_rate)
    plotLimAverage(scenarios[0], algorithms[2], politics[0], sample_rate)
    plotAcc(scenarios[0])

    # phase 2
    plotAll(scenarios[1], algorithms[0], politics[0], sample_rate)
    plotAll(scenarios[1], algorithms[0], politics[1], sample_rate)
    plotLimAverage(scenarios[1], algorithms[0], politics[0], sample_rate)
    plotLimAverage(scenarios[1], algorithms[0], politics[1], sample_rate)
    plotAcc(scenarios[1])

    # phase 3
    plotAll(scenarios[2], algorithms[0], politics[0], sample_rate)
    plotAll(scenarios[2], algorithms[0], politics[1], sample_rate)
    plotLimAverage(scenarios[2], algorithms[0], politics[0], sample_rate)
    plotLimAverage(scenarios[2], algorithms[0], politics[1], sample_rate)
    plotAcc(scenarios[2])

def allChapuza():
    chapuza(scenarios[0], algorithms[1], politics[0], 0)
    chapuza(scenarios[0], algorithms[2], politics[0], 0)
    chapuza(scenarios[0], algorithms[0], politics[0], 0)

    chapuza(scenarios[1], algorithms[0], politics[0], 1)
    chapuza(scenarios[1], algorithms[0], politics[1], 1)

    chapuza(scenarios[2], algorithms[0], politics[0], 2)
    chapuza(scenarios[2], algorithms[0], politics[1], 2)


def chapuza(scenario, algorithm, politic, fase):

    fig, (ax1,ax2) = plt.subplots(ncols=2, sharey=True)
    fig.set_figheight(5)
    fig.set_figwidth(12)

    plotAll(scenario, algorithm, politic, sample_rate, ax1)
    plotLimAverage(scenario, algorithm, politic, sample_rate, ax2)

    fig.subplots_adjust(wspace=0) 

    ax1.grid(axis='y')
    ax2.grid(axis='y')

    plt.sca(ax1)
    plt.yticks([0, 0.25, 0.5, 0.75, 1], ['0', '25', '50', '75', '100'])

    if fase == 0:
        plt.suptitle("Precisión de los agentes a lo largo del entrenamiento en {} con {}"
                        .format(scenario[12:], algorithm))
    else:
        plt.suptitle("Precisión de los agentes a lo largo del entrenamiento en {} con {}"
                        .format(scenario[12:], politic))
    ax1.set_title("Todos los agentes")
    ax2.set_title("Mínimo, máximo y media de la mejor precisión")

    ax1.set_xticks(ax1.get_xticks()[:-2]) 
    
    plt.margins(y=-.49999999999)

    plt.ylabel("Precisión")
    plt.xlabel("Episodios", x=1)
    plt.savefig('../Data/Graphs/plotall+minmaxavg {} {} {}.pdf'
                    .format(scenario[12:], algorithm, politic),bbox_inches='tight') # bbox_inches='tight'

def main(argv):

    allChapuza()

    # generateAll()

    # scenario = 0
    # algorithm = 0
    # politic = 0
    # graph = 1

    # if   types[graph] == 'plotAll':
    #     plotAll(scenarios[scenario], algorithms[algorithm], politics[politic], sample_rate)
        
    # elif types[graph] == 'plotLimAverage':
    #     plotLimAverage(scenarios[scenario], algorithms[algorithm], politics[politic], sample_rate)

    # elif types[graph] == 'plotAccuracies':
    #     plotAcc(scenarios[scenario])

    # else: assert False, "Unhandled graph"


if __name__ == "__main__" :
    main(sys.argv[1:])