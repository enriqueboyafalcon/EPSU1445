#   Proj:   TFG - Evaluating RL algorithms and policies in gFootbal enviroment
#   File:   results.py
#   Desc:   Loads the results from evaluate.py and performs stadistics tests to compare the data
#   Auth:   Enrique Boya Falcón
#   Date:   2021

import os
import sys

import csv

from scipy import stats

import statistics

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

def formatNumber(n: float):
    if n < 0.01:
        return "{:.2e}".format(n)
    elif n < 1:
        return "{:.2g}".format(n)
    else:
        return "{:.2f}".format(n)

def printResult(stat: float, p: float):
    print("\tStatistics={}, p={}".format(formatNumber(stat), formatNumber(p)))


# https://machinelearningmastery.com/statistical-hypothesis-tests-in-python-cheat-sheet/
            
def penaltyTest():

    print("Penalty")

    PPO, DQN, A2C = importResults('../Data/Results/Penalty.csv', 'algorithms')

    print("\nA2C:\tMean={}\tVariance={}"
            .format(formatNumber(statistics.mean(A2C)),
                    formatNumber(statistics.pvariance(A2C))))

    print("DQN:\tMean={}\tVariance={}"
            .format(formatNumber(statistics.mean(DQN)),
                    formatNumber(statistics.pvariance(DQN))))

    print("PPO:\tMean={}\tVariance={}"
            .format(formatNumber(statistics.mean(PPO)),
                    formatNumber(statistics.pvariance(PPO))))

    #   -------------------------------------------------------------

    print("\nShapiro-Wilk Normality test")

    print("\tA2C")
    stat, p = stats.shapiro(A2C)
    printResult(stat,p)
    alpha = 0.05
    if p > alpha: print('\t-> Probably gaussian')
    else:         print('\t-> Probably not gaussian')

    print("\n\tDQN")
    stat, p = stats.shapiro(DQN)
    printResult(stat,p)
    alpha = 0.05
    if p > alpha: print('\t-> Probably gaussian')
    else:         print('\t-> Probably not gaussian')

    print("\n\tPPO")
    stat, p = stats.shapiro(PPO)
    printResult(stat,p)
    alpha = 0.05
    if p > alpha: print('\t-> Probably gaussian')
    else:         print('\t-> Probably not gaussian')

    #   -------------------------------------------------------------

    print("\nBartlett’s test for equal variances")
    stat, p = stats.bartlett(A2C, DQN, PPO)
    printResult(stat,p)
    alpha = 0.05
    if p > alpha: print('\t-> Probably equal variances')
    else:         print('\t-> Probably different variances')

    #   -------------------------------------------------------------

    print("\nKruskal-Wallis H-test")
    stat, p = stats.kruskal(A2C, DQN, PPO)
    printResult(stat,p)
    alpha = 0.05
    if p > alpha: print('\t-> Probably the same distribution')
    else:         print('\t-> Probably different distributions')

    #   -------------------------------------------------------------

    print("\nMann-Whitney U Test")
    
    print("\tA2C - DQN")
    stat, p = stats.mannwhitneyu(A2C, DQN)
    printResult(stat,p)
    alpha = 0.05
    if p > alpha: print('\t-> Probably the same distribution')
    else:         print('\t-> Probably different distributions')

    print("\tDQN - PPO")
    stat, p = stats.mannwhitneyu(DQN, PPO)
    printResult(stat,p)
    alpha = 0.05
    if p > alpha: print('\t-> Probably the same distribution')
    else:         print('\t-> Probably different distributions')

    print("\tA2C - PPO")
    stat, p = stats.mannwhitneyu(A2C, PPO)
    printResult(stat,p)
    alpha = 0.05
    if p > alpha: print('\t-> Probably the same distribution')
    else:         print('\t-> Probably different distributions')

    #   -------------------------------------------------------------

    print("\nMedian comparison")
    medians = [np.median(A2C), np.median(DQN), np.median(PPO)]
    print("\tA2C {:.2f}".format(medians[0] * 100))
    print("\tDQN {:.2f}".format(medians[1] * 100))
    print("\tPPO {:.2f}".format(medians[2] * 100))


def counterattackTest():

    print("Counterattack")

    scoring, checkpoint = importResults('../Data/Results/Counterattack.csv', 'policies')

    print("\nscoring:\tMean={}\tVariance={}"
            .format(formatNumber(statistics.mean(scoring)),
                    formatNumber(statistics.pvariance(scoring))))

    print("checkpoint:\tMean={}\tVariance={}"
            .format(formatNumber(statistics.mean(checkpoint)),
                    formatNumber(statistics.pvariance(checkpoint))))

    #   -------------------------------------------------------------

    print("\nShapiro-Wilk Normality test")

    print("\tScoring")
    stat, p = stats.shapiro(scoring)
    printResult(stat,p)
    alpha = 0.05
    if p > alpha: print('\t-> Probably gaussian')
    else:         print('\t-> Probably not gaussian')

    print("\n\tCheckpoint")
    stat, p = stats.shapiro(checkpoint)
    printResult(stat,p)
    alpha = 0.05
    if p > alpha: print('\t-> Probably gaussian')
    else:         print('\t-> Probably not gaussian')

    #   -------------------------------------------------------------

    print("\nBartlett’s test for equal variances")
    stat, p = stats.bartlett(scoring, checkpoint)
    printResult(stat,p)
    alpha = 0.05
    if p > alpha: print('\t-> Probably equal variances')
    else:         print('\t-> Probably different variances')

    #   -------------------------------------------------------------

    print("\nMann-Whitney U Test")
    
    print("\tScoring - Checkpoint")
    stat, p = stats.mannwhitneyu(scoring, checkpoint)
    printResult(stat,p)
    alpha = 0.05
    if p > alpha: print('\t-> Probably the same distribution')
    else:         print('\t-> Probably different distributions')

    #   -------------------------------------------------------------

    print("\nMedian comparison")
    medians = [np.median(scoring), np.median(checkpoint)]
    print("\tScoring {:.2f}".format(medians[0] * 100))
    print("\tCheckpoint {:.2f}".format(medians[1] * 100))



def iniestaTest():

    print("Iniesta")

    scoring, checkpoint = importResults('../Data/Results/Iniesta.csv', 'policies')

    print("\nscoring:\tMean={}\tVariance={}"
            .format(formatNumber(statistics.mean(scoring)),
                    formatNumber(statistics.pvariance(scoring))))

    print("checkpoint:\tMean={}\tVariance={}"
            .format(formatNumber(statistics.mean(checkpoint)),
                    formatNumber(statistics.pvariance(checkpoint))))

    #   -------------------------------------------------------------

    print("\nShapiro-Wilk Normality test")

    print("\tScoring")
    stat, p = stats.shapiro(scoring)
    printResult(stat,p)
    alpha = 0.05
    if p > alpha: print('\t-> Probably gaussian')
    else:         print('\t-> Probably not gaussian')

    print("\n\tCheckpoint")
    stat, p = stats.shapiro(checkpoint)
    printResult(stat,p)
    alpha = 0.05
    if p > alpha: print('\t-> Probably gaussian')
    else:         print('\t-> Probably not gaussian')

    #   -------------------------------------------------------------

    print("\nBartlett’s test for equal variances")
    stat, p = stats.bartlett(scoring, checkpoint)
    printResult(stat,p)
    alpha = 0.05
    if p > alpha: print('\t-> Probably equal variances')
    else:         print('\t-> Probably different variances')

    #   -------------------------------------------------------------

    print("\nMann-Whitney U Test")
    
    print("\tScoring - Checkpoint")
    stat, p = stats.mannwhitneyu(scoring, checkpoint)
    printResult(stat,p)
    alpha = 0.05
    if p > alpha: print('\t-> Probably the same distribution')
    else:         print('\t-> Probably different distributions')

    #   -------------------------------------------------------------

    # print("\nMedian comparison")
    # medians = [np.median(scoring), np.median(checkpoint)]
    # print("\tScoring {:.2f}".format(medians[0] * 100))
    # print("\tCheckpoint {:.2f}".format(medians[1] * 100))


# to output the results to a file:
# python results.py > ../Data/Results/Results.txt
def main(argv):

    penaltyTest()
    print("\n-------------------------------------------\n")
    counterattackTest()
    print("\n-------------------------------------------\n")
    iniestaTest()


if __name__ == "__main__" :
    main(sys.argv[1:])