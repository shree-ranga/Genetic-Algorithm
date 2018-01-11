
'''
A simple Genetic Algorithm to minimize and objective function.
'''

from __future__ import division
import numpy as np
np.random.seed(0)

# objective function / fitness function
# a + 2*b + 3*c + 4*d - 30
# variables a,b,c,d can take values between 0 and 30, both inclusive
def fit_func(ch):
    f_x = np.abs((ch[0] + 2*ch[1] + 3*ch[2] + 4*ch[3]) - 30)
    return f_x

def selection(f_x):
    fit_scores = []
    cumulative_prob = 0
    cumulative_probs = []
    new_chromosomes = []

    for i in f_x:
        score = 1 / (1 + i)
        fit_scores.append(score)
    total_score = np.sum(fit_scores)
    prob_chrom_fit = fit_scores / total_score

    for i in prob_chrom_fit:
        cumulative_prob += i
        cumulative_probs.append(cumulative_prob)

    rnd_num = np.random.uniform(size=(len(f_x),1))
    for i in xrange(1, len(rnd_num)):
        # if ((rnd_num[i-1] > cumulative_probs[i-1]) and (rnd_num[i-1] < cumulative_prob[i])):
        #     new_chromosomes.append(f_x[i])

    return new_chromosomes






def crossover():
    pass


def mutation():
    pass




if __name__ == "__main__":

    generations = 50
    f_x = []

    init_chromosomes = np.random.randint(0,30,size=(8,4))

    for i in xrange(init_chromosomes.shape[0]):
        calc_fit_func = fit_func(init_chromosomes[i,:])
        f_x.append(calc_fit_func)

    new_chromosomes = selection(f_x)
