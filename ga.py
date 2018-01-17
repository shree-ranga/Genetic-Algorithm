
'''
A simple Genetic Algorithm to minimize and objective function.
'''

from __future__ import division
from itertools import combinations
import numpy as np
# np.random.seed(0)

# objective function / fitness function
# a + 2*b + 3*c + 4*d - 30
# variables a,b,c,d can take values between 0 and 30, both inclusive
def fit_func(ch):
	f_x = np.abs((ch[0] + 2*ch[1] + 3*ch[2] + 4*ch[3]) - 30)
	return f_x


# selection
def selection(init_chromosomes):
	# print init_chromosomes

	f_x = []
	fit_scores = []
	cumulative_prob = 0
	cumulative_probs = []
	idx = []
	new_chromosomes = np.zeros((init_chromosomes.shape[1]))

	# calculate fitness
	for i in xrange(init_chromosomes.shape[0]):
		calc_fit_func = fit_func(init_chromosomes[i,:])
		f_x.append(calc_fit_func)

	# fitness probabilities
	for i in f_x:
		score = 1 / (1 + i)
		fit_scores.append(score)
	total_score = np.sum(fit_scores)
	prob_chrom_fit = fit_scores / total_score

	# cumulative probabilities
	for i in prob_chrom_fit:
		cumulative_prob += i
		cumulative_probs.append(cumulative_prob)

	# Roulette wheel selection
	rnd_number = np.random.uniform(size=len(cumulative_probs))
	for j in rnd_number:
		for i in xrange(len(cumulative_probs)):
			k = (i+1) % len(cumulative_probs)
			if (j > cumulative_probs[i]) and (j < cumulative_probs[k]):
				idx.append(k)

	rem_chromosomes_len = np.abs(len(cumulative_probs) - len(idx))
	if (rem_chromosomes_len != 0):
		rem_chroms_idx = np.random.randint(low=1, high=len(cumulative_probs), size=(rem_chromosomes_len))
		# print rem_chroms_idx
		idx = idx + rem_chroms_idx.tolist()

	for i in idx:
		new_chromosomes = np.vstack([new_chromosomes, init_chromosomes[i,:]])

	return new_chromosomes[1:,], init_chromosomes, f_x


# one-cut point crossover
def crossover(new_chromosomes):

	crossover_rate = 0.25
	parent_idx = []

	# choose parents for crossover
	for i in xrange(new_chromosomes.shape[0]):
		rnd_number = np.random.uniform()
		if rnd_number < crossover_rate:
			parent_idx.append(i)

	# crossover
	crossover_idxs = list(combinations(parent_idx, 2))
	for i in xrange(len(crossover_idxs)):
		rnd_number = np.random.randint(1, new_chromosomes.shape[1] - 1)
		c1 = crossover_idxs[i][0]
		c2 = crossover_idxs[i][1]
		new_chromosomes[c1][rnd_number+1:] = new_chromosomes[c2][rnd_number+1:]

	return new_chromosomes


# mutation
def mutation(crossover_chroms):

	mutation_rate = 0.05
	total_gen = crossover_chroms.shape[0] * crossover_chroms.shape[1]
	n_mutations = int(np.around(mutation_rate * total_gen))
	rnd_number1 = np.random.randint(0, total_gen, size=(n_mutations))
	rnd_number2 = np.random.randint(0, 30, size=(len(rnd_number1)))
	flat_chroms = crossover_chroms.flatten()
	for i in xrange(len(rnd_number1)):
		flat_chroms[rnd_number1[i]] = rnd_number2[i]
	mutated_chroms = flat_chroms.reshape(crossover_chroms.shape[0], crossover_chroms.shape[1])
	return mutated_chroms


if __name__ == "__main__":

	generation = 0
	init_chromosomes = np.random.randint(0,30,size=(8,4))
	# init_chromosomes = np.array([[12,5,23,8],
	# 	[2,21,18,3],
	# 	[10,4,13,14],
	# 	[20,1,10,6],
	# 	[1,4,13,19],
	# 	[20,5,17,1]])

	while(True):

		generation += 1
		print "genertion", generation
		new_chromosomes, fit_chroms, f_x = selection(init_chromosomes)
		crossover_chroms = crossover(new_chromosomes)
		mutated_chroms = mutation(crossover_chroms)
		init_chromosomes = mutated_chroms

		min_fit_score = np.min(f_x)
		arg_fit_chrom = np.argmin(f_x)
		print "Fit chromosome:", str(fit_chroms[arg_fit_chrom])
		print "Fit chromosome score:", str(min_fit_score) + "\n"
		if (min_fit_score == 0):
			break
