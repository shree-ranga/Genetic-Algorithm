
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

def selection(init_chromosomes):
	# print init_chromosomes
	f_x = []
	fit_scores = []
	cumulative_prob = 0
	cumulative_probs = []
	idx = []
	new_chromosomes = np.zeros((init_chromosomes.shape[1]))
	for i in xrange(init_chromosomes.shape[0]):
		calc_fit_func = fit_func(init_chromosomes[i,:])
		f_x.append(calc_fit_func)

	for i in f_x:
		score = 1 / (1 + i)
		fit_scores.append(score)
	total_score = np.sum(fit_scores)
	prob_chrom_fit = fit_scores / total_score

	for i in prob_chrom_fit:
		cumulative_prob += i
		cumulative_probs.append(cumulative_prob)
	# print cumulative_probs

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

	return new_chromosomes[1:,]





def crossover(new_chromosomes):
	pass


def mutation():
	pass




if __name__ == "__main__":

	generations = 50

	init_chromosomes = np.random.randint(0,30,size=(8,4))
	# init_chromosomes = np.array([[12,5,23,8],
	# 	[2,21,18,3],
	# 	[10,4,13,14],
	# 	[20,1,10,6],
	# 	[1,4,13,19],
	# 	[20,5,17,1]])



	new_chromosomes = selection(init_chromosomes)
	print new_chromosomes
