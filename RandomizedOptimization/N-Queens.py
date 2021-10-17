import six
import sys
sys.modules['sklearn.externals.six'] = six
import mlrose
import numpy as np
import matplotlib.pyplot as plt
import time

times = np.zeros(4)

fitness = mlrose.Queens()
problem = mlrose.DiscreteOpt(length = 8, fitness_fn = fitness, maximize = False, max_val = 8)

t0 = time.time()
rhc_best_state, rhc_best_fitness, rhc_fitness_curve = \
                mlrose.random_hill_climb(problem, max_attempts = 100, max_iters = 1000,\
                                         curve = True, random_state = 1)
t1 = time.time()
times[0] = t1-t0

print(rhc_best_state)
print(rhc_best_fitness)
print(rhc_fitness_curve)
plt.plot(rhc_fitness_curve, label='Random Hill Climb')
#plt.show()

t0 = time.time()
sa_best_state, sa_best_fitness, sa_fitness_curve = \
               mlrose.simulated_annealing(problem, schedule = mlrose.ExpDecay(),\
                                          max_attempts = 100, max_iters = 1000,\
                                          curve = True, random_state = 1)
t1 = time.time()
times[1] = t1-t0

print(sa_best_state)
print(sa_best_fitness)
#print(sa_fitness_curve)
plt.plot(sa_fitness_curve, label='Simulated Annealing')
#plt.show()

t0 = time.time()
ga_best_state, ga_best_fitness, ga_fitness_curve = \
               mlrose.genetic_alg(problem, max_attempts = 100, max_iters = 1000,\
                                  curve = True, random_state = 1, mutation_prob = 0.5)
t1 = time.time()
times[2] = t1-t0

print(ga_best_state)
print(ga_best_fitness)
#print(ga_fitness_curve)
plt.plot(ga_fitness_curve, label='Genetic Algorithm')
#plt.show()

t0 = time.time()
mimic_best_state, mimic_best_fitness, mimic_fitness_curve = \
                  mlrose.mimic(problem, max_attempts = 100, max_iters = 1000, curve = True,\
                               random_state = 1, pop_size = 2000)
t1 = time.time()
times[3] = t1-t0

print(mimic_best_state)
print(mimic_best_fitness)
#print(mimic_fitness_curve)
plt.plot(mimic_fitness_curve, label='MIMIC')
plt.title('N-Queens: Performance')
plt.ylabel('Fitness')
plt.xlabel('Iterations')
plt.legend(loc='lower right')
plt.show()

algorithms = ('Random Hill Climb', 'Simulated Annealing', 'Genetic Algorithm', 'MIMIC')
y_pos = np.arange(len(algorithms))

plt.figure()
plt.barh(y_pos, times)
plt.gca().set_yticks(y_pos)
plt.gca().set_yticklabels(algorithms)
plt.gca().invert_yaxis()
plt.title('Comparison of Solving Time')
plt.xlabel('Solving Time (in seconds)')
plt.subplots_adjust(left=0.2)
plt.show()
