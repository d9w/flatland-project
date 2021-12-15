from evaluate import get_env, get_state_action_size, evaluate, create_rail_env
from policy import NeuroevoPolicy
from argparse import ArgumentParser, Namespace
import numpy as np
import logging
import time
from cmaes import SepCMA

parser = ArgumentParser()
parser.add_argument('-g', '--gens', help='number of generations',
                    default=100, type=int)
parser.add_argument('-p', '--pop', help='population size (lambda for the 1+lambda ES)',
                    default=10, type=int)
parser.add_argument('-s', '--seed', help='seed for evolution',
                    default=0, type=int)
parser.add_argument('-m', '--sigma', help='std deviation for evolution',
                    default=0.01, type=float)
#parser.add_argument('-h', '--h1', help='first hidden layer size',
#                    default=32, type=int)
#parser.add_argument('-j', '--h2', help='second hidden layer size',
#                    default=32, type=int)
parser.add_argument('--log', help='log file',
                    default='evolution.log', type=str)
parser.add_argument('--weights', help='filename to save policy weights',
                    default='weights', type=str)
args = parser.parse_args()
logging.basicConfig(filename=args.log, level=logging.DEBUG, format='%(asctime)s %(message)s')

# starting point
small_env, small_params = get_env("small")
medium_env, medium_params = get_env("medium")
large_env, large_params = get_env("large")
all_params = [small_params, medium_params, large_params]
p = -1
s, a = get_state_action_size(large_env)
h1 = 32
h2 = 32
policy = NeuroevoPolicy(s, a, h1, h2)

# evolution
rng = np.random.default_rng(args.seed)
start = policy.get_params()
starttime = time.time()

# CMAES
gens = args.gens
sigma = args.sigma
x = start
x_best = start
f_best = np.Inf
n_evals = 0
lower_bounds = -np.ones(len(x)) * 20
upper_bounds = np.ones(len(x)) * 20
optimizer = SepCMA(mean=np.zeros(len(x)), sigma=sigma)
for generation in range(gens):
    # generate new env per 10 gens
    if generation % 10 == 0:
        p = (p + 1) % len(all_params)
        params = all_params[p]
        env = create_rail_env(Namespace(**params))
    gen_best = start
    gen_fit = np.Inf

    # evaluate
    solutions = []
    for i in range(optimizer.population_size):
        x = optimizer.ask()
        policy = NeuroevoPolicy(s, a, h1, h2)
        policy.set_params(x)
        fit = -evaluate(env, params, policy)
        solutions.append((x, fit))
        if fit < gen_fit:
            gen_fit = fit
            gen_best = x

    # evaluate best on large_env
    policy = NeuroevoPolicy(s, a, h1, h2)
    policy.set_params(gen_best)
    fit = -evaluate(large_env, large_params, policy)
    if fit < f_best:
        f_best = fit
        x_best = x

    # advance CMA-ES
    optimizer.tell(solutions)
    n_evals += optimizer.population_size

    # restart with new popsize multiplied by 2 (or 3)
    if optimizer.should_stop():
        popsize = optimizer.population_size * 2
        mean = lower_bounds + (np.random.rand(2) * (upper_bounds - lower_bounds))
        optimizer = SepCMA(mean=mean, sigma=sigma, population_size=popsize)
        print(f"Restart CMA-ES with popsize={popsize}")

    logging.info('\t%d\t%d\t%d\t%d\t%d', generation, n_evals, p, gen_fit, f_best)

# Evaluation
policy.set_params(x_best)
policy.save(args.weights)
best_eval = evaluate(large_env, large_params, policy)
print('Best individual: ', x_best[:5])
print('Fitness: ', best_eval)
