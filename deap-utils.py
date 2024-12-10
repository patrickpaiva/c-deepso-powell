# Apoio para geração de pesos otimizados
import numpy as np
from powell_cdeepso import c_deepso_powell_global_best_com_kmeans, c_deepso, c_deepso_powell_global_best_paralelo
from deap import base, creator, tools, algorithms
from functools import partial
from functions import shifted_rosenbrock, schwefel_1_2, griewank, shifted_ackley, elliptic_function
from cec2013lsgo.cec2013 import Benchmark

def deap_func(particle):
  return elliptic_function(particle)

bench = Benchmark()

fun_fitness = bench.get_function(2)
def rastrigin_shifted(sol):
    return fun_fitness(sol)

def function_ambigua(sol):
    fun_fitness = rastrigin_shifted
    if sol.ndim == 2:
        return np.apply_along_axis(fun_fitness, 1, sol)
    elif sol.ndim == 1:
        return fun_fitness(sol)

# Função objetivo para o DEEPSO com pesos otimizados
def evaluate_weights(weights, function, dimension, swarmSize, lowerBound, upperBound, max_fun_evals=100_000):
    W_i, W_a, W_c, T_com, T_mut, type_idx = weights
    
    types = ['sgpb', 'sg', 'pb']
    type_val = types[int(round(type_idx * 2))]

    best_fitness, _, _, _, _, _, _, _, _, _ = c_deepso_powell_global_best_paralelo(function, dimension, swarmSize, lowerBound, upperBound, max_fun_evals=max_fun_evals, W_i=W_i, W_a=W_a, W_c=W_c, T_com=T_com, T_mut=T_mut, type=type_val)
    bench.next_run()
    return best_fitness,

def custom_crossover(ind1, ind2, alpha=0.5):
    for i in range(len(ind1)):
        ind1[i] = np.clip(ind1[i] * (1 - alpha) + ind2[i] * alpha, 0, 1)
        ind2[i] = np.clip(ind2[i] * (1 - alpha) + ind1[i] * alpha, 0, 1)
    return ind1, ind2

def custom_mutation(individual, indpb):
    for i in range(len(individual)):
        if np.random.rand() < indpb:
            if i == 4:  # Índice de T_mut
                individual[i] = np.clip(np.random.normal(individual[i], 0.2), 0.1, 1.0)
            else:
                individual[i] = np.clip(np.random.normal(individual[i], 0.2), 0, 1)
    return individual,

def main():
    # Configurar DEAP
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", np.random.uniform, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, 6)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", partial(evaluate_weights, function=function_ambigua, dimension=1000,
                                        swarmSize=500, lowerBound=-5, upperBound=5))
    #toolbox.register("mate", tools.cxBlend, alpha=0.5)
    #toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)

    toolbox.register("mate", custom_crossover, alpha=0.5)

    toolbox.register("mutate", custom_mutation, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=50)
    ngen = 5
    cxpb = 0.5
    mutpb = 0.2

    # Executar o algoritmo genético
    print("Iniciando...")
    result = algorithms.eaSimple(population, toolbox, cxpb, mutpb, ngen, stats=None, halloffame=None, verbose=True)

    # Melhor solução encontrada
    best_individual = tools.selBest(population, 1)[0]
    best_type = ['sgpb', 'sg', 'pb'][int(round(best_individual[5] * 2))]
    print('Melhores pesos encontrados:', best_individual)
    print('Melhor valor de type encontrado:', best_type)

if __name__ == "__main__":
    main()