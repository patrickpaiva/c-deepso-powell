import numpy as np
from scipy.stats import ttest_ind

def generatePopulation(dimension, populationSize, lowerBound, upperBound):
    population = np.random.uniform(lowerBound, upperBound, size=(populationSize, dimension))
    return population

def generateMultiplicationMatrix(dimension, T_com):
  C = np.zeros((dimension, dimension))

  for i in range(dimension):
    C[i][i] = 1 if np.random.uniform() <= T_com else 0
  return C

def print_statistics(function, dimension, PCDEEPSO_stats, CDEEPSO_stats, PCDEEPSO_evals, CDEEPSO_evals):
    print("Estatísticas dos Resultados dos Algoritmos PC-DEEPSO e C-DEEPSO\n")
    print(f"{function.__name__} em {dimension} dimensões\n")
    print(f"{'Metric':<20} {'PC-DEEPSO':<15} {'C_DEEPSO':<15}")
    print("-" * 50)
    print(f"{'Minimum':<20} {PCDEEPSO_stats[0]:<15.5f} {CDEEPSO_stats[0]:<15.5f}")
    print(f"{'Maximum':<20} {PCDEEPSO_stats[1]:<15.5f} {CDEEPSO_stats[1]:<15.5f}")
    print(f"{'Mean':<20} {PCDEEPSO_stats[2]:<15.5f} {CDEEPSO_stats[2]:<15.5f}")
    print(f"{'Standard Deviation':<20} {PCDEEPSO_stats[3]:<15.5f} {CDEEPSO_stats[3]:<15.5f}")
    print(f"{'Function Evaluations':<20} {PCDEEPSO_evals:<15.0f} {CDEEPSO_evals:<15.0f}")

def calculate_statistics(results):
    minimum = np.min(results)
    maximum = np.max(results)
    mean = np.mean(results)
    std_dev = np.std(results)
    median = np.median(results)
    return minimum, maximum, mean, std_dev, median

def perform_t_test(results_PCDEEPSO, results_CDEEPSO, alpha=0.05):
    t_stat, p_value = ttest_ind(results_PCDEEPSO, results_CDEEPSO, equal_var=False)
    action = 'Rejeitar' if p_value < alpha else 'Aceitar'
    winner = 'PC-DEEPSO' if np.mean(results_PCDEEPSO) < np.mean(results_CDEEPSO) else 'C-DEEPSO'
    print(f"\nResultado do t-test\n")
    print(f"{'Metric':<20} {'Value':<15}")
    print("-" * 35)
    print(f"{'t-statistic':<20} {t_stat:<15.5f}")
    print(f"{'p-value':<20} {p_value:<15.10f}")
    print(f"{'Action':<20} {action:<15}")
    print(f"{'Winner':<20} {winner:<15}")
  