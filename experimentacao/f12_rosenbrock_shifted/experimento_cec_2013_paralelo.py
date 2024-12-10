import sys
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from scipy.optimize import rosen
import numpy as np
from tqdm import tqdm
import pandas as pd
from utils import calculate_statistics
from powell_cdeepso import c_deepso_powell_global_best_paralelo
from cec2013lsgo.cec2013 import Benchmark

bench = Benchmark()

def rosenbrock_shifted(sol):
    fun_fitness = bench.get_function(12)
    return fun_fitness(sol)

dimension = bench.get_info(12)['dimension']

def function_ambigua(sol):
    fun_fitness = rosenbrock_shifted
    if sol.ndim == 2:
        return np.apply_along_axis(fun_fitness, 1, sol)
    elif sol.ndim == 1:
        return fun_fitness(sol)

# Função individual para executar c_deepso_powell_global_best
def executar_experimento(function, dimension, swarm_size, lower_bound, upper_bound, percent_powell_start_moment, percent_powell_func_evals, wi, wa, wc, tcom, tmut, max_v, max_fun_evals, max_iter):
    best_fitness, g_best, g_best_list, _, _, function_evals = c_deepso_powell_global_best_paralelo(
        function, dimension, swarm_size, lower_bound, upper_bound,
        percent_powell_start_moment=percent_powell_start_moment,
        percent_powell_func_evals=percent_powell_func_evals,
        max_iter=max_iter,
        max_fun_evals=max_fun_evals,
        type='pb', W_i=wi, W_a=wa, W_c=wc,
        T_mut=tmut, T_com=tcom, max_v=max_v
    )
    
    return {
        'best_fitness': best_fitness,
        'global_best': g_best,
        'function_evals': function_evals,
        'global_best_list': g_best_list
    }

def experimentacao_powell(function, dimension, swarm_size, lower_bound, upper_bound, percent_powell_start_moment, percent_powell_func_evals, wi, wa, wc, tcom, tmut, max_v, max_fun_evals, max_iter):
    results = []
    global_best_data = []

    # Usa ThreadPoolExecutor para executar em paralelo
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(executar_experimento, function, dimension, swarm_size, lower_bound, upper_bound,
                            percent_powell_start_moment, percent_powell_func_evals, wi, wa, wc, tcom, tmut, max_v, max_fun_evals, max_iter)
            for _ in range(25)
        ]

        for future in tqdm(as_completed(futures), total=25, desc="Executando em paralelo...", unit="iter"):
            result = future.result()
            results.append({
                'best_fitness': result['best_fitness'],
                'global_best': result['global_best'],
                'function_evals': result['function_evals']
            })
            global_best_data.append(result['global_best_list'])

    # Agora, calcula as estatísticas
    best_fitnesses = [res['best_fitness'] for res in results]
    fun_evals = [res['function_evals'] for res in results]

    for result in results:
        result['global_best'] = ', '.join(map(str, result['global_best']))

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(by='best_fitness', ascending=True)

    function_evals_mean = np.mean(fun_evals)
    minimum, maximum, mean, std_dev, median = calculate_statistics(best_fitnesses)

    statistics = [{
        'Minimo': minimum,
        'Maximo': maximum,
        'Media': mean,
        'Mediana': median,
        'Desvio_Padrao': std_dev,
        'Aval_Func_Media': function_evals_mean
    }]
    df_stats = pd.DataFrame(statistics)

    global_best_array = np.array(global_best_data)
    global_best_mean = np.mean(global_best_array, axis=0)
    df_global_best_mean = pd.DataFrame(global_best_mean, columns=['Convergencia_Media'])

    nome_arquivo = f"experimento_{function.__name__}_{dimension}_dimensoes_pcdeepso.xlsx"

    with pd.ExcelWriter(nome_arquivo, engine='openpyxl') as writer:
        df_results.to_excel(writer, sheet_name='Dados', index=False)
        df_stats.to_excel(writer, sheet_name='Estatisticas', index=False)
        df_global_best_mean.to_excel(writer, sheet_name='Convergencia_Media', index=False)


experimentacao_powell(
    function=function_ambigua,
    dimension=1000,
    swarm_size=500,
    lower_bound=-100,
    upper_bound=100,
    percent_powell_start_moment=0.5,
    percent_powell_func_evals=0.05,
    wi=0.4019092098808389,
    wa=0.3791940368874607,
    wc=0.7539312405916303,
    tcom=0.5819630448962767,
    tmut=0.3,
    max_v=1.01,
    max_fun_evals=120_000,
    max_iter=None
)
