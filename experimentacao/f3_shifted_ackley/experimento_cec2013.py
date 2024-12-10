import sys
import os
from datetime import datetime
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from scipy.optimize import rosen
import numpy as np
from tqdm import tqdm
import pandas as pd
from utils import calculate_statistics
from powell_cdeepso import c_deepso, c_deepso_powell_global_best
from cec2013lsgo.cec2013 import Benchmark

bench = Benchmark()

def ackley_shifted(sol):
    fun_fitness = bench.get_function(3)
    return fun_fitness(sol)

dimension = bench.get_info(3)['dimension']

def experimentacao_powell(function, dimension, swarm_size, lower_bound, upper_bound, percent_powell_start_moment, percent_powell_func_evals, wi, wa, wc, tcom, tmut, max_v, max_fun_evals, max_iter):
    results = []
    global_best_data = []

    for _ in tqdm(range(25), desc="Executando...", unit="iter"):
        best_fitness, g_best, g_best_list, _, _, function_evals = c_deepso_powell_global_best(function, dimension, swarm_size, lower_bound, upper_bound, percent_powell_start_moment=percent_powell_start_moment, percent_powell_func_evals=percent_powell_func_evals, max_iter=max_iter, max_fun_evals=max_fun_evals, type='pb', W_i=wi, W_a=wa, W_c=wc, T_mut=tmut, T_com=tcom, max_v=max_v)
        
        results.append({
            'best_fitness': best_fitness,
            'global_best': g_best,
            'function_evals': function_evals
        })
        global_best_data.append(g_best_list)

    best_fitnesses = [res['best_fitness'] for res in results]
    fun_evals = [res['function_evals'] for res in results]

    for result in results:
        result['global_best'] = ', '.join(map(str,result['global_best']))

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
        function=ackley_shifted, 
        dimension=1000, 
        swarm_size=500, 
        lower_bound=-32, 
        upper_bound=32,
        percent_powell_start_moment=0.5,
        percent_powell_func_evals=0.1,
        wi = 0.5134421262618705, 
        wa = 0.6697432209120159, 
        wc = 0.2052748212508989, 
        tcom= 0.6959287765030406, 
        tmut= 0.3683112445711794, 
        max_v=1.01,
        max_fun_evals=120_000,
        max_iter=None)

# def experimentacao_cdeepso(function, dimension, swarm_size, lower_bound, upper_bound, wi, wa, wc, tcom, tmut, max_v, max_fun_evals, max_iter):
#     results = []
#     global_best_data = []

#     for _ in tqdm(range(25), desc="Executando...", unit="iter"):
#         best_fitness, g_best, g_best_list, _, _, function_evals = c_deepso(function, dimension, swarm_size, lower_bound, upper_bound, max_iter=max_iter, max_fun_evals=max_fun_evals, type='pb', W_i=wi, W_a=wa, W_c=wc, T_mut=tmut, T_com=tcom, max_v=max_v)
#         results.append({
#             'best_fitness': best_fitness,
#             'global_best': g_best,
#             'function_evals': function_evals
#         })
#         global_best_data.append(g_best_list)

#     best_fitnesses = [res['best_fitness'] for res in results]
#     fun_evals = [res['function_evals'] for res in results]

#     for result in results:
#         result['global_best'] = ', '.join(map(str,result['global_best']))

#     df_results = pd.DataFrame(results)
#     df_results = df_results.sort_values(by='best_fitness', ascending=True)

#     function_evals_mean = np.mean(fun_evals)
#     minimum, maximum, mean, std_dev = calculate_statistics(best_fitnesses)

#     statistics = [{
#         'Minimo': minimum,
#         'Maximo': maximum,
#         'Media': mean,
#         'Desvio_Padrao': std_dev,
#         'Aval_Func_Media': function_evals_mean
#     }]
#     df_stats = pd.DataFrame(statistics)

#     global_best_array = np.array(global_best_data)  
#     global_best_mean = np.mean(global_best_array, axis=0)
#     df_global_best_mean = pd.DataFrame(global_best_mean, columns=['Convergencia_Media'])
    
#     nome_arquivo = f"experimento_{function.__name__}_{dimension}_dimensoes_cdeepso.xlsx"
#     with pd.ExcelWriter(nome_arquivo, engine='openpyxl') as writer:
#         df_results.to_excel(writer, sheet_name='Dados', index=False)
#         df_stats.to_excel(writer, sheet_name='Estatisticas', index=False)
#         df_global_best_mean.to_excel(writer, sheet_name='Convergencia_Media', index=False)

# for i in range(len(dimensions)):
#     n = dimensions[i-1]    
#     experimentacao_cdeepso(
#             function=rosen, 
#             dimension=n, 
#             swarm_size=n, 
#             lower_bound=-2.048, 
#             upper_bound=2.048, 
#             wi = 0.4019092098808389, 
#             wa = 0.3791940368874607, 
#             wc = 0.7539312405916303, 
#             tcom= 0.5819630448962767, 
#             tmut= 0.3, 
#             max_v=1.01,
#             max_fun_evals=100_000,
#             max_iter=None)