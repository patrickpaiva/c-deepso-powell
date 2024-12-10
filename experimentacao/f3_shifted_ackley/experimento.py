import sys
import os
from datetime import datetime
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from functions import shifted_ackley
import numpy as np
from tqdm import tqdm
import pandas as pd
from utils import calculate_statistics
from powell_cdeepso import c_deepso_powell_global_best_com_kmeans_v4, c_deepso

dimensions = [30, 50, 100]

def experimentacao_powell(function, dimension, swarm_size, lower_bound, upper_bound, dispersion_tol, wi, wa, wc, tcom, tmut, max_v, max_fun_evals, max_iter):
    results = []

    for _ in tqdm(range(30), desc="Executando...", unit="iter"):
        best_fitness, g_best, _, _, _, function_evals = c_deepso_powell_global_best_com_kmeans_v4(function, dimension, swarm_size, lower_bound, upper_bound, dispersion_tol, max_iter=max_iter, max_fun_evals=max_fun_evals, type='pb', W_i=wi, W_a=wa, W_c=wc, T_mut=tmut, T_com=tcom, max_v=max_v)
        results.append({
            'best_fitness': best_fitness,
            'global_best': g_best,
            'function_evals': function_evals
        })
        best_fitnesses = [res['best_fitness'] for res in results]
        fun_evals = [res['function_evals'] for res in results]

    for result in results:
        result['global_best'] = ', '.join(map(str,result['global_best']))

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(by='best_fitness', ascending=True)

    function_evals_mean = np.mean(fun_evals)
    minimum, maximum, mean, std_dev = calculate_statistics(best_fitnesses)

    statistics = [{
        'Minimo': minimum,
        'Maximo': maximum,
        'Media': mean,
        'Desvio_Padrao': std_dev,
        'Aval_Func_Media': function_evals_mean
    }]
    df_stats = pd.DataFrame(statistics)
    data_hora_atual = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    nome_arquivo = f"experimento_{function.__name__}_{dimension}_dimensoes_pcdeepso_{data_hora_atual}.xlsx"
    with pd.ExcelWriter(nome_arquivo, engine='openpyxl') as writer:
        df_results.to_excel(writer, sheet_name='Dados', index=False)
        df_stats.to_excel(writer, sheet_name='Estatisticas', index=False)

for i in range(len(dimensions)):
    n = dimensions[i-1]    
    experimentacao_powell(
            function=shifted_ackley, 
            dimension=n, 
            swarm_size=n, 
            lower_bound=-32, 
            upper_bound=32,
            dispersion_tol = 1e-4,
            wi = 0.5134421262618705, 
            wa = 0.6697432209120159, 
            wc = 0.2052748212508989, 
            tcom= 0.6959287765030406, 
            tmut= 0.3683112445711794, 
            max_v=1.01,
            max_fun_evals=100_000,
            max_iter=None)


def experimentacao_cdeepso(function, dimension, swarm_size, lower_bound, upper_bound, wi, wa, wc, tcom, tmut, max_v, max_fun_evals, max_iter):
    results = []

    for _ in tqdm(range(30), desc="Executando...", unit="iter"):
        best_fitness, g_best, _, _, _, function_evals = c_deepso(function, dimension, swarm_size, lower_bound, upper_bound, max_iter=max_iter, max_fun_evals=max_fun_evals, type='pb', W_i=wi, W_a=wa, W_c=wc, T_mut=tmut, T_com=tcom, max_v=max_v)
        results.append({
            'best_fitness': best_fitness,
            'global_best': g_best,
            'function_evals': function_evals
        })
        best_fitnesses = [res['best_fitness'] for res in results]
        fun_evals = [res['function_evals'] for res in results]

    for result in results:
        result['global_best'] = ', '.join(map(str,result['global_best']))

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(by='best_fitness', ascending=True)

    function_evals_mean = np.mean(fun_evals)
    minimum, maximum, mean, std_dev = calculate_statistics(best_fitnesses)

    statistics = [{
        'Minimo': minimum,
        'Maximo': maximum,
        'Media': mean,
        'Desvio_Padrao': std_dev,
        'Aval_Func_Media': function_evals_mean
    }]
    df_stats = pd.DataFrame(statistics)

    data_hora_atual = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    nome_arquivo = f"experimento_{function.__name__}_{dimension}_dimensoes_cdeepso_{data_hora_atual}.xlsx"
    with pd.ExcelWriter(nome_arquivo, engine='openpyxl') as writer:
        df_results.to_excel(writer, sheet_name='Dados', index=False)
        df_stats.to_excel(writer, sheet_name='Estatisticas', index=False)
    

for i in range(len(dimensions)):
    n = dimensions[i-1]
    experimentacao_cdeepso(
        function=shifted_ackley, 
        dimension=n, 
        swarm_size=n, 
        lower_bound=-32, 
        upper_bound=32, 
        wi = 0.5134421262618705, 
        wa = 0.6697432209120159, 
        wc = 0.2052748212508989, 
        tcom= 0.6959287765030406, 
        tmut= 0.3683112445711794, 
        max_v=1.01,
        max_fun_evals=100_000,
        max_iter=None)
