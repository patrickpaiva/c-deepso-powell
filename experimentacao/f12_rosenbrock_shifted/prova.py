import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from functions import shifted_rosenbrock
import numpy as np
import pandas as pd
import ast

current_dir = os.path.dirname(__file__)

file_path = os.path.join(current_dir, 'experimento_shifted_rosenbrock_100_dimensoes_pcdeepso.xlsx')

df = pd.read_excel(file_path)

global_best = df['global_best'].iloc[0]

global_best = ast.literal_eval(global_best)

valor = shifted_rosenbrock(global_best)

print(f"Prova do Fitness: {valor}")