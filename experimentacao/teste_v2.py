import os
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind

# List of functions with dimensions
function_names = [
    "f1_shifted_elliptic_1000",
    "f2_rastrigin_shifted_1000",
    "f3_ackley_shifted_1000",
    "f4_shifted_rotated_elliptic_1000",
    "f5_shifted_rotated_rastrigin_1000",
    "f6_shifted_rotated_ackley_1000",
    "f7_schwefel_shifted_1000",
    "f8_shifted_rotated_elliptic_1000",
    "f9_shifted_rotated_rastrigin_1000",
    "f10_shifted_rotated_ackley_1000",
    "f11_schwefel_shifted_1000",
    "f12_rosenbrock_shifted_1000",
    "f13_schwefel_overlapping_905",
    "f14_schwefel_overlapping_905",
    "f15_schwefel_shifted_1000"
]

# Algorithms and their corresponding suffixes in filenames
alg_suffixes = {
    'C-DEEPSO': 'cdeepso',
    'C-DEEPSO-Powell': 'pcdeepso'
}

# Alpha value for the t-test
alpha = 0.05

# Directory containing the data files
data_dir = '.'

# Store results for the LaTeX table
results = []

for func_name in function_names:
    data = {}
    for alg_name, alg_suffix in alg_suffixes.items():
        # Construct the filename
        filename = f"experimento_{func_name}_dimensoes_{alg_suffix}.xlsx"
        filepath = os.path.join(data_dir, filename)
        try:
            # Read the Excel file
            df = pd.read_excel(filepath, sheet_name='Dados')
            # Extract the 'best_fitness' column
            best_fitness_values = df['best_fitness'].values
            if len(best_fitness_values) != 25:
                print(f"Warning: Expected 25 best_fitness values in {filepath}, got {len(best_fitness_values)}")
            data[alg_name] = best_fitness_values
        except FileNotFoundError:
            print(f"Error: File {filepath} not found.")
            data[alg_name] = []
            continue
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            data[alg_name] = []
            continue
    # Check if we have data for both algorithms
    if len(data.get('C-DEEPSO', [])) == 0 or len(data.get('C-DEEPSO-Powell', [])) == 0:
        print(f"Skipping function {func_name} due to missing data.")
        continue
    # Perform the t-test
    t_stat, p_value = ttest_ind(data['C-DEEPSO'], data['C-DEEPSO-Powell'], equal_var=False)
    
    # Determine if we reject or accept the null hypothesis
    if p_value < alpha:
        hypothesis = 'Rejeitar H0'
    else:
        hypothesis = 'Aceitar H0'
    
    # Determine the winning algorithm
    mean_cdeepso = np.mean(data['C-DEEPSO'])
    mean_cdeepso_powell = np.mean(data['C-DEEPSO-Powell'])
    
    if hypothesis == 'Rejeitar H0':
        if mean_cdeepso < mean_cdeepso_powell:
            winner = 'C-DEEPSO'
        elif mean_cdeepso_powell < mean_cdeepso:
            winner = 'C-DEEPSO-Powell'
        else:
            winner = 'ERRO'
    else:
        winner = 'Empate'
    
    # Append the results
    results.append({
        'Function': func_name,
        'p-value': p_value,
        'Hypothesis': hypothesis,
        'Winner': winner
    })

# Generate LaTeX table
latex_table = r'''\begin{table}[ht]
\centering
\caption{T-Test Results between C-DEEPSO and C-DEEPSO-Powell}
\begin{tabular}{lccc}
\hline
Function & p-value & Hypothesis & Winner \\
\hline
'''

for res in results:
    latex_table += f"{res['Function']} & {res['p-value']:.5f} & {res['Hypothesis']} & {res['Winner']} \\\\\n"

latex_table += r'''\hline
\end{tabular}
\end{table}
'''

# Write LaTeX table to a file
with open('t_test_results.tex', 'w') as f:
    f.write(latex_table)

print("LaTeX table generated and saved as t_test_results.tex")
