import pandas as pd
from glob import glob
import os



# Helper function to format numbers in scientific notation with E
def format_scientific(value):
    return "{:.2E}".format(value)

# Function to extract statistics from a DataFrame
def get_statistics_from_df(df):
    stats = {
        "Best": df.loc[0, 'Minimo'],
        "Median": df.loc[0, 'Mediana'],
        "Worst": df.loc[0, 'Maximo'],
        "Mean": df.loc[0, 'Media'],
        "Std": df.loc[0, 'Desvio_Padrao']
    }
    return stats

# Function to generate the LaTeX table with final formatting
def generate_latex_table_final_fixed(files):
    with open("results_table_final_fixed.tex", "w") as f:
        # Writing the beginning of the LaTeX table
        f.write("\\begin{table*}[!htb]\n")
        f.write("\\centering\n")
        f.write("\\caption{Experiment Results}\n")
        f.write("\\begin{tabular}{|c|c|c|c|c|c|c|c|c|c|c|}\n")  # Fixing the vertical lines
        f.write("\\hline\n")
        
        # Writing header for the functions in blocks of 5
        for i in range(0, len(files), 5):
            functions = files[i:i+5]
            # Covering two columns for the 'Functions' label and joining the blocks
            f.write("\\multicolumn{2}{|c|}{Functions} & " + " & ".join([f"$f_{{{i+1}}}$" for i in range(i, i+len(functions))]) + " \\\\ \n")
            f.write("\\hline\n")
            
            # For each evaluation threshold
            for eval_threshold, sheet_name in zip(
                ["$1.2 \\times 10^5$", "$6.0 \\times 10^5$", "$3.0 \\times 10^6$"], 
                ["Estatisticas_120k", "Estatisticas_600k", "Estatisticas"]
            ):
                # Write the evaluation threshold in the first column with vertical alignment
                f.write(f"\\multirow{{5}}{{*}}[0pt]{{\\centering {eval_threshold}}} & Best")
                for file in functions:
                    file_path = file
                    stats = get_statistics_from_df(pd.read_excel(file_path, sheet_name=sheet_name))
                    f.write(f" & {format_scientific(stats['Best'])}")
                f.write(" \\\\ \n")
                
                f.write(f" & Median")
                for file in functions:
                    stats = get_statistics_from_df(pd.read_excel(file, sheet_name=sheet_name))
                    f.write(f" & {format_scientific(stats['Median'])}")
                f.write(" \\\\ \n")
                
                f.write(f" & Worst")
                for file in functions:
                    stats = get_statistics_from_df(pd.read_excel(file, sheet_name=sheet_name))
                    f.write(f" & {format_scientific(stats['Worst'])}")
                f.write(" \\\\ \n")
                
                f.write(f" & Mean")
                for file in functions:
                    stats = get_statistics_from_df(pd.read_excel(file, sheet_name=sheet_name))
                    f.write(f" & {format_scientific(stats['Mean'])}")
                f.write(" \\\\ \n")
                
                f.write(f" & Std")
                for file in functions:
                    stats = get_statistics_from_df(pd.read_excel(file, sheet_name=sheet_name))
                    f.write(f" & {format_scientific(stats['Std'])}")
                f.write(" \\\\ \n")
                f.write("\\hline\n")  # Single line separating blocks
        
        # Finishing the table without extra lines
        f.write("\\end{tabular}\n")
        f.write("\\end{table*}\n")

# Example usage
files = [
    'experimento_f1_shifted_elliptic_1000_dimensoes_pcdeepso.xlsx',  # f1
    'experimento_f2_rastrigin_shifted_1000_dimensoes_pcdeepso.xlsx',  # f2
    'experimento_f3_ackley_shifted_1000_dimensoes_pcdeepso.xlsx',     # f3
    'experimento_f4_shifted_rotated_elliptic_1000_dimensoes_pcdeepso.xlsx',  # f4
    'experimento_f5_shifted_rotated_rastrigin_1000_dimensoes_pcdeepso.xlsx', # f5
    'experimento_f6_shifted_rotated_ackley_1000_dimensoes_pcdeepso.xlsx',    # f6
    'experimento_f7_schwefel_shifted_1000_dimensoes_pcdeepso.xlsx',          # f7
    'experimento_f8_shifted_rotated_elliptic_1000_dimensoes_pcdeepso.xlsx',  # f8
    'experimento_f9_shifted_rotated_rastrigin_1000_dimensoes_pcdeepso.xlsx', # f9
    'experimento_f10_shifted_rotated_ackley_1000_dimensoes_pcdeepso.xlsx',   # f10
    'experimento_f11_schwefel_shifted_1000_dimensoes_pcdeepso.xlsx',         # f11
    'experimento_f12_rosenbrock_shifted_1000_dimensoes_pcdeepso.xlsx',       # f12
    'experimento_f13_schwefel_overlapping_905_dimensoes_pcdeepso.xlsx',      # f13
    'experimento_f14_schwefel_overlapping_905_dimensoes_pcdeepso.xlsx',      # f14
    'experimento_f15_schwefel_shifted_1000_dimensoes_pcdeepso.xlsx'          # f15
]
folder_path = './'
excel_files = glob(os.path.join(folder_path, "*.xlsx"))
# Generating the final LaTeX table
generate_latex_table_final_fixed(files)
