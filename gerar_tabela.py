import pandas as pd
import numpy as np
import os
from glob import glob
import re


def toLatex(global_df, filename_latex):
    """
    Transform the DataFrame to Latex
    """
    def extract(milestone, name, df_grouped, funs):
        bests = df_grouped[funs].tolist()
        values = [milestone, name] + bests
        return values

    def fun2latex(name):
        return re.sub(r'F0?(\d*)$', '$f_{\\1}$', name)

    funs = [name for name in global_df.columns if name.startswith('F')]
    # milestones = global_df['milestone'].unique()
    milestones = [1.2e5, 6e5, 3e6]
    total = 5*len(milestones)
    df_latex = pd.DataFrame(columns=['Milestone', 'Category']+funs, index=np.arange(total))
    i = 0

    for milestone, df in global_df.groupby(['milestone']):
        if milestone not in milestones:
            continue

        df_latex.loc[i] = extract(milestone, 'Best', df.min(), funs)
        df_latex.loc[i+1] = extract(milestone, 'Median', df.median(), funs)
        df_latex.loc[i+2] = extract(milestone, 'Worst', df.max(), funs)
        df_latex.loc[i+3] = extract(milestone, 'Mean', df.mean(), funs)
        df_latex.loc[i+4] = extract(milestone, 'StDev', df.std(), funs)
        df_latex[df_latex.isnull()] = 0
        i += 5

    df_latex.columns = [fun2latex(col) for col in df_latex.columns]
    df_latex['Milestone'] = df_latex['Milestone'].map('{:.2e}'.format)
    df_latex = df_latex.set_index(['Milestone','Category'])
    # print(df_latex.groupby(['milestone', 'category']))
    df_latex.to_latex(filename_latex,  multirow=True, multicolumn=False, escape=False)
    return

def extract_statistics_from_excel(file_path):
    # Carregar as planilhas do arquivo Excel e pegar as estatísticas
    df_stats = pd.read_excel(file_path, sheet_name='Estatisticas')
    df_stats_120k = pd.read_excel(file_path, sheet_name='Estatisticas_120k')
    df_stats_600k = pd.read_excel(file_path, sheet_name='Estatisticas_600k')

    # Retornar as estatísticas extraídas
    return {
        'Best': df_stats['Minimo'].values[0],
        'Median': df_stats['Mediana'].values[0],
        'Worst': df_stats['Maximo'].values[0],
        'Mean': df_stats['Media'].values[0],
        'Std': df_stats['Desvio_Padrao'].values[0],
        'Best_120k': df_stats_120k['Minimo'].values[0],
        'Median_120k': df_stats_120k['Mediana'].values[0],
        'Worst_120k': df_stats_120k['Maximo'].values[0],
        'Mean_120k': df_stats_120k['Media'].values[0],
        'Std_120k': df_stats_120k['Desvio_Padrao'].values[0],
        'Best_600k': df_stats_600k['Minimo'].values[0],
        'Median_600k': df_stats_600k['Mediana'].values[0],
        'Worst_600k': df_stats_600k['Maximo'].values[0],
        'Mean_600k': df_stats_600k['Media'].values[0],
        'Std_600k': df_stats_600k['Desvio_Padrao'].values[0],
    }

def main():
    folder_path = './'  # Colocar o caminho para a pasta com os arquivos .xlsx
    excel_files = glob(os.path.join(folder_path, "*.xlsx"))
    
    # Dicionário para armazenar os resultados de cada função
    results = {}

    for excel_file in excel_files:
        function_name = os.path.basename(excel_file).split('_')[1]
        stats = extract_statistics_from_excel(excel_file)
        results[function_name] = stats

    # Criar um DataFrame do pandas para organizar os resultados
    df_results = pd.DataFrame(results)
    
    # Transformar em LaTeX usando a função mencionada
    toLatex(df_results, 'resultado_final.tex')

if __name__ == '__main__':
    main()
