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
        return re.sub(r'F0?(\d*)$', r'$f_{\\1}$', name)

    funs = [name for name in global_df.columns if name.startswith('F')]
    milestones = [1.2e5, 6e5, 3e6]  # Ajustar para os marcos corretos (120K, 600K e 3M)
    total = 5 * len(milestones)  # Best, Median, Worst, Mean, Std para cada marco
    df_latex = pd.DataFrame(columns=['Milestone', 'Category'] + funs, index=np.arange(total))
    i = 0

    # Agora que temos 'Milestone' no DataFrame, usamos isso para agrupar
    for milestone, df in global_df.groupby(['Milestone']):
        if milestone[0] not in milestones:
            continue

        df_latex.loc[i] = extract(milestone[0], 'Best', df.min(), funs)
        df_latex.loc[i+1] = extract(milestone[0], 'Median', df.median(), funs)
        df_latex.loc[i+2] = extract(milestone[0], 'Worst', df.max(), funs)
        df_latex.loc[i+3] = extract(milestone[0], 'Mean', df.mean(), funs)
        df_latex.loc[i+4] = extract(milestone[0], 'StDev', df.std(), funs)
        df_latex[df_latex.isnull()] = 0  # Substituir possíveis valores NaN por 0
        i += 5
    
    print(df_latex)
    # Substituir nomes de funções no formato correto para LaTeX
    df_latex.columns = [fun2latex(col) for col in df_latex.columns]
    df_latex['Milestone'] = df_latex['Milestone'].map('{:.2e}'.format)
    df_latex = df_latex.set_index(['Milestone', 'Category'])
    
    # Escrever o arquivo LaTeX
    df_latex.to_latex(filename_latex,  multirow=True, multicolumn=False, escape=False)
    print(f"Tabela LaTeX gerada com sucesso em {filename_latex}")
    return



def extract_statistics_from_excel(file_path):
    """
    Extrair os dados das três planilhas específicas para simular milestones.
    """
    df_stats = pd.read_excel(file_path, sheet_name='Estatisticas')
    df_stats_120k = pd.read_excel(file_path, sheet_name='Estatisticas_120k')
    df_stats_600k = pd.read_excel(file_path, sheet_name='Estatisticas_600k')

    # Organizar os dados para os três marcos de avaliação
    data = {
        '3e6': {
            'Best': df_stats['Minimo'].values[0],
            'Median': df_stats['Mediana'].values[0],
            'Worst': df_stats['Maximo'].values[0],
            'Mean': df_stats['Media'].values[0],
            'Std': df_stats['Desvio_Padrao'].values[0],
        },
        '1.2e5': {
            'Best': df_stats_120k['Minimo'].values[0],
            'Median': df_stats_120k['Mediana'].values[0],
            'Worst': df_stats_120k['Maximo'].values[0],
            'Mean': df_stats_120k['Media'].values[0],
            'Std': df_stats_120k['Desvio_Padrao'].values[0],
        },
        '6e5': {
            'Best': df_stats_600k['Minimo'].values[0],
            'Median': df_stats_600k['Mediana'].values[0],
            'Worst': df_stats_600k['Maximo'].values[0],
            'Mean': df_stats_600k['Media'].values[0],
            'Std': df_stats_600k['Desvio_Padrao'].values[0],
        }
    }
    
    # Adicionar um print para verificar o que está sendo extraído
    # print(f"Extraído de {file_path}: {data}")
    return data


def main():
    folder_path = './'  # Colocar o caminho para a pasta com os arquivos .xlsx
    excel_files = glob(os.path.join(folder_path, "*.xlsx"))
    
    # Dicionário para armazenar os resultados de cada função
    results = {}

    for excel_file in excel_files:
        function_name = os.path.basename(excel_file).split('_')[1]
        stats = extract_statistics_from_excel(excel_file)

        # Armazenar as estatísticas separadamente para cada milestone (120k, 600k, 3M)
        if 'F{}'.format(function_name) not in results:
            results['F{}'.format(function_name)] = {}

        # Adiciona os dados extraídos a cada marco
        for milestone, values in stats.items():
            results['F{}'.format(function_name)][milestone] = values

    # Agora temos um dicionário com resultados organizados por função e por marco
    # Criar um DataFrame organizado a partir dos resultados
    global_df = pd.DataFrame({
        'Milestone': [milestone for func in results.values() for milestone in [1.2e5, 6e5, 3e6]],
        'Function': [func_name for func_name in results.keys() for _ in range(3)],
        'Best': [stats['Best'] for func in results.values() for stats in func.values()],
        'Median': [stats['Median'] for func in results.values() for stats in func.values()],
        'Worst': [stats['Worst'] for func in results.values() for stats in func.values()],
        'Mean': [stats['Mean'] for func in results.values() for stats in func.values()],
        'Std': [stats['Std'] for func in results.values() for stats in func.values()],
    })

    print(global_df)

    # Transforma em LaTeX
    toLatex(global_df, 'resultado_final.tex')


if __name__ == '__main__':
    main()
