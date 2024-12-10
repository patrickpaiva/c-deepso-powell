import pandas as pd
import matplotlib.pyplot as plt
import os

plt.rcParams.update({
    'font.size': 14,  # Tamanho geral do texto
    'axes.titlesize': 16,  # Tamanho do título do gráfico
    'axes.labelsize': 12,  # Tamanho dos rótulos dos eixos
    'xtick.labelsize': 14,  # Tamanho das marcações do eixo X
    'ytick.labelsize': 14,  # Tamanho das marcações do eixo Y
    'legend.fontsize': 12,  # Tamanho da legenda
})

# Criar a pasta de saída, se não existir
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# Lista de funções e caminhos para os arquivos
functions = [
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

# Iterar sobre as funções para gerar os gráficos
for func in functions:
    file_path_pcdeepso = f"experimento_{func}_dimensoes_pcdeepso_convergencia.csv"
    file_path_cdeepso = f"experimento_{func}_dimensoes_cdeepso_convergencia.csv"
    
    try:
        # Carregar os arquivos CSV
        data_pcdeepso = pd.read_csv(file_path_pcdeepso)
        data_cdeepso = pd.read_csv(file_path_cdeepso)

        # Extrair as colunas de convergência média
        convergencia_media_pcdeepso = data_pcdeepso['Convergencia_Media']
        convergencia_media_cdeepso = data_cdeepso['Convergencia_Media']

        # Definir os marcos e seus rótulos
        milestones = [int(1.2e5) - 1, int(3.6e5) - 1, int(6e5) - 1, int(1.8e6) - 1, int(3e6) - 1]  # Índices ajustados para Python
        milestone_labels = ['0.12', '0.36', '0.6', '1.8', '3.0']

        # Criar o gráfico
        plt.figure(figsize=(10, 6))
        plt.plot(convergencia_media_pcdeepso, color='blue', label='C-DEEPSO-Powell')  # Plotar a curva para PC-DEEPSO
        plt.plot(convergencia_media_cdeepso, color='red', label='C-DEEPSO')  # Plotar a curva para C-DEEPSO

        # Adicionar os marcos no eixo X
        plt.xticks(milestones, milestone_labels)

        # Configurar a escala logarítmica no eixo Y
        plt.yscale('log')

        # Configurações do gráfico
        plt.xlabel("Avaliações de Função ($10^6$)")
        plt.ylabel("Valor de Fitness")
        plt.legend()
        plt.grid(True)

        # Salvar o gráfico como imagem PNG
        output_file = os.path.join(output_dir, f"comparacao_convergencia_{func}.png")
        plt.savefig(output_file, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Gráfico salvo: {output_file}")

    except FileNotFoundError as e:
        print(f"Erro: Arquivo não encontrado para a função {func}. {e}")
    except Exception as e:
        print(f"Erro ao processar a função {func}. {e}")
