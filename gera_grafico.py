import pandas as pd
import matplotlib.pyplot as plt

# Carregar o arquivo CSV
file_path = "experimento_f1_shifted_elliptic_1000_dimensoes_pcdeepso_convergencia.csv"
data = pd.read_csv(file_path)

# Extrair a coluna de convergência média
convergencia_media = data['Convergencia_Media']

# Definir os marcos e seus rótulos
milestones = [1.2e5, 6e5, 3e6]
milestone_labels = ['1.2x10^5', '6x10^5', '3x10^6']

# Criar o gráfico
plt.figure(figsize=(10, 6))
plt.plot(convergencia_media, label='Convergência Média')

# Marcar os marcos no eixo X
for milestone, label in zip(milestones, milestone_labels):
    plt.axvline(x=milestone, color='red', linestyle='--', linewidth=0.8, label=f'Marco: {label}')

# Configurações do gráfico
plt.title("Gráfico de Convergência Média")
plt.xlabel("Avaliações de Função")
plt.ylabel("Convergência Média")
plt.legend()
plt.grid(True)

# Mostrar o gráfico
plt.show()
