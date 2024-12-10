import numpy as np
import bisect
from concurrent.futures import ProcessPoolExecutor, as_completed
from random import choice
from sklearn.cluster import KMeans
from utils import generatePopulation, generateMultiplicationMatrix
from powell_method import powell
import gc
from tqdm import tqdm
import tracemalloc
from pympler import tracker
import traceback

def c_deepso_powell_global_best_com_limite(function, dimension, swarmSize, lowerBound, upperBound,
             max_iter=100,
             W_i=0.4019092098808389,
             W_a=0.3791940368874607,
             W_c=0.7539312405916303,
             max_v=1.01,
             T_com=0.5819630448962767,
             T_mut=0.1,
             type='sgpb',
             localSearch=False,
             searchRadius=0.5962189101390463,
             localSearchStartIter=99,
             localSearchEndIter=99,
             F=0.5):
    k = 0
    swarm = generatePopulation(dimension, swarmSize, lowerBound, upperBound)
    velocity = np.zeros((swarmSize, dimension))
    p_best = swarm.copy()
    p_best_fitness = np.array([function(p) for p in p_best])
    g_best = p_best[np.argmin(p_best_fitness)].copy()
    g_best_fitness_list = []
    velocities = []
    positions = []
    function_evals = 0
    improvement_tol = 1e-3
    no_improvement_counter = 0

    def evaluate_function(particle):
        nonlocal function_evals
        function_evals += 1
        return function(particle)

    g_best_fitness = evaluate_function(g_best)
    prev_g_best_fitness = g_best_fitness

    # Inicializa a lista ordenada para as 10% melhores partículas
    num_top_particles = max(1, swarmSize // 10)
    sorted_smb_particles = sorted((p_best_fitness[i], p_best[i].copy()) for i in range(num_top_particles))

    while k < max_iter:
        for i in range(swarmSize):
            # Mutação dos pesos
            W_i = np.clip(W_i + (T_mut * np.random.normal(0, 1)), 0, 1)
            W_a = np.clip(W_a + (T_mut * np.random.normal(0, 1)), 0, 1)
            W_c = np.clip(W_c + (T_mut * np.random.normal(0, 1)), 0, 1)

            # Atualiza Particle Best e Global Best
            particle = swarm[i]
            fitness = evaluate_function(particle)

            if fitness < p_best_fitness[i]:
                p_best[i] = particle.copy()
                p_best_fitness[i] = fitness
                if fitness < g_best_fitness:
                    g_best = particle.copy()
                    g_best_fitness = fitness

            # Atualiza lista com as 10% melhores partículas utilizando busca binária
            if fitness < sorted_smb_particles[-1][0]:
                sorted_smb_particles.pop()  # Remove o pior (último) elemento
                indice = bisect.bisect_left(sorted_smb_particles, fitness, key=lambda i: i[0])
                sorted_smb_particles.insert(indice, (fitness, particle.copy()))

            # Gera Matriz de Comunicação
            C = generateMultiplicationMatrix(dimension, T_com)

            # Extrai Xr conforme o tipo selecionado
            Xr = np.zeros((dimension))

            r = np.random.randint(0, swarmSize)
            while r == i:
                r = np.random.randint(0, swarmSize)

            if type == 'sg':  # Extraído aleatoriamente da população atual
                Xr = swarm[r].copy()
            elif type == 'pb':  # Extraído aleatoriamente dentre os 10% melhores salvos
                Xr = choice([p for _, p in sorted_smb_particles])
            elif type == 'sgpb':  # Média entre Sg e Pb
                pb = choice([p for _, p in sorted_smb_particles])
                Xr = ((swarm[r] + pb) / 2).copy()
            else:
                return "Tipo inválido. Aceitos: sg, pb e sgpb."

            if evaluate_function(Xr) > fitness:  # Fica com Xr apenas se o fitness dele for menor que o da minha partícula corrente
                Xr = particle.copy()

            # Realiza possível mutação do global best
            selected_global_best = g_best
            mutated_global_best = np.clip(g_best * (1 + T_mut * np.random.normal(0, 1)), lowerBound, upperBound)
            if evaluate_function(mutated_global_best) < g_best_fitness:
                selected_global_best = mutated_global_best

            # Implementa a estratégia current-to-best para o Xst
            r1, r2 = np.random.randint(0, swarmSize), np.random.randint(0, swarmSize)
            while r1 == i:
                r1 = np.random.randint(0, swarmSize)
            while r2 == i or r2 == r1:
                r2 = np.random.randint(0, swarmSize)

            X_best = g_best.copy()
            X_r1 = swarm[r1].copy()
            X_r2 = swarm[r2].copy()

            X_st = Xr + F * (X_best - Xr)

            # Calcula vetor de velocidade da partícula
            inertia = W_i * velocity[i]
            cognitive = W_a * (X_st - particle)
            social = W_c * (C @ (selected_global_best - particle))

            velocity[i] = np.clip(inertia + cognitive + social, -max_v, max_v)

            # Atualiza posição somando a velocidade calculada
            swarm[i] = np.clip(particle + velocity[i], lowerBound, upperBound)

        # Avalia se está na hora de mudar para o powell
        improvement = abs(g_best_fitness - prev_g_best_fitness) / abs(prev_g_best_fitness)
        if improvement < improvement_tol:
            no_improvement_counter += 1
        else:
            no_improvement_counter = 0

        if no_improvement_counter >= 10 or k == max_iter - 1:
          result = powell(evaluate_function, g_best, (lowerBound, upperBound))
          candidate = result.copy()
          candidate_fitness = evaluate_function(candidate)
          # result = fmin_powell(evaluate_function, g_best, full_output=1, maxiter=100)
          # candidate = result[0].copy()
          # candidate_fitness = result[1]
          if candidate_fitness < g_best_fitness:
              g_best = candidate.copy()
              g_best_fitness = candidate_fitness
              break

        g_best_fitness_list.append(g_best_fitness)
        positions.append(swarm.copy())
        velocities.append(velocity.copy())
        prev_g_best_fitness = g_best_fitness
        k += 1

    return g_best_fitness, g_best, g_best_fitness_list, positions, velocities, function_evals

def c_deepso_powell_global_best_com_kmeans(function, dimension, swarmSize, lowerBound, upperBound,
             dispersion_threshold,
             max_iter=None,
             max_fun_evals=None,
             W_i=0.4019092098808389,
             W_a=0.3791940368874607,
             W_c=0.7539312405916303,
             max_v=1.01,
             T_com=0.5819630448962767,
             T_mut=0.1,
             type='sgpb',
             F=0.5):
    k = 0
    swarm = generatePopulation(dimension, swarmSize, lowerBound, upperBound)
    velocity = np.zeros((swarmSize, dimension))
    p_best = swarm.copy()
    p_best_fitness = np.array([function(p) for p in p_best])
    g_best = p_best[np.argmin(p_best_fitness)].copy()
    g_best_fitness_list = []
    velocities = []
    positions = []
    function_evals = 0

    if max_iter is None and max_fun_evals is None:
        max_iter = 100
    
    def get_function_evals():
        nonlocal function_evals
        return function_evals
    
    def evaluate_function(particle):
        nonlocal function_evals
        function_evals += 1
        return function(particle)

    g_best_fitness = evaluate_function(g_best)

    # Inicializa a lista ordenada para as 10% melhores partículas
    num_top_particles = max(1, swarmSize // 10)
    sorted_smb_particles = sorted((p_best_fitness[i], p_best[i].copy()) for i in range(num_top_particles))

    while True:
        for i in range(swarmSize):
            # Mutação dos pesos
            W_i = np.clip(W_i + (T_mut * np.random.normal(0, 1)), 0, 1)
            W_a = np.clip(W_a + (T_mut * np.random.normal(0, 1)), 0, 1)
            W_c = np.clip(W_c + (T_mut * np.random.normal(0, 1)), 0, 1)

            # Atualiza Particle Best e Global Best
            particle = swarm[i]
            fitness = evaluate_function(particle)

            if fitness < p_best_fitness[i]:
                p_best[i] = particle.copy()
                p_best_fitness[i] = fitness
                if fitness < g_best_fitness:
                    g_best = particle.copy()
                    g_best_fitness = fitness

            # Atualiza lista com as 10% melhores partículas utilizando busca binária
            if fitness < sorted_smb_particles[-1][0]:
                sorted_smb_particles.pop()  # Remove o pior (último) elemento
                indice = bisect.bisect_left(sorted_smb_particles, fitness, key=lambda i: i[0])
                sorted_smb_particles.insert(indice, (fitness, particle.copy()))

            # Gera Matriz de Comunicação
            C = generateMultiplicationMatrix(dimension, T_com)

            # Extrai Xr conforme o tipo selecionado
            Xr = np.zeros((dimension))

            r = np.random.randint(0, swarmSize)
            while r == i:
                r = np.random.randint(0, swarmSize)

            if type == 'sg':  # Extraído aleatoriamente da população atual
                Xr = swarm[r].copy()
            elif type == 'pb':  # Extraído aleatoriamente dentre os 10% melhores salvos
                Xr = choice([p for _, p in sorted_smb_particles])
            elif type == 'sgpb':  # Média entre Sg e Pb
                pb = choice([p for _, p in sorted_smb_particles])
                Xr = ((swarm[r] + pb) / 2).copy()
            else:
                return "Tipo inválido. Aceitos: sg, pb e sgpb."

            if evaluate_function(Xr) > fitness:  # Fica com Xr apenas se o fitness dele for menor que o da minha partícula corrente
                Xr = particle.copy()

            # Realiza possível mutação do global best
            selected_global_best = g_best
            mutated_global_best = np.clip(g_best * (1 + T_mut * np.random.normal(0, 1)), lowerBound, upperBound)
            if evaluate_function(mutated_global_best) < g_best_fitness:
                selected_global_best = mutated_global_best

            # Implementa a estratégia current-to-best para o Xst
            r1, r2 = np.random.randint(0, swarmSize), np.random.randint(0, swarmSize)
            while r1 == i:
                r1 = np.random.randint(0, swarmSize)
            while r2 == i or r2 == r1:
                r2 = np.random.randint(0, swarmSize)

            X_best = g_best.copy()
            X_r1 = swarm[r1].copy()
            X_r2 = swarm[r2].copy()

            X_st = Xr + F * (X_best - Xr)

            # Calcula vetor de velocidade da partícula
            inertia = W_i * velocity[i]
            cognitive = W_a * (X_st - particle)
            social = W_c * (C @ (selected_global_best - particle))

            velocity[i] = np.clip(inertia + cognitive + social, -max_v, max_v)

            # Atualiza posição somando a velocidade calculada
            swarm[i] = np.clip(particle + velocity[i], lowerBound, upperBound)

        # Avalia se está na hora de mudar para o powell
        if k > 10 and k % 5 == 0:  # Pode ajustar o intervalo de verificação
          kmeans = KMeans(n_clusters=2, n_init=10)  # Número de clusters pode ser ajustado
          swarm_positions = np.array(swarm)
          kmeans.fit(swarm_positions)
          labels = kmeans.labels_
          cluster_centers = kmeans.cluster_centers_

          # Calcula a dispersão entre os clusters
          intra_cluster_distance = np.mean([np.linalg.norm(swarm_positions[i] - cluster_centers[labels[i]])
                                            for i in range(swarmSize)])

          # Define um limiar de dispersão que, quando atingido, entrega para o Powell
          print(f"V1 - Geração: {k} Dispersão: {intra_cluster_distance} fitness: {g_best_fitness} funcalls: {function_evals} maxiter: {max_iter}")
          if intra_cluster_distance < dispersion_threshold \
            or (max_iter is not None and k == (max_iter - 5)) \
            or (max_fun_evals is not None and (function_evals >= max_fun_evals*0.7)):
              print(f"V1 - Aconteceu na iteracao {k}")
              result = powell(evaluate_function, g_best, (lowerBound, upperBound), max_fun_evals, get_function_evals)
              candidate = result.copy()
              candidate_fitness = evaluate_function(candidate)
              if candidate_fitness < g_best_fitness:
                  g_best = candidate.copy()
                  g_best_fitness = candidate_fitness
                  g_best_fitness_list.append(g_best_fitness)
              break

        g_best_fitness_list.append(g_best_fitness)
        positions.append(swarm.copy())
        velocities.append(velocity.copy())
        k += 1

        if max_iter is not None and max_iter == k:
            break
        
        if max_fun_evals is not None and max_fun_evals <= function_evals:
            break

    return g_best_fitness, g_best, g_best_fitness_list, positions, velocities, function_evals

def c_deepso_powell_global_best_com_kmeans_v4(function, dimension, swarmSize, lowerBound, upperBound,
             dispersion_tol = 1e-4,
             max_iter=None,
             max_fun_evals=None,
             W_i=0.4019092098808389,
             W_a=0.3791940368874607,
             W_c=0.7539312405916303,
             max_v=1.01,
             T_com=0.5819630448962767,
             T_mut=0.1,
             type='sgpb',
             F=0.5):
    k = 0
    swarm = generatePopulation(dimension, swarmSize, lowerBound, upperBound)
    velocity = np.zeros((swarmSize, dimension))
    p_best = swarm.copy()
    p_best_fitness = np.array([function(p) for p in p_best])
    g_best = p_best[np.argmin(p_best_fitness)].copy()
    g_best_fitness_list = []
    g_best_fitness = np.min(p_best_fitness)
    velocities = []
    positions = []
    function_evals = 0
    past_intra_cluster_distance = None

    if max_iter is None and max_fun_evals is None:
        max_iter = 100
    
    def get_function_evals():
        nonlocal function_evals
        return function_evals
    
    def evaluate_function(particle):
        nonlocal function_evals
        nonlocal max_iter
        nonlocal g_best_fitness
        nonlocal g_best_fitness_list
        function_evals += 1
        if max_iter is None:
            g_best_fitness_list.append(g_best_fitness)
        return function(particle)

    g_best_fitness = evaluate_function(g_best)

    # Inicializa a lista ordenada para as 10% melhores partículas
    num_top_particles = max(1, swarmSize // 10)
    sorted_smb_particles = sorted((p_best_fitness[i], p_best[i].copy()) for i in range(num_top_particles))

    while True:
        for i in range(swarmSize):
            # Mutação dos pesos
            W_i = np.clip(W_i + (T_mut * np.random.normal(0, 1)), 0, 1)
            W_a = np.clip(W_a + (T_mut * np.random.normal(0, 1)), 0, 1)
            W_c = np.clip(W_c + (T_mut * np.random.normal(0, 1)), 0, 1)

            # Atualiza Particle Best e Global Best
            particle = swarm[i]
            fitness = evaluate_function(particle)

            if fitness < p_best_fitness[i]:
                p_best[i] = particle.copy()
                p_best_fitness[i] = fitness
                if fitness < g_best_fitness:
                    g_best = particle.copy()
                    g_best_fitness = fitness

            # Atualiza lista com as 10% melhores partículas utilizando busca binária
            if fitness < sorted_smb_particles[-1][0]:
                sorted_smb_particles.pop()  # Remove o pior (último) elemento
                indice = bisect.bisect_left(sorted_smb_particles, fitness, key=lambda i: i[0])
                sorted_smb_particles.insert(indice, (fitness, particle.copy()))

            # Gera Matriz de Comunicação
            C = generateMultiplicationMatrix(dimension, T_com)

            # Extrai Xr conforme o tipo selecionado
            Xr = np.zeros((dimension))

            r = np.random.randint(0, swarmSize)
            while r == i:
                r = np.random.randint(0, swarmSize)

            if type == 'sg':  # Extraído aleatoriamente da população atual
                Xr = swarm[r].copy()
            elif type == 'pb':  # Extraído aleatoriamente dentre os 10% melhores salvos
                Xr = choice([p for _, p in sorted_smb_particles])
            elif type == 'sgpb':  # Média entre Sg e Pb
                pb = choice([p for _, p in sorted_smb_particles])
                Xr = ((swarm[r] + pb) / 2).copy()
            else:
                return "Tipo inválido. Aceitos: sg, pb e sgpb."

            if evaluate_function(Xr) > fitness:  # Fica com Xr apenas se o fitness dele for menor que o da minha partícula corrente
                Xr = particle.copy()

            # Realiza possível mutação do global best
            selected_global_best = g_best
            mutated_global_best = np.clip(g_best * (1 + T_mut * np.random.normal(0, 1)), lowerBound, upperBound)
            if evaluate_function(mutated_global_best) < g_best_fitness:
                selected_global_best = mutated_global_best

            # Implementa a estratégia current-to-best para o Xst
            r1, r2 = np.random.randint(0, swarmSize), np.random.randint(0, swarmSize)
            while r1 == i:
                r1 = np.random.randint(0, swarmSize)
            while r2 == i or r2 == r1:
                r2 = np.random.randint(0, swarmSize)

            X_best = g_best.copy()
            X_r1 = swarm[r1].copy()
            X_r2 = swarm[r2].copy()

            X_st = Xr + F * (X_best - Xr)

            # Calcula vetor de velocidade da partícula
            inertia = W_i * velocity[i]
            cognitive = W_a * (X_st - particle)
            social = W_c * (C @ (selected_global_best - particle))

            velocity[i] = np.clip(inertia + cognitive + social, -max_v, max_v)

            # Atualiza posição somando a velocidade calculada
            swarm[i] = np.clip(particle + velocity[i], lowerBound, upperBound)

        # Avalia se está na hora de mudar para o powell
        if k > 10 and k % 5 == 0:
          kmeans = KMeans(n_clusters=2, n_init=10)
          swarm_positions = np.array(swarm)
          kmeans.fit(swarm_positions)
          labels = kmeans.labels_
          cluster_centers = kmeans.cluster_centers_

          
          intra_cluster_distance = np.mean([np.linalg.norm(swarm_positions[i] - cluster_centers[labels[i]])
                                            for i in range(swarmSize)])
        # bnd = ftol * (np.abs(f_old) + np.abs(f_ret)) + 1e-20
        # if 2.0 * (f_old - f_ret) <= bnd:
        #     break
        #   print(f"V1 - Geração: {k} Dispersão: {intra_cluster_distance} Past_disp:{past_intra_cluster_distance} fitness: {g_best_fitness} funcalls: {function_evals} maxiter: {max_iter}")
          if past_intra_cluster_distance is not None:
             change_bound = dispersion_tol * (np.abs(past_intra_cluster_distance) + np.abs(intra_cluster_distance)) + 1e-20
             if (2.0 * (past_intra_cluster_distance - intra_cluster_distance) <= change_bound) \
                or (max_iter is not None and k == (max_iter - 5)) \
                or (max_fun_evals is not None and (function_evals >= max_fun_evals*0.7)):
          # Define um limiar de dispersão que, quando atingido, entrega para o Powell
        #   if intra_cluster_distance < dispersion_threshold \
        #     or (max_iter is not None and k == (max_iter - 5)) \
        #     or (max_fun_evals is not None and (function_evals >= max_fun_evals*0.7)):
            #   print(f"V1 - Aconteceu na iteracao {k}")
              result = powell(evaluate_function, g_best, (lowerBound, upperBound), max_fun_evals, get_function_evals)
              candidate = result.copy()
              candidate_fitness = evaluate_function(candidate)
              if candidate_fitness < g_best_fitness:
                  g_best = candidate.copy()
                  g_best_fitness = candidate_fitness
                  if max_iter is not None:
                    g_best_fitness_list.append(g_best_fitness)
              break
          past_intra_cluster_distance = intra_cluster_distance

        if max_iter is not None:
            g_best_fitness_list.append(g_best_fitness)
        positions.append(swarm.copy())
        velocities.append(velocity.copy())
        k += 1

        if max_iter is not None and max_iter == k:
            break
        
        if max_fun_evals is not None and max_fun_evals <= function_evals:
            break

    return g_best_fitness, g_best, g_best_fitness_list, positions, velocities, function_evals

def c_deepso_powell_global_best_com_kmeans_v3(function, dimension, swarmSize, lowerBound, upperBound,
             dispersion_threshold,
             max_iter=None,
             max_fun_evals=None,
             W_i=0.4019092098808389,
             W_a=0.3791940368874607,
             W_c=0.7539312405916303,
             max_v=1.01,
             T_com=0.5819630448962767,
             T_mut=0.1,
             type='sgpb',
             F=0.5,
             diff_window=3):
    k = 0
    swarm = generatePopulation(dimension, swarmSize, lowerBound, upperBound)
    velocity = np.zeros((swarmSize, dimension))
    p_best = swarm.copy()
    p_best_fitness = np.array([function(p) for p in p_best])
    g_best = p_best[np.argmin(p_best_fitness)].copy()
    g_best_fitness_list = []
    velocities = []
    positions = []
    function_evals = 0
    dispersions = []
    dispersion_diffs = []

    if max_iter is None and max_fun_evals is None:
        max_iter = 100
    
    def get_function_evals():
        nonlocal function_evals
        return function_evals
    
    def evaluate_function(particle):
        nonlocal function_evals
        function_evals += 1
        return function(particle)

    g_best_fitness = evaluate_function(g_best)

    # Inicializa a lista ordenada para as 10% melhores partículas
    num_top_particles = max(1, swarmSize // 10)
    sorted_smb_particles = sorted((p_best_fitness[i], p_best[i].copy()) for i in range(num_top_particles))

    while True:
        for i in range(swarmSize):
            # Mutação dos pesos
            W_i = np.clip(W_i + (T_mut * np.random.normal(0, 1)), 0, 1)
            W_a = np.clip(W_a + (T_mut * np.random.normal(0, 1)), 0, 1)
            W_c = np.clip(W_c + (T_mut * np.random.normal(0, 1)), 0, 1)

            # Atualiza Particle Best e Global Best
            particle = swarm[i]
            fitness = evaluate_function(particle)

            if fitness < p_best_fitness[i]:
                p_best[i] = particle.copy()
                p_best_fitness[i] = fitness
                if fitness < g_best_fitness:
                    g_best = particle.copy()
                    g_best_fitness = fitness

            # Atualiza lista com as 10% melhores partículas utilizando busca binária
            if fitness < sorted_smb_particles[-1][0]:
                sorted_smb_particles.pop()  # Remove o pior (último) elemento
                indice = bisect.bisect_left(sorted_smb_particles, fitness, key=lambda i: i[0])
                sorted_smb_particles.insert(indice, (fitness, particle.copy()))

            # Gera Matriz de Comunicação
            C = generateMultiplicationMatrix(dimension, T_com)

            # Extrai Xr conforme o tipo selecionado
            Xr = np.zeros((dimension))

            r = np.random.randint(0, swarmSize)
            while r == i:
                r = np.random.randint(0, swarmSize)

            if type == 'sg':  # Extraído aleatoriamente da população atual
                Xr = swarm[r].copy()
            elif type == 'pb':  # Extraído aleatoriamente dentre os 10% melhores salvos
                Xr = choice([p for _, p in sorted_smb_particles])
            elif type == 'sgpb':  # Média entre Sg e Pb
                pb = choice([p for _, p in sorted_smb_particles])
                Xr = ((swarm[r] + pb) / 2).copy()
            else:
                return "Tipo inválido. Aceitos: sg, pb e sgpb."

            if evaluate_function(Xr) > fitness:  # Fica com Xr apenas se o fitness dele for menor que o da minha partícula corrente
                Xr = particle.copy()

            # Realiza possível mutação do global best
            selected_global_best = g_best
            mutated_global_best = np.clip(g_best * (1 + T_mut * np.random.normal(0, 1)), lowerBound, upperBound)
            if evaluate_function(mutated_global_best) < g_best_fitness:
                selected_global_best = mutated_global_best

            # Implementa a estratégia current-to-best para o Xst
            r1, r2 = np.random.randint(0, swarmSize), np.random.randint(0, swarmSize)
            while r1 == i:
                r1 = np.random.randint(0, swarmSize)
            while r2 == i or r2 == r1:
                r2 = np.random.randint(0, swarmSize)

            X_best = g_best.copy()
            X_r1 = swarm[r1].copy()
            X_r2 = swarm[r2].copy()

            X_st = Xr + F * (X_best - Xr)

            # Calcula vetor de velocidade da partícula
            inertia = W_i * velocity[i]
            cognitive = W_a * (X_st - particle)
            social = W_c * (C @ (selected_global_best - particle))

            velocity[i] = np.clip(inertia + cognitive + social, -max_v, max_v)

            # Atualiza posição somando a velocidade calculada
            swarm[i] = np.clip(particle + velocity[i], lowerBound, upperBound)

        # Avalia se está na hora de mudar para o powell
        if k > 10 and k % 5 == 0:  # Pode ajustar o intervalo de verificação
          kmeans = KMeans(n_clusters=2, n_init=10)  # Número de clusters pode ser ajustado
          swarm_positions = np.array(swarm)
          kmeans.fit(swarm_positions)
          labels = kmeans.labels_
          cluster_centers = kmeans.cluster_centers_

          # Calcula a dispersão entre os clusters
          intra_cluster_distance = np.mean([np.linalg.norm(swarm_positions[i] - cluster_centers[labels[i]])
                                            for i in range(swarmSize)])

          dispersions.append(intra_cluster_distance)

          if len(dispersions) > 1:
                dispersion_diffs.append(abs(dispersions[-1] - dispersions[-2]))
                
          # Verifica a estabilidade das últimas diferenças
          if len(dispersion_diffs) >= diff_window:
              recent_diffs = dispersion_diffs[-diff_window:]
              max_diff = max(recent_diffs)
              min_diff = min(recent_diffs)
              
              # Se as últimas diferenças estiverem abaixo do limiar e forem próximas
              diff = max_diff - min_diff
              print(f'geração: {k} diff: {diff} dispersion: {intra_cluster_distance} fitness: {g_best_fitness} funcalls: {function_evals}')
              if diff <= dispersion_threshold\
                or (max_iter is not None and k == (max_iter - 5)) \
                or (max_fun_evals is not None and (function_evals >= max_fun_evals*0.7)):
                  print(f'Aconteceu na geração: {k}')
                  result = powell(evaluate_function, g_best, (lowerBound, upperBound), max_fun_evals, get_function_evals)
                  candidate = result.copy()
                  candidate_fitness = evaluate_function(candidate)
                  if candidate_fitness < g_best_fitness:
                      g_best = candidate.copy()
                      g_best_fitness = candidate_fitness
                      g_best_fitness_list.append(g_best_fitness)
                  break

        g_best_fitness_list.append(g_best_fitness)
        positions.append(swarm.copy())
        velocities.append(velocity.copy())
        k += 1

        if max_iter is not None and max_iter == k:
            break
        
        if max_fun_evals is not None and max_fun_evals <= function_evals:
            break

    return g_best_fitness, g_best, g_best_fitness_list, positions, velocities, function_evals

def c_deepso_powell_global_best_com_kmeans_v2(function, dimension, swarmSize, lowerBound, upperBound,
             max_iter=None,
             max_fun_evals=None,
             W_i=0.4019092098808389,
             W_a=0.3791940368874607,
             W_c=0.7539312405916303,
             max_v=1.01,
             T_com=0.5819630448962767,
             T_mut=0.1,
             type='sgpb',
             F=0.5):
    k = 0
    swarm = generatePopulation(dimension, swarmSize, lowerBound, upperBound)
    velocity = np.zeros((swarmSize, dimension))
    p_best = swarm.copy()
    p_best_fitness = np.array([function(p) for p in p_best])
    g_best = p_best[np.argmin(p_best_fitness)].copy()
    g_best_fitness_list = []
    velocities = []
    positions = []
    function_evals = 0
    D_max = np.sqrt(dimension)

    if max_iter is None and max_fun_evals is None:
        max_iter = 100
    
    def get_function_evals():
        nonlocal function_evals
        return function_evals
    
    def evaluate_function(particle):
        nonlocal function_evals
        function_evals += 1
        return function(particle)

    g_best_fitness = evaluate_function(g_best)

    # Inicializa a lista ordenada para as 10% melhores partículas
    num_top_particles = max(1, swarmSize // 10)
    sorted_smb_particles = sorted((p_best_fitness[i], p_best[i].copy()) for i in range(num_top_particles))

    while True:
        for i in range(swarmSize):
            # Mutação dos pesos
            W_i = np.clip(W_i + (T_mut * np.random.normal(0, 1)), 0, 1)
            W_a = np.clip(W_a + (T_mut * np.random.normal(0, 1)), 0, 1)
            W_c = np.clip(W_c + (T_mut * np.random.normal(0, 1)), 0, 1)

            # Atualiza Particle Best e Global Best
            particle = swarm[i]
            fitness = evaluate_function(particle)

            if fitness < p_best_fitness[i]:
                p_best[i] = particle.copy()
                p_best_fitness[i] = fitness
                if fitness < g_best_fitness:
                    g_best = particle.copy()
                    g_best_fitness = fitness

            # Atualiza lista com as 10% melhores partículas utilizando busca binária
            if fitness < sorted_smb_particles[-1][0]:
                sorted_smb_particles.pop()  # Remove o pior (último) elemento
                indice = bisect.bisect_left(sorted_smb_particles, fitness, key=lambda i: i[0])
                sorted_smb_particles.insert(indice, (fitness, particle.copy()))

            # Gera Matriz de Comunicação
            C = generateMultiplicationMatrix(dimension, T_com)

            # Extrai Xr conforme o tipo selecionado
            Xr = np.zeros((dimension))

            r = np.random.randint(0, swarmSize)
            while r == i:
                r = np.random.randint(0, swarmSize)

            if type == 'sg':  # Extraído aleatoriamente da população atual
                Xr = swarm[r].copy()
            elif type == 'pb':  # Extraído aleatoriamente dentre os 10% melhores salvos
                Xr = choice([p for _, p in sorted_smb_particles])
            elif type == 'sgpb':  # Média entre Sg e Pb
                pb = choice([p for _, p in sorted_smb_particles])
                Xr = ((swarm[r] + pb) / 2).copy()
            else:
                return "Tipo inválido. Aceitos: sg, pb e sgpb."

            if evaluate_function(Xr) > fitness:  # Fica com Xr apenas se o fitness dele for menor que o da minha partícula corrente
                Xr = particle.copy()

            # Realiza possível mutação do global best
            selected_global_best = g_best
            mutated_global_best = np.clip(g_best * (1 + T_mut * np.random.normal(0, 1)), lowerBound, upperBound)
            if evaluate_function(mutated_global_best) < g_best_fitness:
                selected_global_best = mutated_global_best

            # Implementa a estratégia current-to-best para o Xst
            r1, r2 = np.random.randint(0, swarmSize), np.random.randint(0, swarmSize)
            while r1 == i:
                r1 = np.random.randint(0, swarmSize)
            while r2 == i or r2 == r1:
                r2 = np.random.randint(0, swarmSize)

            X_best = g_best.copy()
            X_r1 = swarm[r1].copy()
            X_r2 = swarm[r2].copy()

            X_st = Xr + F * (X_best - Xr)

            # Calcula vetor de velocidade da partícula
            inertia = W_i * velocity[i]
            cognitive = W_a * (X_st - particle)
            social = W_c * (C @ (selected_global_best - particle))

            velocity[i] = np.clip(inertia + cognitive + social, -max_v, max_v)

            # Atualiza posição somando a velocidade calculada
            swarm[i] = np.clip(particle + velocity[i], lowerBound, upperBound)

        # Avalia se está na hora de mudar para o powell
        if k > 10 and k % 5 == 0:  # Pode ajustar o intervalo de verificação
            swarm_positions = swarm.copy()
            current_lower_bound = np.min(swarm_positions, axis=0)
            current_upper_bound = np.max(swarm_positions, axis=0)

            normalized_swarm_positions = (swarm_positions - current_lower_bound) / (current_upper_bound - current_lower_bound)

            kmeans = KMeans(n_clusters=2, n_init=10)
            kmeans.fit(normalized_swarm_positions)
            labels = kmeans.labels_
            cluster_centers = kmeans.cluster_centers_

            intra_cluster_distance = np.mean([np.linalg.norm(normalized_swarm_positions[i] - cluster_centers[labels[i]]) 
                                  for i in range(swarmSize)])
            
            proximity_threshold = 0.15 * D_max

            # print(f"V2 - Geração: {k} Dispersão: {intra_cluster_distance} LIMITE: {proximity_threshold}")
            if intra_cluster_distance < proximity_threshold:
                # print(f"V2 - Aconteceu na iteracao {k}")
                result = powell(evaluate_function, g_best, (lowerBound, upperBound), max_fun_evals, get_function_evals)
                candidate = result.copy()
                candidate_fitness = evaluate_function(candidate)
                if candidate_fitness < g_best_fitness:
                    g_best = candidate.copy()
                    g_best_fitness = candidate_fitness
                    g_best_fitness_list.append(g_best_fitness)
                break

        g_best_fitness_list.append(g_best_fitness)
        positions.append(swarm.copy())
        velocities.append(velocity.copy())
        k += 1

        if max_iter is not None and max_iter == k:
            break
        
        if max_fun_evals is not None and max_fun_evals <= function_evals:
            break

    return g_best_fitness, g_best, g_best_fitness_list, positions, velocities, function_evals

def c_deepso(function, dimension, swarmSize, lowerBound, upperBound,
             max_iter=None,
             max_fun_evals=None,
             W_i=0.4019092098808389,
             W_a=0.3791940368874607,
             W_c=0.7539312405916303,
             max_v=1.01,
             T_com=0.5819630448962767,
             T_mut=0.1,
             type='sgpb',
             F=0.5):
    k = 0
    swarm = generatePopulation(dimension, swarmSize, lowerBound, upperBound)
    velocity = np.zeros((swarmSize, dimension))
    p_best = swarm.copy()
    p_best_fitness = np.array([function(p) for p in p_best])
    g_best = p_best[np.argmin(p_best_fitness)].copy()
    g_best_fitness_list = []
    velocities = []
    positions = []
    function_evals = 0
    g_best_fitness = np.min(p_best_fitness)

    if max_iter is None and max_fun_evals is None:
        max_iter = 100

    def evaluate_function(particle):
        nonlocal function_evals
        nonlocal max_iter
        nonlocal g_best_fitness
        nonlocal g_best_fitness_list
        result = function(particle)
        function_evals += 1
        if result < g_best_fitness:
            g_best_fitness_list.append(result)
        else:
            g_best_fitness_list.append(g_best_fitness)
        if function_evals >= max_fun_evals:
            raise MaxFunEvalsReached
        return result

    # Inicializa a lista ordenada para as 10% melhores partículas
    num_top_particles = max(1, swarmSize // 10)
    sorted_smb_particles = sorted((p_best_fitness[i], p_best[i].copy()) for i in range(num_top_particles))

    while True:
        try:
            for i in range(swarmSize):
                # Mutação dos pesos
                W_i = np.clip(W_i + (T_mut * np.random.normal(0, 1)), 0, 1)
                W_a = np.clip(W_a + (T_mut * np.random.normal(0, 1)), 0, 1)
                W_c = np.clip(W_c + (T_mut * np.random.normal(0, 1)), 0, 1)

                # Atualiza Particle Best e Global Best
                particle = swarm[i]
                fitness = evaluate_function(particle)

                if fitness < p_best_fitness[i]:
                    p_best[i] = particle.copy()
                    p_best_fitness[i] = fitness
                    if fitness < g_best_fitness:
                        g_best = particle.copy()
                        g_best_fitness = fitness
                        g_best_fitness_list[-1] = g_best_fitness

                # Atualiza lista com as 10% melhores partículas utilizando busca binária
                if fitness < sorted_smb_particles[-1][0]:
                    sorted_smb_particles.pop()  # Remove o pior (último) elemento
                    indice = bisect.bisect_left(sorted_smb_particles, fitness, key=lambda i: i[0])
                    sorted_smb_particles.insert(indice, (fitness, particle.copy()))

                # Gera Matriz de Comunicação
                C = generateMultiplicationMatrix(dimension, T_com)

                # Extrai Xr conforme o tipo selecionado
                Xr = np.zeros((dimension))

                r = np.random.randint(0, swarmSize)
                while r == i:
                    r = np.random.randint(0, swarmSize)

                if type == 'sg':  # Extraído aleatoriamente da população atual
                    Xr = swarm[r].copy()
                elif type == 'pb':  # Extraído aleatoriamente dentre os 10% melhores salvos
                    Xr = choice([p for _, p in sorted_smb_particles])
                elif type == 'sgpb':  # Média entre Sg e Pb
                    pb = choice([p for _, p in sorted_smb_particles])
                    Xr = ((swarm[r] + pb) / 2).copy()
                else:
                    return "Tipo inválido. Aceitos: sg, pb e sgpb."

                if evaluate_function(Xr) > fitness:  # Fica com Xr apenas se o fitness dele for menor que o da minha partícula corrente
                    Xr = particle.copy()

                # Realiza possível mutação do global best
                selected_global_best = g_best
                mutated_global_best = np.clip(g_best * (1 + T_mut * np.random.normal(0, 1)), lowerBound, upperBound)
                if evaluate_function(mutated_global_best) < g_best_fitness:
                    selected_global_best = mutated_global_best

                # Implementa a estratégia current-to-best para o Xst
                r1, r2 = np.random.randint(0, swarmSize), np.random.randint(0, swarmSize)
                while r1 == i:
                    r1 = np.random.randint(0, swarmSize)
                while r2 == i or r2 == r1:
                    r2 = np.random.randint(0, swarmSize)

                X_best = g_best.copy()
                X_r1 = swarm[r1].copy()
                X_r2 = swarm[r2].copy()

                X_st = Xr + F * (X_best - Xr)

                # Calcula vetor de velocidade da partícula
                inertia = W_i * velocity[i]
                cognitive = W_a * (X_st - particle)
                social = W_c * (C @ (selected_global_best - particle))

                velocity[i] = np.clip(inertia + cognitive + social, -max_v, max_v)

                # Atualiza posição somando a velocidade calculada
                swarm[i] = np.clip(particle + velocity[i], lowerBound, upperBound)

            positions.append(swarm.copy())
            velocities.append(velocity.copy())
            k += 1

            if max_iter is not None and max_iter == k:
                break
            
            if max_fun_evals is not None and max_fun_evals <= function_evals:
                break
        
        except MaxFunEvalsReached:
            return g_best_fitness, g_best, g_best_fitness_list, positions, velocities, function_evals

    return g_best_fitness, g_best, g_best_fitness_list, positions, velocities, function_evals

def c_deepso_powell_global_best(function, dimension, swarmSize, lowerBound, upperBound,
             percent_powell_start_moment = 0.5,
             percent_powell_func_evals = 0.15,
             max_iter=None,
             max_fun_evals=None,
             W_i=0.4019092098808389,
             W_a=0.3791940368874607,
             W_c=0.7539312405916303,
             max_v=1.01,
             T_com=0.5819630448962767,
             T_mut=0.1,
             type='sgpb',
             F=0.5):
    k = 0
    swarm = generatePopulation(dimension, swarmSize, lowerBound, upperBound)
    velocity = np.zeros((swarmSize, dimension))
    p_best = swarm.copy()
    p_best_fitness = np.array([function(p) for p in p_best])
    g_best = p_best[np.argmin(p_best_fitness)].copy()
    g_best_fitness_list = []
    g_best_fitness = np.min(p_best_fitness)
    velocities = []
    positions = []
    function_evals = 0
    powell_func_evals = int(max_fun_evals * percent_powell_func_evals) if max_fun_evals is not None else None
    powell_start_func_evals = int(max_fun_evals*percent_powell_start_moment) if max_fun_evals is not None else None
    powell_stop_func_evals = int(powell_start_func_evals + powell_func_evals) if max_fun_evals is not None else None
    g_best_fitness_120k_evals = None
    g_best_fitness_600k_evals = None
    

    if max_iter is None and max_fun_evals is None:
        max_iter = 100
    
    def get_function_evals():
        nonlocal function_evals
        return function_evals
    
    def evaluate_function(particle):
        nonlocal function_evals
        nonlocal max_iter
        nonlocal g_best_fitness
        nonlocal g_best_fitness_list
        nonlocal g_best_fitness_120k_evals, g_best_fitness_600k_evals
        result = function(particle)
        function_evals += 1
        if result < g_best_fitness:
            g_best_fitness_list.append(result)
        else:
            g_best_fitness_list.append(g_best_fitness)
        if function_evals == 120_000:
            g_best_fitness_120k_evals = g_best_fitness
        if function_evals == 600_000:
            g_best_fitness_600k_evals = g_best_fitness
        if function_evals >= max_fun_evals:
            raise MaxFunEvalsReached
        return result

    # Inicializa a lista ordenada para as 10% melhores partículas
    num_top_particles = max(1, swarmSize // 10)
    sorted_smb_particles = sorted((p_best_fitness[i], p_best[i].copy()) for i in range(num_top_particles))

    while True:
        try:
            for i in range(swarmSize):
                # Mutação dos pesos
                W_i = np.clip(W_i + (T_mut * np.random.normal(0, 1)), 0, 1)
                W_a = np.clip(W_a + (T_mut * np.random.normal(0, 1)), 0, 1)
                W_c = np.clip(W_c + (T_mut * np.random.normal(0, 1)), 0, 1)

                # Atualiza Particle Best e Global Best
                particle = swarm[i]
                fitness = evaluate_function(particle)

                if fitness < p_best_fitness[i]:
                    p_best[i] = particle.copy()
                    p_best_fitness[i] = fitness
                    if fitness < g_best_fitness:
                        g_best = particle.copy()
                        g_best_fitness = fitness
                        g_best_fitness_list[-1] = g_best_fitness

                # Atualiza lista com as 10% melhores partículas utilizando busca binária
                if fitness < sorted_smb_particles[-1][0]:
                    sorted_smb_particles.pop()  # Remove o pior (último) elemento
                    indice = bisect.bisect_left(sorted_smb_particles, fitness, key=lambda i: i[0])
                    sorted_smb_particles.insert(indice, (fitness, particle.copy()))

                # Gera Matriz de Comunicação
                C = generateMultiplicationMatrix(dimension, T_com)

                # Extrai Xr conforme o tipo selecionado
                Xr = np.zeros((dimension))

                r = np.random.randint(0, swarmSize)
                while r == i:
                    r = np.random.randint(0, swarmSize)

                if type == 'sg':  # Extraído aleatoriamente da população atual
                    Xr = swarm[r].copy()
                elif type == 'pb':  # Extraído aleatoriamente dentre os 10% melhores salvos
                    Xr = choice([p for _, p in sorted_smb_particles])
                elif type == 'sgpb':  # Média entre Sg e Pb
                    pb = choice([p for _, p in sorted_smb_particles])
                    Xr = ((swarm[r] + pb) / 2).copy()
                else:
                    return "Tipo inválido. Aceitos: sg, pb e sgpb."

                if evaluate_function(Xr) > fitness:  # Fica com Xr apenas se o fitness dele for menor que o da minha partícula corrente
                    Xr = particle.copy()

                # Realiza possível mutação do global best
                selected_global_best = g_best
                mutated_global_best = np.clip(g_best * (1 + T_mut * np.random.normal(0, 1)), lowerBound, upperBound)
                if evaluate_function(mutated_global_best) < g_best_fitness:
                    selected_global_best = mutated_global_best

                # Implementa a estratégia current-to-best para o Xst
                r1, r2 = np.random.randint(0, swarmSize), np.random.randint(0, swarmSize)
                while r1 == i:
                    r1 = np.random.randint(0, swarmSize)
                while r2 == i or r2 == r1:
                    r2 = np.random.randint(0, swarmSize)

                X_best = g_best.copy()
                X_r1 = swarm[r1].copy()
                X_r2 = swarm[r2].copy()

                X_st = Xr + F * (X_best - Xr)

                # Calcula vetor de velocidade da partícula
                inertia = W_i * velocity[i]
                cognitive = W_a * (X_st - particle)
                social = W_c * (C @ (selected_global_best - particle))

                velocity[i] = np.clip(inertia + cognitive + social, -max_v, max_v)

                # Atualiza posição somando a velocidade calculada
                swarm[i] = np.clip(particle + velocity[i], lowerBound, upperBound)

                # print(f"Geração: {k} funcalls: {function_evals} powell_start: {powell_start_func_evals} powell_stop: {powell_stop_func_evals}")
                if max_fun_evals is not None and (function_evals >= powell_start_func_evals and function_evals <= powell_stop_func_evals):
                    # print(f"Aconteceu na iteracao {k} com {function_evals} fun evals")
                    result = powell(evaluate_function, g_best, (lowerBound, upperBound), powell_stop_func_evals, get_function_evals)
                    candidate = result.copy()
                    candidate_fitness = evaluate_function(candidate)
                    if candidate_fitness < g_best_fitness:
                        g_best = candidate.copy()
                        g_best_fitness = candidate_fitness
                        g_best_fitness_list[-1] = g_best_fitness
                    # print(f"Terminou com {function_evals} fun evals")
                    

                # if max_iter is not None and k == max_iter//2:
                #     result = powell(evaluate_function, g_best, (lowerBound, upperBound), max_fun_evals, get_function_evals)
                #     candidate = result.copy()
                #     candidate_fitness = evaluate_function(candidate)
                #     if candidate_fitness < g_best_fitness:
                #         g_best = candidate.copy()
                #         g_best_fitness = candidate_fitness
                #         g_best_fitness_list.append(g_best_fitness)
                #     break

            if max_iter is not None:
                g_best_fitness_list.append(g_best_fitness)
            # positions.append(swarm.copy())
            # velocities.append(velocity.copy())
            k += 1

            if max_iter is not None and max_iter == k:
                break
            
            if max_fun_evals is not None and max_fun_evals <= function_evals:
                break
        
        except MaxFunEvalsReached:
            return g_best_fitness, g_best, g_best_fitness_list, positions, velocities, function_evals, g_best_fitness_120k_evals, g_best_fitness_600k_evals

    return g_best_fitness, g_best, g_best_fitness_list, positions, velocities, function_evals, g_best_fitness_120k_evals, g_best_fitness_600k_evals

def c_deepso_powell_global_best_paralelo_metade(function, dimension, swarmSize, lowerBound, upperBound, id_execucao=0,
             percent_powell_start_moment = 0.5,
             percent_powell_func_evals = 0.15,
             max_iter=None,
             max_fun_evals=None,
             W_i=0.4019092098808389,
             W_a=0.3791940368874607,
             W_c=0.7539312405916303,
             max_v=1.01,
             T_com=0.5819630448962767,
             T_mut=0.1,
             type='sgpb',
             F=0.5):
    k = 0
    swarm = generatePopulation(dimension, swarmSize, lowerBound, upperBound)
    velocity = np.zeros((swarmSize, dimension))
    g_best_fitness_list = []
    fitness_list = []
    velocities = []
    positions = []
    function_evals = 0
    powell_func_evals = int(max_fun_evals * percent_powell_func_evals) if max_fun_evals is not None else None
    powell_start_func_evals = int(max_fun_evals*percent_powell_start_moment) if max_fun_evals is not None else None
    powell_stop_func_evals = int(powell_start_func_evals + powell_func_evals) if max_fun_evals is not None else None
    g_best_fitness_120k_evals = None
    g_best_fitness_600k_evals = None
    g_best_fitness_before_powell = None
    g_best_fitness_after_powell = None
    previous_func_evals = 0

    if max_iter is None and max_fun_evals is None:
        max_iter = 100
    
    def get_function_evals():
        nonlocal function_evals
        return function_evals
    
    def evaluate_function(particle):
        nonlocal function_evals
        nonlocal max_iter
        nonlocal g_best_fitness, g_best
        nonlocal g_best_fitness_list
        nonlocal g_best_fitness_120k_evals, g_best_fitness_600k_evals
        result = function(particle)
        function_evals += 1
        if result < g_best_fitness:
            g_best_fitness = result
            g_best = particle
            g_best_fitness_list.append(result)
        else:
            g_best_fitness_list.append(g_best_fitness)
        if function_evals == 120_000:
            g_best_fitness_120k_evals = g_best_fitness
        if function_evals == 600_000:
            g_best_fitness_600k_evals = g_best_fitness
        if function_evals >= max_fun_evals:
            if result < g_best_fitness:
                g_best_fitness = result
                g_best = particle
            raise MaxFunEvalsReached
        return result
    
    def evaluate_population(swarm):
        nonlocal function_evals
        nonlocal fitness_list
        nonlocal g_best_fitness
        nonlocal g_best
        nonlocal g_best_fitness_120k_evals, g_best_fitness_600k_evals

        remaining_evals = max_fun_evals - function_evals

        if remaining_evals < swarmSize:
            fitness_list = function(swarm[:remaining_evals])
            function_evals += remaining_evals
        else:
            fitness_list = function(swarm)
            function_evals += swarmSize

        if function_evals >= 120_000 and g_best_fitness_120k_evals is None:
            g_best_fitness_120k_evals = g_best_fitness
        if function_evals >= 600_000 and g_best_fitness_600k_evals is None:
            g_best_fitness_600k_evals = g_best_fitness

        if function_evals >= max_fun_evals:
            candidate_g_best_index = np.argmin(fitness_list)
            candidate_g_best_fitness = fitness_list[candidate_g_best_index]
            if candidate_g_best_fitness < g_best_fitness:
                g_best_fitness = candidate_g_best_fitness
                g_best = swarm[candidate_g_best_index]
            g_best_fitness_list.extend([g_best_fitness] * len(fitness_list))
            raise MaxFunEvalsReached

    evaluate_population(swarm)
    g_best_index = np.argmin(fitness_list)
    g_best = swarm[g_best_index]
    g_best_fitness = fitness_list[g_best_index]
    g_best_fitness_list.extend([g_best_fitness] * len(fitness_list))    

    # Inicializa a lista ordenada para as 10% melhores partículas
    num_top_particles = max(1, swarmSize // 10)

    def get_top_10_percent_particles(swarm, fitness_list, num_top_particles):
        sorted_indices = np.argsort(fitness_list)
        top_indices = sorted_indices[:num_top_particles]
        top_particles = swarm[top_indices]
        return top_particles, top_indices
    
    position_tqdm = (id_execucao % 5) + 1
    with tqdm(total=max_fun_evals, position=position_tqdm, desc=f"Execução {id_execucao}", unit="evals", leave=False) as pbar:
        while True:
            try:
                # Mutação dos pesos
                W_i = np.clip(W_i + (T_mut * np.random.normal(0, 1)), 0, 1)
                W_a = np.clip(W_a + (T_mut * np.random.normal(0, 1)), 0, 1)
                W_c = np.clip(W_c + (T_mut * np.random.normal(0, 1)), 0, 1)

                # Extrai Xr conforme o tipo selecionado
                
                
                Xr_matrix = None
                top_indices = None
                top_particles = None

                if type == 'sg':
                    random_indices = np.random.randint(0, swarmSize, size=(swarmSize,))
                    same_particle_mask = (random_indices == np.arange(swarmSize))
                    while np.any(same_particle_mask):
                        random_indices[same_particle_mask] = np.random.randint(0, swarmSize, size=np.sum(same_particle_mask))
                        same_particle_mask = (random_indices == np.arange(swarmSize))
                    
                    Xr_matrix = swarm[random_indices]
                elif type == 'pb':
                    top_particles, top_indices = get_top_10_percent_particles(swarm, fitness_list, num_top_particles)
                    random_top_indices = np.random.randint(0, len(top_particles), size=(swarmSize,))
                    same_particle_mask = (top_indices[random_top_indices] == np.arange(swarmSize))
                    while np.any(same_particle_mask):
                        random_top_indices[same_particle_mask] = np.random.randint(0, len(top_particles), size=np.sum(same_particle_mask))
                        same_particle_mask = (top_indices[random_top_indices] == np.arange(swarmSize))
                    
                    Xr_matrix = top_particles[random_top_indices]
                elif type == 'sgpb':  
                    random_indices = np.random.randint(0, swarmSize, size=(swarmSize,))
                    
                    same_particle_mask = (random_indices == np.arange(swarmSize))
                    while np.any(same_particle_mask):
                        random_indices[same_particle_mask] = np.random.randint(0, swarmSize, size=np.sum(same_particle_mask))
                        same_particle_mask = (random_indices == np.arange(swarmSize))
                    
                    Xr_matrix = swarm[random_indices]
                    top_particles, top_indices = get_top_10_percent_particles(swarm, fitness_list, num_top_particles)
                    random_top_indices = np.random.randint(0, len(top_particles), size=(swarmSize,))
                    
                    same_particle_mask = (top_indices[random_top_indices] == np.arange(swarmSize))
                    while np.any(same_particle_mask):
                        random_top_indices[same_particle_mask] = np.random.randint(0, len(top_particles), size=np.sum(same_particle_mask))
                        same_particle_mask = (top_indices[random_top_indices] == np.arange(swarmSize))

                    Xr_matrix = (Xr_matrix + top_particles[random_top_indices]) / 2
                
                # Pega X_r1 e X_r2
                # random_indices_r1 = np.random.randint(0, swarmSize, size=(swarmSize,))
                # random_indices_r2 = np.random.randint(0, swarmSize, size=(swarmSize,))
                # same_particle_mask_r1 = (random_indices_r1 == np.arange(swarmSize))
                # same_particle_mask_r2 = (random_indices_r2 == np.arange(swarmSize))
                # while np.any(same_particle_mask_r1) or np.any(same_particle_mask_r2):
                #     random_indices_r1[same_particle_mask_r1] = np.random.randint(0, swarmSize, size=np.sum(same_particle_mask_r1))
                #     random_indices_r2[same_particle_mask_r2] = np.random.randint(0, swarmSize, size=np.sum(same_particle_mask_r2))
                #     same_particle_mask_r1 = (random_indices_r1 == np.arange(swarmSize))
                #     same_particle_mask_r2 = (random_indices_r2 == np.arange(swarmSize))
                # Xr1_matrix = swarm[random_indices_r1]
                # Xr2_matrix = swarm[random_indices_r2]
                
                # Calcula X_st com a estratégia current-to-best-2
                X_st_matrix = Xr_matrix + F * (g_best - Xr_matrix) #+ F * (Xr1_matrix - Xr2_matrix)
                # del Xr1_matrix, Xr2_matrix
                # Gera matriz/vetor de comunicação
                C = np.random.choice([0, 1], size=dimension, p=[1-T_com, T_com])

                selected_global_best = g_best
                mutated_global_best = np.clip(g_best * (1 + T_mut * np.random.normal(0, 1)), lowerBound, upperBound)
                # if evaluate_function(mutated_global_best) < g_best_fitness:
                selected_global_best = mutated_global_best

                # Equação de movimento nas matrizes
                #V = np.clip(wm*V + c1m*(Xst - X) + c2m*C*(gbestm_X - X), -diff, diff)
                # inertia = W_i * velocity[i]
                # cognitive = W_a * (X_st - particle)
                # social = W_c * (C @ (selected_global_best - particle))
                # inertia = np.multiply(W_i * velocity)
                # cognitive = np.multiply(W_a, np.subtract(X_st, swarm))
                # social = np.multiply(W_c, np.multiply(C, np.subtract(selected_global_best, swarm)))
                velocity = np.clip(np.add(np.multiply(W_i, velocity), np.add(np.multiply(W_a, np.subtract(X_st_matrix, swarm)), np.multiply(W_c, np.multiply(C, np.subtract(selected_global_best, swarm))))), -max_v, max_v)
                # velocity = np.clip(W_i * velocity + W_a * (X_st_matrix - swarm) + W_c * C * (selected_global_best - swarm), -max_v, max_v)
                swarm = np.clip(swarm + velocity, lowerBound, upperBound)

                evaluate_population(swarm)

                candidate_g_best_index = np.argmin(fitness_list)
                candidate_g_best_fitness = fitness_list[candidate_g_best_index]
                if candidate_g_best_fitness < g_best_fitness:
                    g_best_fitness = candidate_g_best_fitness
                    g_best = swarm[candidate_g_best_index]
                    g_best_index = candidate_g_best_index
                g_best_fitness_list.extend([g_best_fitness] * len(fitness_list))

                if max_fun_evals is not None and (function_evals >= powell_start_func_evals and function_evals <= powell_stop_func_evals):
                    # print(f"Comecou powell na iter {k} com {function_evals} fun evals e g_best: {g_best_fitness}")
                    # print(f"powell_start: {powell_start_func_evals} powell_stop: {powell_stop_func_evals}")
                    g_best_fitness_before_powell = g_best_fitness
                    result = powell(evaluate_function, g_best, (lowerBound, upperBound), powell_stop_func_evals, get_function_evals)
                    candidate = result.copy()
                    candidate_fitness = evaluate_function(candidate)
                    if candidate_fitness < g_best_fitness:
                        g_best = candidate.copy()
                        g_best_fitness = candidate_fitness
                    g_best_fitness_after_powell = g_best_fitness
                    # print(f"Terminou com {function_evals} fun evals e g_best: {g_best_fitness}")
                    

                # if max_iter is not None and k == max_iter//2:
                #     result = powell(evaluate_function, g_best, (lowerBound, upperBound), max_fun_evals, get_function_evals)
                #     candidate = result.copy()
                #     candidate_fitness = evaluate_function(candidate)
                #     if candidate_fitness < g_best_fitness:
                #         g_best = candidate.copy()
                #         g_best_fitness = candidate_fitness
                #         g_best_fitness_list.append(g_best_fitness)
                #     break

                # if max_iter is not None:
                #     g_best_fitness_list.append(g_best_fitness)
                # positions.append(swarm.copy())
                # velocities.append(velocity.copy())
                k += 1
                pbar.update(function_evals - previous_func_evals)
                previous_func_evals = function_evals
                # print(f'iteração: {k} - função: {function_evals}')

                del C, Xr_matrix, top_particles, X_st_matrix
                gc.collect()
                # pbar.update(function_evals)

                if max_iter is not None and max_iter == k:
                    break
                
                if max_fun_evals is not None and max_fun_evals <= function_evals:
                    break
            
            except MaxFunEvalsReached:
                del fitness_list, velocity, swarm, Xr_matrix, C, top_particles, X_st_matrix
                gc.collect()
                return g_best_fitness, g_best, g_best_fitness_list, positions, velocities, function_evals, g_best_fitness_120k_evals, g_best_fitness_600k_evals, g_best_fitness_before_powell, g_best_fitness_after_powell
            except Exception as e:
                # Loga o erro e a stack trace em um arquivo
                with open("error_log_cdeepso.txt", "a") as log_file:  # Abre o arquivo em modo de anexar
                    log_file.write(f"Erro durante a execução do cdeepso: {str(e)}\n")
                    traceback.print_exc(file=log_file)  # Salva a stack trace no arquivo
                    log_file.write("\n")
                print(f"Erro durante a execução do C-DEEPSO: {str(e)} - Verifique o arquivo error_log_cdeepso.txt para mais detalhes.")
    del fitness_list, velocity, swarm, Xr_matrix, C, top_particles, X_st_matrix
    gc.collect()
    return g_best_fitness, g_best, g_best_fitness_list, positions, velocities, function_evals, g_best_fitness_120k_evals, g_best_fitness_600k_evals, g_best_fitness_before_powell, g_best_fitness_after_powell

def c_deepso_powell_global_best_paralelo(function, dimension, swarmSize, lowerBound, upperBound, id_execucao=0,
             percent_powell_start_moment = 0.5,
             percent_powell_func_evals = 0.1,
             max_iter=None,
             max_fun_evals=None,
             W_i=0.40,
             W_a=0.37,
             W_c=0.75,
             max_v=1.01,
             T_com=0.58,
             T_mut=0.1,
             type='sgpb',
             F=0.8):
    k = 0
    swarm = generatePopulation(dimension, swarmSize, lowerBound, upperBound)
    velocity = np.zeros((swarmSize, dimension))
    g_best_fitness_list = []
    fitness_list = []
    velocities = []
    positions = []
    function_evals = 0
    powell_func_evals = int(max_fun_evals * percent_powell_func_evals) if max_fun_evals is not None else None
    powell_start_func_evals = int(max_fun_evals*percent_powell_start_moment) if max_fun_evals is not None else None
    powell_stop_func_evals = int(powell_start_func_evals + powell_func_evals) if max_fun_evals is not None else None
    g_best_fitness_120k_evals = None
    g_best_fitness_600k_evals = None
    g_best_fitness_before_powell = None
    g_best_fitness_after_powell = None
    previous_func_evals = 0
    powell_primeiro_marco_evals = int(powell_func_evals * 0.04)
    powell_segundo_marco_evals = int(powell_func_evals * 0.2)
    powell_terceiro_marco_evals = int(powell_func_evals * 0.76)

    if max_iter is None and max_fun_evals is None:
        max_iter = 100
    
    def get_function_evals():
        nonlocal function_evals
        return function_evals
    
    def evaluate_function(particle):
        nonlocal function_evals
        nonlocal max_iter
        nonlocal g_best_fitness, g_best
        nonlocal g_best_fitness_list
        nonlocal g_best_fitness_120k_evals, g_best_fitness_600k_evals
        result = function(particle)
        function_evals += 1
        if result < g_best_fitness:
            g_best_fitness = result
            g_best = particle
            g_best_fitness_list.append(result)
        else:
            g_best_fitness_list.append(g_best_fitness)
        if function_evals == 120_000:
            g_best_fitness_120k_evals = g_best_fitness
        if function_evals == 600_000:
            g_best_fitness_600k_evals = g_best_fitness
        if function_evals >= max_fun_evals:
            if result < g_best_fitness:
                g_best_fitness = result
                g_best = particle
            raise MaxFunEvalsReached
        return result
    
    def evaluate_population(swarm):
        nonlocal function_evals
        nonlocal fitness_list
        nonlocal g_best_fitness
        nonlocal g_best
        nonlocal g_best_fitness_120k_evals, g_best_fitness_600k_evals

        remaining_evals = max_fun_evals - function_evals

        if remaining_evals < swarmSize:
            fitness_list = function(swarm[:remaining_evals])
            function_evals += remaining_evals
        else:
            fitness_list = function(swarm)
            function_evals += swarmSize

        if function_evals >= 120_000 and g_best_fitness_120k_evals is None:
            g_best_fitness_120k_evals = g_best_fitness
        if function_evals >= 600_000 and g_best_fitness_600k_evals is None:
            g_best_fitness_600k_evals = g_best_fitness

        if function_evals >= max_fun_evals:
            candidate_g_best_index = np.argmin(fitness_list)
            candidate_g_best_fitness = fitness_list[candidate_g_best_index]
            if candidate_g_best_fitness < g_best_fitness:
                g_best_fitness = candidate_g_best_fitness
                g_best = swarm[candidate_g_best_index]
            g_best_fitness_list.extend([g_best_fitness] * len(fitness_list))
            raise MaxFunEvalsReached

    evaluate_population(swarm)
    g_best_index = np.argmin(fitness_list)
    g_best = swarm[g_best_index]
    g_best_fitness = fitness_list[g_best_index]
    g_best_fitness_list.extend([g_best_fitness] * len(fitness_list))    

    # Inicializa a lista ordenada para as 10% melhores partículas
    num_top_particles = max(1, swarmSize // 10)

    def get_top_10_percent_particles(swarm, fitness_list, num_top_particles):
        sorted_indices = np.argsort(fitness_list)
        top_indices = sorted_indices[:num_top_particles]
        top_particles = swarm[top_indices]
        return top_particles, top_indices
    
    position_tqdm = (id_execucao % 5) + 1
    with tqdm(total=max_fun_evals, position=position_tqdm, desc=f"Execução {id_execucao}", unit="evals", leave=False) as pbar:
        while True:
            try:
                # Mutação dos pesos
                W_i = np.clip(W_i + (T_mut * np.random.normal(0, 1)), 0, 1)
                W_a = np.clip(W_a + (T_mut * np.random.normal(0, 1)), 0, 1)
                W_c = np.clip(W_c + (T_mut * np.random.normal(0, 1)), 0, 1)

                # Extrai Xr conforme o tipo selecionado
                Xr_matrix = None
                top_indices = None
                top_particles = None

                if type == 'sg':
                    random_indices = np.random.randint(0, swarmSize, size=(swarmSize,))
                    same_particle_mask = (random_indices == np.arange(swarmSize))
                    while np.any(same_particle_mask):
                        random_indices[same_particle_mask] = np.random.randint(0, swarmSize, size=np.sum(same_particle_mask))
                        same_particle_mask = (random_indices == np.arange(swarmSize))
                    
                    Xr_matrix = swarm[random_indices]
                elif type == 'pb':
                    top_particles, top_indices = get_top_10_percent_particles(swarm, fitness_list, num_top_particles)
                    random_top_indices = np.random.randint(0, len(top_particles), size=(swarmSize,))
                    same_particle_mask = (top_indices[random_top_indices] == np.arange(swarmSize))
                    while np.any(same_particle_mask):
                        random_top_indices[same_particle_mask] = np.random.randint(0, len(top_particles), size=np.sum(same_particle_mask))
                        same_particle_mask = (top_indices[random_top_indices] == np.arange(swarmSize))
                    
                    Xr_matrix = top_particles[random_top_indices]
                elif type == 'sgpb':  
                    random_indices = np.random.randint(0, swarmSize, size=(swarmSize,))
                    
                    same_particle_mask = (random_indices == np.arange(swarmSize))
                    while np.any(same_particle_mask):
                        random_indices[same_particle_mask] = np.random.randint(0, swarmSize, size=np.sum(same_particle_mask))
                        same_particle_mask = (random_indices == np.arange(swarmSize))
                    
                    Xr_matrix = swarm[random_indices]
                    top_particles, top_indices = get_top_10_percent_particles(swarm, fitness_list, num_top_particles)
                    random_top_indices = np.random.randint(0, len(top_particles), size=(swarmSize,))
                    
                    same_particle_mask = (top_indices[random_top_indices] == np.arange(swarmSize))
                    while np.any(same_particle_mask):
                        random_top_indices[same_particle_mask] = np.random.randint(0, len(top_particles), size=np.sum(same_particle_mask))
                        same_particle_mask = (top_indices[random_top_indices] == np.arange(swarmSize))

                    Xr_matrix = (Xr_matrix + top_particles[random_top_indices]) / 2
                
                # Pega X_r1 e X_r2
                # random_indices_r1 = np.random.randint(0, swarmSize, size=(swarmSize,))
                # random_indices_r2 = np.random.randint(0, swarmSize, size=(swarmSize,))
                # same_particle_mask_r1 = (random_indices_r1 == np.arange(swarmSize))
                # same_particle_mask_r2 = (random_indices_r2 == np.arange(swarmSize))
                # while np.any(same_particle_mask_r1) or np.any(same_particle_mask_r2):
                #     random_indices_r1[same_particle_mask_r1] = np.random.randint(0, swarmSize, size=np.sum(same_particle_mask_r1))
                #     random_indices_r2[same_particle_mask_r2] = np.random.randint(0, swarmSize, size=np.sum(same_particle_mask_r2))
                #     same_particle_mask_r1 = (random_indices_r1 == np.arange(swarmSize))
                #     same_particle_mask_r2 = (random_indices_r2 == np.arange(swarmSize))
                # Xr1_matrix = swarm[random_indices_r1]
                # Xr2_matrix = swarm[random_indices_r2]
                
                # Calcula X_st com a estratégia current-to-best-2
                X_st_matrix = Xr_matrix + F * (g_best - Xr_matrix) #+ F * (Xr1_matrix - Xr2_matrix)

                # Gera matriz/vetor de comunicação
                C = np.random.choice([0, 1], size=dimension, p=[1-T_com, T_com])

                selected_global_best = g_best
                mutated_global_best = np.clip(g_best * (1 + T_mut * np.random.normal(0, 1)), lowerBound, upperBound)
                # if evaluate_function(mutated_global_best) < g_best_fitness:
                selected_global_best = mutated_global_best

                # Equação de movimento nas matrizes
                inertia = W_i * velocity
                cognitive = W_a * (X_st_matrix - swarm)
                social = W_c * C * (selected_global_best - swarm)
                velocity = np.clip(inertia + cognitive + social, -max_v, max_v)
                swarm = np.clip(swarm + velocity, lowerBound, upperBound)

                evaluate_population(swarm)

                candidate_g_best_index = np.argmin(fitness_list)
                candidate_g_best_fitness = fitness_list[candidate_g_best_index]
                if candidate_g_best_fitness < g_best_fitness:
                    g_best_fitness = candidate_g_best_fitness
                    g_best = swarm[candidate_g_best_index]
                    g_best_index = candidate_g_best_index
                g_best_fitness_list.extend([g_best_fitness] * len(fitness_list))

                

                if function_evals >= 60_000 and function_evals <= 60_000 + powell_primeiro_marco_evals \
                    or function_evals >= 360_000 and function_evals <= 360_000 + powell_segundo_marco_evals \
                    or function_evals >= 1_800_000 and function_evals <= 1_800_000 + powell_terceiro_marco_evals:
                    # print(f"Comecou powell na iter {k} com {function_evals} fun evals e g_best: {g_best_fitness}")
                    # print(f"powell_start: {powell_start_func_evals} powell_stop: {powell_stop_func_evals}")
                    if 60000 <= function_evals <= 60000 + powell_primeiro_marco_evals:
                        powell_stop_func_evals = function_evals + powell_primeiro_marco_evals
                    elif 360000 <= function_evals <= 360000 + powell_segundo_marco_evals:
                        powell_stop_func_evals = function_evals + powell_segundo_marco_evals
                    elif 1800000 <= function_evals <= 1800000 + powell_terceiro_marco_evals:
                        powell_stop_func_evals = function_evals + powell_terceiro_marco_evals
                    g_best_fitness_before_powell = g_best_fitness
                    result = powell(evaluate_function, g_best, (lowerBound, upperBound), powell_stop_func_evals, get_function_evals)
                    candidate = result.copy()
                    candidate_fitness = evaluate_function(candidate)
                    if candidate_fitness < g_best_fitness:
                        g_best = candidate.copy()
                        g_best_fitness = candidate_fitness
                    g_best_fitness_after_powell = g_best_fitness
                    # print(f"Terminou com {function_evals} fun evals e g_best: {g_best_fitness}")
                    
                k += 1
                pbar.update(function_evals - previous_func_evals)
                previous_func_evals = function_evals
                # print(f'iteração: {k} - função: {function_evals}')

                del C, Xr_matrix, top_particles, X_st_matrix
                gc.collect()
                # pbar.update(function_evals)

                if max_iter is not None and max_iter == k:
                    break
                
                if max_fun_evals is not None and max_fun_evals <= function_evals:
                    break
            
            except MaxFunEvalsReached:
                del fitness_list, velocity, swarm, Xr_matrix, C, top_particles, X_st_matrix
                gc.collect()
                return g_best_fitness, g_best, g_best_fitness_list, positions, velocities, function_evals, g_best_fitness_120k_evals, g_best_fitness_600k_evals, g_best_fitness_before_powell, g_best_fitness_after_powell
            except Exception as e:
                # Loga o erro e a stack trace em um arquivo
                with open("error_log_cdeepso.txt", "a") as log_file:  # Abre o arquivo em modo de anexar
                    log_file.write(f"Erro durante a execução do cdeepso: {str(e)}\n")
                    traceback.print_exc(file=log_file)  # Salva a stack trace no arquivo
                    log_file.write("\n")
                print(f"Erro durante a execução do C-DEEPSO: {str(e)} - Verifique o arquivo error_log_cdeepso.txt para mais detalhes.")
    del fitness_list, velocity, swarm, Xr_matrix, C, top_particles, X_st_matrix
    gc.collect()
    return g_best_fitness, g_best, g_best_fitness_list, positions, velocities, function_evals, g_best_fitness_120k_evals, g_best_fitness_600k_evals, g_best_fitness_before_powell, g_best_fitness_after_powell

def c_deepso_powell_global_best_paralelo_powell_varias_vezes(function, dimension, swarmSize, lowerBound, upperBound, id_execucao=0,
             percent_powell_start_moment = 0.5,
             percent_powell_func_evals = 0.1,
             max_iter=None,
             max_fun_evals=None,
             W_i=0.4019092098808389,
             W_a=0.3791940368874607,
             W_c=0.7539312405916303,
             max_v=1.01,
             T_com=0.5819630448962767,
             T_mut=0.1,
             type='sgpb',
             F=0.8):
    k = 0
    swarm = generatePopulation(dimension, swarmSize, lowerBound, upperBound)
    velocity = np.zeros((swarmSize, dimension))
    g_best_fitness_list = []
    fitness_list = []
    velocities = []
    positions = []
    function_evals = 0
    powell_func_evals = int(max_fun_evals * percent_powell_func_evals) if max_fun_evals is not None else None
    powell_start_func_evals = int(max_fun_evals*percent_powell_start_moment) if max_fun_evals is not None else None
    powell_stop_func_evals = int(powell_start_func_evals + powell_func_evals) if max_fun_evals is not None else None
    g_best_fitness_120k_evals = None
    g_best_fitness_600k_evals = None
    g_best_fitness_before_powell = None
    g_best_fitness_after_powell = None
    previous_func_evals = 0
    powell_primeiro_marco_evals = int(powell_func_evals * 0.04)
    powell_segundo_marco_evals = int(powell_func_evals * 0.2)
    powell_terceiro_marco_evals = int(powell_func_evals * 0.76)
    next_powell_milestone = 100000

    if max_iter is None and max_fun_evals is None:
        max_iter = 100
    
    def get_function_evals():
        nonlocal function_evals
        return function_evals
    
    def evaluate_function(particle):
        nonlocal function_evals
        nonlocal max_iter
        nonlocal g_best_fitness, g_best
        nonlocal g_best_fitness_list
        nonlocal g_best_fitness_120k_evals, g_best_fitness_600k_evals
        result = function(particle)
        function_evals += 1
        if result < g_best_fitness:
            g_best_fitness = result
            g_best = particle
            g_best_fitness_list.append(result)
        else:
            g_best_fitness_list.append(g_best_fitness)
        if function_evals == 120_000:
            g_best_fitness_120k_evals = g_best_fitness
        if function_evals == 600_000:
            g_best_fitness_600k_evals = g_best_fitness
        if function_evals >= max_fun_evals:
            if result < g_best_fitness:
                g_best_fitness = result
                g_best = particle
            raise MaxFunEvalsReached
        return result
    
    def evaluate_population(swarm):
        nonlocal function_evals
        nonlocal fitness_list
        nonlocal g_best_fitness
        nonlocal g_best
        nonlocal g_best_fitness_120k_evals, g_best_fitness_600k_evals

        remaining_evals = max_fun_evals - function_evals

        if remaining_evals < swarmSize:
            fitness_list = function(swarm[:remaining_evals])
            function_evals += remaining_evals
        else:
            fitness_list = function(swarm)
            function_evals += swarmSize

        if function_evals >= 120_000 and g_best_fitness_120k_evals is None:
            g_best_fitness_120k_evals = g_best_fitness
        if function_evals >= 600_000 and g_best_fitness_600k_evals is None:
            g_best_fitness_600k_evals = g_best_fitness

        if function_evals >= max_fun_evals:
            candidate_g_best_index = np.argmin(fitness_list)
            candidate_g_best_fitness = fitness_list[candidate_g_best_index]
            if candidate_g_best_fitness < g_best_fitness:
                g_best_fitness = candidate_g_best_fitness
                g_best = swarm[candidate_g_best_index]
            g_best_fitness_list.extend([g_best_fitness] * len(fitness_list))
            raise MaxFunEvalsReached

    evaluate_population(swarm)
    g_best_index = np.argmin(fitness_list)
    g_best = swarm[g_best_index]
    g_best_fitness = fitness_list[g_best_index]
    g_best_fitness_list.extend([g_best_fitness] * len(fitness_list))    

    # Inicializa a lista ordenada para as 10% melhores partículas
    num_top_particles = max(1, swarmSize // 10)

    def get_top_10_percent_particles(swarm, fitness_list, num_top_particles):
        sorted_indices = np.argsort(fitness_list)
        top_indices = sorted_indices[:num_top_particles]
        top_particles = swarm[top_indices]
        return top_particles, top_indices
    
    position_tqdm = (id_execucao % 5) + 1
    with tqdm(total=max_fun_evals, position=position_tqdm, desc=f"Execução {id_execucao}", unit="evals", leave=False) as pbar:
        while True:
            try:
                # Mutação dos pesos
                W_i = np.clip(W_i + (T_mut * np.random.normal(0, 1)), 0, 1)
                W_a = np.clip(W_a + (T_mut * np.random.normal(0, 1)), 0, 1)
                W_c = np.clip(W_c + (T_mut * np.random.normal(0, 1)), 0, 1)

                # Extrai Xr conforme o tipo selecionado
                
                
                Xr_matrix = None
                top_indices = None
                top_particles = None

                if type == 'sg':
                    random_indices = np.random.randint(0, swarmSize, size=(swarmSize,))
                    same_particle_mask = (random_indices == np.arange(swarmSize))
                    while np.any(same_particle_mask):
                        random_indices[same_particle_mask] = np.random.randint(0, swarmSize, size=np.sum(same_particle_mask))
                        same_particle_mask = (random_indices == np.arange(swarmSize))
                    
                    Xr_matrix = swarm[random_indices]
                elif type == 'pb':
                    top_particles, top_indices = get_top_10_percent_particles(swarm, fitness_list, num_top_particles)
                    random_top_indices = np.random.randint(0, len(top_particles), size=(swarmSize,))
                    same_particle_mask = (top_indices[random_top_indices] == np.arange(swarmSize))
                    while np.any(same_particle_mask):
                        random_top_indices[same_particle_mask] = np.random.randint(0, len(top_particles), size=np.sum(same_particle_mask))
                        same_particle_mask = (top_indices[random_top_indices] == np.arange(swarmSize))
                    
                    Xr_matrix = top_particles[random_top_indices]
                elif type == 'sgpb':  
                    random_indices = np.random.randint(0, swarmSize, size=(swarmSize,))
                    
                    same_particle_mask = (random_indices == np.arange(swarmSize))
                    while np.any(same_particle_mask):
                        random_indices[same_particle_mask] = np.random.randint(0, swarmSize, size=np.sum(same_particle_mask))
                        same_particle_mask = (random_indices == np.arange(swarmSize))
                    
                    Xr_matrix = swarm[random_indices]
                    top_particles, top_indices = get_top_10_percent_particles(swarm, fitness_list, num_top_particles)
                    random_top_indices = np.random.randint(0, len(top_particles), size=(swarmSize,))
                    
                    same_particle_mask = (top_indices[random_top_indices] == np.arange(swarmSize))
                    while np.any(same_particle_mask):
                        random_top_indices[same_particle_mask] = np.random.randint(0, len(top_particles), size=np.sum(same_particle_mask))
                        same_particle_mask = (top_indices[random_top_indices] == np.arange(swarmSize))

                    Xr_matrix = (Xr_matrix + top_particles[random_top_indices]) / 2
                
                # Pega X_r1 e X_r2
                # random_indices_r1 = np.random.randint(0, swarmSize, size=(swarmSize,))
                # random_indices_r2 = np.random.randint(0, swarmSize, size=(swarmSize,))
                # same_particle_mask_r1 = (random_indices_r1 == np.arange(swarmSize))
                # same_particle_mask_r2 = (random_indices_r2 == np.arange(swarmSize))
                # while np.any(same_particle_mask_r1) or np.any(same_particle_mask_r2):
                #     random_indices_r1[same_particle_mask_r1] = np.random.randint(0, swarmSize, size=np.sum(same_particle_mask_r1))
                #     random_indices_r2[same_particle_mask_r2] = np.random.randint(0, swarmSize, size=np.sum(same_particle_mask_r2))
                #     same_particle_mask_r1 = (random_indices_r1 == np.arange(swarmSize))
                #     same_particle_mask_r2 = (random_indices_r2 == np.arange(swarmSize))
                # Xr1_matrix = swarm[random_indices_r1]
                # Xr2_matrix = swarm[random_indices_r2]
                
                # Calcula X_st com a estratégia current-to-best-2
                X_st_matrix = Xr_matrix + F * (g_best - Xr_matrix) #+ F * (Xr1_matrix - Xr2_matrix)
                #del Xr1_matrix, Xr2_matrix
                # Gera matriz/vetor de comunicação
                C = np.random.choice([0, 1], size=dimension, p=[1-T_com, T_com])

                selected_global_best = g_best
                mutated_global_best = np.clip(g_best * (1 + T_mut * np.random.normal(0, 1)), lowerBound, upperBound)
                # if evaluate_function(mutated_global_best) < g_best_fitness:
                selected_global_best = mutated_global_best

                # Equação de movimento nas matrizes
                #V = np.clip(wm*V + c1m*(Xst - X) + c2m*C*(gbestm_X - X), -diff, diff)
                # inertia = W_i * velocity[i]
                # cognitive = W_a * (X_st - particle)
                # social = W_c * (C @ (selected_global_best - particle))
                # inertia = np.multiply(W_i * velocity)
                # cognitive = np.multiply(W_a, np.subtract(X_st, swarm))
                # social = np.multiply(W_c, np.multiply(C, np.subtract(selected_global_best, swarm)))
                velocity = np.clip(np.add(np.multiply(W_i, velocity), np.add(np.multiply(W_a, np.subtract(X_st_matrix, swarm)), np.multiply(W_c, np.multiply(C, np.subtract(selected_global_best, swarm))))), -max_v, max_v)
                # velocity = np.clip(W_i * velocity + W_a * (X_st_matrix - swarm) + W_c * C * (selected_global_best - swarm), -max_v, max_v)
                swarm = np.clip(swarm + velocity, lowerBound, upperBound)

                evaluate_population(swarm)

                candidate_g_best_index = np.argmin(fitness_list)
                candidate_g_best_fitness = fitness_list[candidate_g_best_index]
                if candidate_g_best_fitness < g_best_fitness:
                    g_best_fitness = candidate_g_best_fitness
                    g_best = swarm[candidate_g_best_index]
                    g_best_index = candidate_g_best_index
                g_best_fitness_list.extend([g_best_fitness] * len(fitness_list))

                

                # Check for powell call
                if function_evals >= next_powell_milestone:
                    # Call powell
                    g_best_fitness_before_powell = g_best_fitness
                    result = powell(evaluate_function, g_best, (lowerBound, upperBound), max_fun_evals, get_function_evals, max_iter=1)
                    candidate = result.copy()
                    candidate_fitness = evaluate_function(candidate)
                    if candidate_fitness < g_best_fitness:
                        g_best = candidate.copy()
                        g_best_fitness = candidate_fitness
                    g_best_fitness_after_powell = g_best_fitness

                    # Update the next milestone
                    next_powell_milestone += 100000
                    # print(f"Terminou com {function_evals} fun evals e g_best: {g_best_fitness}")
                    

                # if max_iter is not None and k == max_iter//2:
                #     result = powell(evaluate_function, g_best, (lowerBound, upperBound), max_fun_evals, get_function_evals)
                #     candidate = result.copy()
                #     candidate_fitness = evaluate_function(candidate)
                #     if candidate_fitness < g_best_fitness:
                #         g_best = candidate.copy()
                #         g_best_fitness = candidate_fitness
                #         g_best_fitness_list.append(g_best_fitness)
                #     break

                # if max_iter is not None:
                #     g_best_fitness_list.append(g_best_fitness)
                # positions.append(swarm.copy())
                # velocities.append(velocity.copy())
                k += 1
                pbar.update(function_evals - previous_func_evals)
                previous_func_evals = function_evals
                # print(f'iteração: {k} - função: {function_evals}')

                del C, Xr_matrix, top_particles, X_st_matrix
                gc.collect()
                # pbar.update(function_evals)

                if max_iter is not None and max_iter == k:
                    break
                
                if max_fun_evals is not None and max_fun_evals <= function_evals:
                    break
            
            except MaxFunEvalsReached:
                del fitness_list, velocity, swarm, Xr_matrix, C, top_particles, X_st_matrix
                gc.collect()
                return g_best_fitness, g_best, g_best_fitness_list, positions, velocities, function_evals, g_best_fitness_120k_evals, g_best_fitness_600k_evals, g_best_fitness_before_powell, g_best_fitness_after_powell
            except Exception as e:
                # Loga o erro e a stack trace em um arquivo
                with open("error_log_cdeepso.txt", "a") as log_file:  # Abre o arquivo em modo de anexar
                    log_file.write(f"Erro durante a execução do cdeepso: {str(e)}\n")
                    traceback.print_exc(file=log_file)  # Salva a stack trace no arquivo
                    log_file.write("\n")
                print(f"Erro durante a execução do C-DEEPSO: {str(e)} - Verifique o arquivo error_log_cdeepso.txt para mais detalhes.")
    del fitness_list, velocity, swarm, Xr_matrix, C, top_particles, X_st_matrix
    gc.collect()
    return g_best_fitness, g_best, g_best_fitness_list, positions, velocities, function_evals, g_best_fitness_120k_evals, g_best_fitness_600k_evals, g_best_fitness_before_powell, g_best_fitness_after_powell

def c_deepso_powell_global_best_paralelo_backup(function, dimension, swarmSize, lowerBound, upperBound, id_execucao=0,
             percent_powell_start_moment = 0.5,
             percent_powell_func_evals = 0.15,
             max_iter=None,
             max_fun_evals=None,
             W_i=0.4019092098808389,
             W_a=0.3791940368874607,
             W_c=0.7539312405916303,
             max_v=1.01,
             T_com=0.5819630448962767,
             T_mut=0.1,
             type='sgpb',
             F=0.5):
    k = 0
    swarm = generatePopulation(dimension, swarmSize, lowerBound, upperBound)
    velocity = np.zeros((swarmSize, dimension))
    g_best_fitness_list = []
    fitness_list = []
    velocities = []
    positions = []
    function_evals = 0
    powell_func_evals = int(max_fun_evals * percent_powell_func_evals) if max_fun_evals is not None else None
    powell_start_func_evals = int(max_fun_evals*percent_powell_start_moment) if max_fun_evals is not None else None
    powell_stop_func_evals = int(powell_start_func_evals + powell_func_evals) if max_fun_evals is not None else None
    g_best_fitness_120k_evals = None
    g_best_fitness_600k_evals = None
    g_best_fitness_before_powell = None
    g_best_fitness_after_powell = None
    previous_func_evals = 0

    if max_iter is None and max_fun_evals is None:
        max_iter = 100
    
    def get_function_evals():
        nonlocal function_evals
        return function_evals
    
    def evaluate_function(particle):
        nonlocal function_evals
        nonlocal max_iter
        nonlocal g_best_fitness, g_best
        nonlocal g_best_fitness_list
        nonlocal g_best_fitness_120k_evals, g_best_fitness_600k_evals
        result = function(particle)
        function_evals += 1
        if result < g_best_fitness:
            g_best_fitness = result
            g_best = particle
            g_best_fitness_list.append(result)
        else:
            g_best_fitness_list.append(g_best_fitness)
        if function_evals == 120_000:
            g_best_fitness_120k_evals = g_best_fitness
        if function_evals == 600_000:
            g_best_fitness_600k_evals = g_best_fitness
        if function_evals >= max_fun_evals:
            if result < g_best_fitness:
                g_best_fitness = result
                g_best = particle
            raise MaxFunEvalsReached
        return result
    
    def evaluate_population(swarm):
        nonlocal function_evals
        nonlocal fitness_list
        nonlocal g_best_fitness
        nonlocal g_best
        nonlocal g_best_fitness_120k_evals, g_best_fitness_600k_evals

        remaining_evals = max_fun_evals - function_evals

        if remaining_evals < swarmSize:
            fitness_list = function(swarm[:remaining_evals])
            function_evals += remaining_evals
        else:
            fitness_list = function(swarm)
            function_evals += swarmSize

        if function_evals >= 120_000 and g_best_fitness_120k_evals is None:
            g_best_fitness_120k_evals = g_best_fitness
        if function_evals >= 600_000 and g_best_fitness_600k_evals is None:
            g_best_fitness_600k_evals = g_best_fitness

        if function_evals >= max_fun_evals:
            candidate_g_best_index = np.argmin(fitness_list)
            candidate_g_best_fitness = fitness_list[candidate_g_best_index]
            if candidate_g_best_fitness < g_best_fitness:
                g_best_fitness = candidate_g_best_fitness
                g_best = swarm[candidate_g_best_index]
            g_best_fitness_list.extend([g_best_fitness] * len(fitness_list))
            raise MaxFunEvalsReached

    evaluate_population(swarm)
    g_best_index = np.argmin(fitness_list)
    g_best = swarm[g_best_index]
    g_best_fitness = fitness_list[g_best_index]
    g_best_fitness_list.extend([g_best_fitness] * len(fitness_list))    

    # Inicializa a lista ordenada para as 10% melhores partículas
    num_top_particles = max(1, swarmSize // 10)

    def get_top_10_percent_particles(swarm, fitness_list, num_top_particles):
        sorted_indices = np.argsort(fitness_list)
        top_indices = sorted_indices[:num_top_particles]
        return swarm[top_indices]
    position_tqdm = (id_execucao % 5) + 1
    with tqdm(total=max_fun_evals, position=position_tqdm, desc=f"Execução {id_execucao}", unit="evals", leave=False) as pbar:
        while True:
            try:
                # Mutação dos pesos
                W_i = np.clip(W_i + (T_mut * np.random.normal(0, 1)), 0, 1)
                W_a = np.clip(W_a + (T_mut * np.random.normal(0, 1)), 0, 1)
                W_c = np.clip(W_c + (T_mut * np.random.normal(0, 1)), 0, 1)

                # # Atualiza Particle Best e Global Best
                # particle = swarm[i]
                # fitness = evaluate_function(particle)

                # if fitness < p_best_fitness[i]:
                #     p_best[i] = particle.copy()
                #     p_best_fitness[i] = fitness
                #     if fitness < g_best_fitness:
                #         g_best = particle.copy()
                #         g_best_fitness = fitness
                #         g_best_fitness_list[-1] = g_best_fitness

                # Gera Matriz de Comunicação
                # C = generateMultiplicationMatrix(dimension, T_com)

                # Extrai Xr conforme o tipo selecionado
                Xr = None
                top_particles = get_top_10_percent_particles(swarm, fitness_list, num_top_particles)

                r = np.random.randint(0, swarmSize)

                if type == 'sg':  # Extraído aleatoriamente da população atual
                    Xr = swarm[r]
                elif type == 'pb':  # Extraído aleatoriamente dentre os 10% melhores salvos
                    Xr = top_particles[np.random.randint(num_top_particles)]
                elif type == 'sgpb':  # Média entre Sg e Pb
                    pb = top_particles[np.random.randint(num_top_particles)]
                    Xr = ((swarm[r] + pb) / 2)
                else:
                    return "Tipo inválido. Aceitos: sg, pb e sgpb."

                # if evaluate_function(Xr) > fitness:  # Fica com Xr apenas se o fitness dele for menor que o da minha partícula corrente
                #     Xr = particle

                # Realiza possível mutação do global best
                selected_global_best = g_best
                mutated_global_best = np.clip(g_best * (1 + T_mut * np.random.normal(0, 1)), lowerBound, upperBound)
                if evaluate_function(mutated_global_best) < g_best_fitness:
                    selected_global_best = mutated_global_best

                #Implementa a estratégia current-to-best para o Xst
                # r1, r2 = np.random.randint(0, swarmSize), np.random.randint(0, swarmSize)
                # X_r1 = swarm[r1]
                # X_r2 = swarm[r2]

                X_st = Xr + F * (g_best - Xr)
                C = np.random.choice([0, 1], size=dimension, p=[1-T_com, T_com])
                # Equação de movimento nas matrizes
                #V = np.clip(wm*V + c1m*(Xst - X) + c2m*C*(gbestm_X - X), -diff, diff)
                # inertia = W_i * velocity[i]
                # cognitive = W_a * (X_st - particle)
                # social = W_c * (C @ (selected_global_best - particle))
                # inertia = np.multiply(W_i * velocity)
                # cognitive = np.multiply(W_a, np.subtract(X_st, swarm))
                # social = np.multiply(W_c, np.multiply(C, np.subtract(selected_global_best, swarm)))
                velocity = np.clip(np.add(np.multiply(W_i, velocity), np.add(np.multiply(W_a, np.subtract(X_st, swarm)), np.multiply(W_c, np.multiply(C, np.subtract(selected_global_best, swarm))))), -max_v, max_v)
                # velocity = np.clip(W_i * velocity + W_a * (X_st - swarm) + W_c * C * (selected_global_best - swarm), -max_v, max_v)
                swarm = np.clip(swarm + velocity, lowerBound, upperBound)

                evaluate_population(swarm)

                candidate_g_best_index = np.argmin(fitness_list)
                candidate_g_best_fitness = fitness_list[candidate_g_best_index]
                if candidate_g_best_fitness < g_best_fitness:
                    g_best_fitness = candidate_g_best_fitness
                    g_best = swarm[candidate_g_best_index]
                    g_best_index = candidate_g_best_index
                g_best_fitness_list.extend([g_best_fitness] * len(fitness_list))

                # if max_fun_evals is not None and (function_evals >= powell_start_func_evals and function_evals <= powell_stop_func_evals):
                #     # print(f"Comecou powell na iter {k} com {function_evals} fun evals e g_best: {g_best_fitness}")
                #     # print(f"powell_start: {powell_start_func_evals} powell_stop: {powell_stop_func_evals}")
                #     g_best_fitness_before_powell = g_best_fitness
                #     result = powell(evaluate_function, g_best, (lowerBound, upperBound), powell_stop_func_evals, get_function_evals)
                #     candidate = result.copy()
                #     candidate_fitness = evaluate_function(candidate)
                #     if candidate_fitness < g_best_fitness:
                #         g_best = candidate.copy()
                #         g_best_fitness = candidate_fitness
                #     g_best_fitness_after_powell = g_best_fitness
                    # print(f"Terminou com {function_evals} fun evals e g_best: {g_best_fitness}")
                    

                # if max_iter is not None and k == max_iter//2:
                #     result = powell(evaluate_function, g_best, (lowerBound, upperBound), max_fun_evals, get_function_evals)
                #     candidate = result.copy()
                #     candidate_fitness = evaluate_function(candidate)
                #     if candidate_fitness < g_best_fitness:
                #         g_best = candidate.copy()
                #         g_best_fitness = candidate_fitness
                #         g_best_fitness_list.append(g_best_fitness)
                #     break

                # if max_iter is not None:
                #     g_best_fitness_list.append(g_best_fitness)
                # positions.append(swarm.copy())
                # velocities.append(velocity.copy())
                k += 1
                pbar.update(function_evals - previous_func_evals)
                previous_func_evals = function_evals
                # print(f'iteração: {k} - função: {function_evals}')

                del C, Xr, top_particles, X_st
                gc.collect()
                # pbar.update(function_evals)

                if max_iter is not None and max_iter == k:
                    break
                
                if max_fun_evals is not None and max_fun_evals <= function_evals:
                    break
            
            except MaxFunEvalsReached:
                del fitness_list, velocity, swarm, Xr, C, top_particles
                gc.collect()
                return g_best_fitness, g_best, g_best_fitness_list, positions, velocities, function_evals, g_best_fitness_120k_evals, g_best_fitness_600k_evals, g_best_fitness_before_powell, g_best_fitness_after_powell
            except Exception as e:
                # Loga o erro e a stack trace em um arquivo
                with open("error_log_cdeepso.txt", "a") as log_file:  # Abre o arquivo em modo de anexar
                    log_file.write(f"Erro durante a execução do cdeepso: {str(e)}\n")
                    traceback.print_exc(file=log_file)  # Salva a stack trace no arquivo
                    log_file.write("\n")
                print(f"Erro durante a execução do C-DEEPSO: {str(e)} - Verifique o arquivo error_log_cdeepso.txt para mais detalhes.")
    del fitness_list, velocity, swarm, Xr, C, top_particles
    gc.collect()
    return g_best_fitness, g_best, g_best_fitness_list, positions, velocities, function_evals, g_best_fitness_120k_evals, g_best_fitness_600k_evals, g_best_fitness_before_powell, g_best_fitness_after_powell


class MaxFunEvalsReached(Exception):
    pass
