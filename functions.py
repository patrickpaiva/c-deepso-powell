import numpy as np

def schwefel_1_2(x):
    D = len(x)
    total_sum = 0
    for i in range(D):
        inner_sum = sum(x[:i+1])  # Somatório de x_j para j=1 até i
        total_sum += inner_sum ** 2
    return total_sum

def shifted_rosenbrock(z):
    D = len(z)
    total = 0
    for i in range(D - 1):
        total += 100 * (z[i]**2 - z[i+1])**2 + (z[i] - 1)**2
    return total

def schwefel(particle):
    dimension = len(particle)
    return 418.9829 * dimension - np.sum(particle * np.sin(np.sqrt(np.abs(particle))))

def griewank(xx):
    d = len(xx)
    sum_term = 0
    prod_term = 1
    
    for ii in range(d):
        xi = xx[ii]
        sum_term += xi**2 / 4000
        prod_term *= np.cos(xi / np.sqrt(ii + 1))
    
    y = sum_term - prod_term + 1
    return y

def shifted_rastrigin(z):
    D = len(z)
    sum_term = np.sum(z**2 - 10 * np.cos(2 * np.pi * z) + 10)
    return sum_term

def shifted_ackley(z):
    D = len(z)
    sum_sq_term = np.sum(z**2) / D
    cos_term = np.sum(np.cos(2 * np.pi * z)) / D
    
    term1 = -20 * np.exp(-0.2 * np.sqrt(sum_sq_term))
    term2 = -np.exp(cos_term)
    
    return term1 + term2 + 20 + np.e

def elliptic_function(x):
    D = len(x)  # Dimensão do vetor x
    result = 0
    for i in range(D):
        result += 10**(6 * (i / (D - 1))) * (x[i]**2)
    return result

