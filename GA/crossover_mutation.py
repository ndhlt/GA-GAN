import torch

# Gaussian Crossoverの実装
def gaussian_crossover(parent1_features, parent2_features):
    mu = torch.randn_like(parent1_features)
    child_features = mu * parent1_features + (1 - mu) * parent2_features
    return child_features

# Simulated Binary Crossoverの実装
def simulated_binary_crossover(parent1_features, parent2_features):
    beta = torch.rand(1).item()
    child1 = 0.5 * ((1 + beta) * parent1_features + (1 - beta) * parent2_features)
    child2 = 0.5 * ((1 - beta) * parent1_features + (1 + beta) * parent2_features)
    return child1, child2

# 動的突然変異の実装
def dynamic_mutation(features, mutation_rate=0.1):
    mutation_strength = mutation_rate * torch.randn_like(features)
    mutated_features = features + mutation_strength
    return mutated_features