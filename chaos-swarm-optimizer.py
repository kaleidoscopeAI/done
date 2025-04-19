#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CHAOS SWARM OPTIMIZER
A boundary-pushing implementation combining chaos theory, swarm intelligence,
and evolutionary computation to provide a complementary opposite approach
to the Quantum Neural Pathfinder. While QNP seeks order in quantum states,
CSO embraces chaos to discover emergent solutions.
"""

import numpy as np
import networkx as nx
from scipy import stats
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import ctypes
from functools import partial
import os
import sys
import time
import random
import math

# Load C extension for critical calculations in evolutionary processes
try:
    _swarm_lib = ctypes.CDLL('./swarm_evolution.so')
    _has_swarm_lib = True
except:
    _has_swarm_lib = False
    # Will define a Python implementation

# Assembly optimized agent update function (embedded as string)
# This would be compiled and linked at runtime for maximum performance
ASSEMBLY_SWARM_UPDATE = """
section .text
global update_agent_position
update_agent_position:
    ; Arguments passed in:
    ; RDI - pointer to agent position array
    ; RSI - pointer to agent velocity array
    ; RDX - pointer to best position array
    ; RCX - number of dimensions
    ; XMM0 - inertia weight
    ; XMM1 - cognitive coefficient
    ; XMM2 - social coefficient
    
    push    rbp
    mov     rbp, rsp
    push    r12
    push    r13
    push    r14
    
    ; Save parameters
    mov     r12, rdi        ; agent position
    mov     r13, rsi        ; agent velocity
    mov     r14, rdx        ; best position
    mov     r10, rcx        ; dimensions counter
    
    ; Prepare scalar constants in vector registers
    movsd   xmm3, xmm0      ; inertia weight
    movsd   xmm4, xmm1      ; cognitive coefficient
    movsd   xmm5, xmm2      ; social coefficient
    
    ; Get random values for cognitive and social components
    ; (We'd normally call random functions, but for simplicity
    ; we're using constants in this assembly example)
    mov     eax, 0x3F000000  ; ~0.5 in single precision
    movd    xmm6, eax
    cvtss2sd xmm6, xmm6      ; convert to double
    mov     eax, 0x3F400000  ; ~0.75 in single precision
    movd    xmm7, eax
    cvtss2sd xmm7, xmm7      ; convert to double
    
    ; Update each dimension
    xor     rcx, rcx        ; dimension index
    
.update_loop:
    ; Update velocity
    ; v = w*v + c1*r1*(pbest-p) + c2*r2*(gbest-p)
    
    ; Load current velocity: v
    movsd   xmm8, [r13 + rcx*8]
    
    ; w*v
    mulsd   xmm8, xmm3
    
    ; Load current position: p
    movsd   xmm9, [r12 + rcx*8]
    
    ; Load personal best: pbest
    movsd   xmm10, [r14 + rcx*8]
    
    ; (pbest-p)
    movsd   xmm11, xmm10
    subsd   xmm11, xmm9
    
    ; c1*r1*(pbest-p)
    mulsd   xmm11, xmm6
    mulsd   xmm11, xmm4
    
    ; Add to velocity
    addsd   xmm8, xmm11
    
    ; For simplicity, we're assuming gbest is passed at r14+dimensions*8
    ; Load global best: gbest
    movsd   xmm10, [r14 + r10*8 + rcx*8]
    
    ; (gbest-p)
    subsd   xmm10, xmm9
    
    ; c2*r2*(gbest-p)
    mulsd   xmm10, xmm7
    mulsd   xmm10, xmm5
    
    ; Add to velocity
    addsd   xmm8, xmm10
    
    ; Store updated velocity
    movsd   [r13 + rcx*8], xmm8
    
    ; Update position: p = p + v
    addsd   xmm9, xmm8
    movsd   [r12 + rcx*8], xmm9
    
    ; Move to next dimension
    inc     rcx
    cmp     rcx, r10
    jl      .update_loop
    
    ; Cleanup and return
    pop     r14
    pop     r13
    pop     r12
    pop     rbp
    ret
"""

# Compile the assembly code if swarm lib not found
if not _has_swarm_lib:
    with open('swarm_update.asm', 'w') as f:
        f.write(ASSEMBLY_SWARM_UPDATE)
    os.system('nasm -f elf64 swarm_update.asm -o swarm_update.o')
    os.system('gcc -shared -o swarm_update.so swarm_update.o')
    try:
        swarm_module = ctypes.CDLL('./swarm_update.so')
        _has_swarm_update = True
    except:
        _has_swarm_update = False


# Cellular Automaton for Chaotic Pattern Generation
class CellularChaosGenerator:
    def __init__(self, size=100, rule=110):
        self.size = size
        self.rule = rule
        self.grid = np.zeros((size, size), dtype=np.int8)
        self.initialize_grid()
        
    def initialize_grid(self):
        """Initialize with a single cell or random pattern"""
        # Middle cell initialization
        self.grid[0, self.size // 2] = 1
        
    def rule_to_transitions(self):
        """Convert rule number to transition dictionary"""
        transitions = {}
        rule_binary = format(self.rule, '08b')
        patterns = ['111', '110', '101', '100', '011', '010', '001', '000']
        for i, pattern in enumerate(patterns):
            transitions[pattern] = int(rule_binary[i])
        return transitions
        
    def evolve(self, steps=1):
        """Evolve the cellular automaton for specified steps"""
        transitions = self.rule_to_transitions()
        
        for _ in range(steps):
            new_grid = np.zeros_like(self.grid)
            
            for i in range(self.size):
                # Get the pattern for each cell including wrapping
                left = np.roll(self.grid[i], 1)
                right = np.roll(self.grid[i], -1)
                
                # Combine to get neighborhood patterns
                patterns = np.vstack((left, self.grid[i], right)).T
                
                # Apply rules
                for j in range(self.size):
                    pattern = ''.join(map(str, patterns[j]))
                    new_grid[i, j] = transitions.get(pattern, 0)
            
            self.grid = new_grid
            
        return self.grid
    
    def get_chaos_features(self):
        """Extract features from the chaos pattern for swarm guidance"""
        # Calculate entropy along rows and columns
        entropy_x = np.zeros(self.size)
        entropy_y = np.zeros(self.size)
        
        for i in range(self.size):
            # Calculate row and column distributions
            row_vals, row_counts = np.unique(self.grid[i], return_counts=True)
            col_vals, col_counts = np.unique(self.grid[:, i], return_counts=True)
            
            # Calculate entropy (if non-zero distributions)
            if len(row_counts) > 1:
                row_probs = row_counts / np.sum(row_counts)
                entropy_x[i] = -np.sum(row_probs * np.log2(row_probs))
            
            if len(col_counts) > 1:
                col_probs = col_counts / np.sum(col_counts)
                entropy_y[i] = -np.sum(col_probs * np.log2(col_probs))
        
        # Detect structures using convolution
        edge_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        edge_response = convolve2d(self.grid, edge_kernel, mode='same', boundary='wrap')
        
        # Return features as probability distributions
        entropy_x = entropy_x / np.sum(entropy_x) if np.sum(entropy_x) > 0 else np.ones(self.size) / self.size
        entropy_y = entropy_y / np.sum(entropy_y) if np.sum(entropy_y) > 0 else np.ones(self.size) / self.size
        edge_features = np.abs(edge_response.flatten())
        edge_features = edge_features / np.sum(edge_features) if np.sum(edge_features) > 0 else np.ones(self.size*self.size) / (self.size*self.size)
        
        return entropy_x, entropy_y, edge_features


# Evolutionary Agent for Swarm Optimization
class EvolutionaryAgent:
    def __init__(self, dimensions, bounds=None, mutation_rate=0.1):
        self.dimensions = dimensions
        self.bounds = bounds if bounds else [(-10, 10) for _ in range(dimensions)]
        self.mutation_rate = mutation_rate
        
        # Initialize position and velocity
        self.position = np.array([np.random.uniform(low, high) for low, high in self.bounds])
        self.velocity = np.random.uniform(-1, 1, dimensions)
        
        # Best known position and fitness
        self.best_position = self.position.copy()
        self.best_fitness = float('-inf')
        
        # Genetic properties for evolution
        self.dna = np.random.uniform(0, 1, dimensions * 2)  # Genes influence behavior
        
        # Adaptation parameters
        self.cognitive = 1.5 + 0.5 * self.dna[0]  # Personal best influence
        self.social = 1.5 + 0.5 * self.dna[1]     # Global best influence
        self.inertia = 0.5 + 0.4 * self.dna[2]    # Velocity retention
        self.exploration = 0.1 + 0.9 * self.dna[3]  # Randomness in movement
        
    def update(self, global_best, chaos_field=None):
        """Update agent position using swarm intelligence and chaos influence"""
        # Standard PSO update with evolutionary parameters
        r1, r2 = np.random.random(2)
        cognitive_velocity = self.cognitive * r1 * (self.best_position - self.position)
        social_velocity = self.social * r2 * (global_best - self.position)
        
        # Apply chaos field influence if provided
        chaos_velocity = np.zeros(self.dimensions)
        if chaos_field is not None:
            # Sample from chaos field distributions
            chaos_sample = np.random.choice(len(chaos_field), self.dimensions, p=chaos_field)
            chaos_direction = chaos_sample / (len(chaos_field) - 1) * 2 - 1  # -1 to 1
            chaos_velocity = self.exploration * chaos_direction
        
        # Update velocity with inertia
        self.velocity = (self.inertia * self.velocity + 
                         cognitive_velocity + 
                         social_velocity + 
                         chaos_velocity)
        
        # Apply velocity constraints
        max_velocity = 0.1 * np.array([high - low for low, high in self.bounds])
        self.velocity = np.clip(self.velocity, -max_velocity, max_velocity)
        
        # Update position
        self.position += self.velocity
        
        # Apply boundary constraints
        for i in range(self.dimensions):
            if self.position[i] < self.bounds[i][0]:
                self.position[i] = self.bounds[i][0]
                self.velocity[i] *= -0.5  # Bounce with damping
            elif self.position[i] > self.bounds[i][1]:
                self.position[i] = self.bounds[i][1]
                self.velocity[i] *= -0.5  # Bounce with damping
        
        return self.position
    
    def evolve(self, fitness, other_agent):
        """Evolve the agent through genetic operations with another agent"""
        if fitness > self.best_fitness:
            self.best_fitness = fitness
            self.best_position = self.position.copy()
            
        # Only evolve if this agent is less fit than the other
        if self.best_fitness < other_agent.best_fitness:
            # Crossover
            crossover_point = np.random.randint(1, len(self.dna) - 1)
            new_dna = np.concatenate([
                self.dna[:crossover_point],
                other_agent.dna[crossover_point:]
            ])
            
            # Mutation
            mutation_mask = np.random.random(len(new_dna)) < self.mutation_rate
            mutation_values = np.random.uniform(-0.2, 0.2, len(new_dna))
            new_dna[mutation_mask] += mutation_values[mutation_mask]
            new_dna = np.clip(new_dna, 0, 1)
            
            # Update DNA and derived parameters
            self.dna = new_dna
            self.cognitive = 1.5 + 0.5 * self.dna[0]
            self.social = 1.5 + 0.5 * self.dna[1]
            self.inertia = 0.5 + 0.4 * self.dna[2]
            self.exploration = 0.1 + 0.9 * self.dna[3]
            
            # Slight movement toward better agent's position
            self.position += 0.1 * (other_agent.best_position - self.position)


# Swarm optimization system that leverages chaos
class ChaosSwarmOptimizer:
    def __init__(self, fitness_func, dimensions, swarm_size=50, 
                 bounds=None, use_chaos=True, chaos_size=100, chaos_rule=110):
        self.fitness_func = fitness_func
        self.dimensions = dimensions
        self.swarm_size = swarm_size
        self.bounds = bounds if bounds else [(-10, 10) for _ in range(dimensions)]
        
        # Initialize swarm
        self.agents = [EvolutionaryAgent(dimensions, bounds) for _ in range(swarm_size)]
        
        # Global best
        self.global_best_position = None
        self.global_best_fitness = float('-inf')
        
        # Chaos field generator
        self.use_chaos = use_chaos
        if use_chaos:
            self.chaos_generator = CellularChaosGenerator(size=chaos_size, rule=chaos_rule)
            self.chaos_field = None
            self.update_chaos_field()
        
        # Statistics and properties
        self.convergence_history = []
        self.diversity_history = []
        self.iteration = 0
    
    def update_chaos_field(self):
        """Update the chaos field by evolving cellular automata"""
        if self.use_chaos:
            # Evolve the cellular automaton
            self.chaos_generator.evolve(steps=5)
            
            # Extract probability distributions for swarm guidance
            entropy_x, entropy_y, edge_features = self.chaos_generator.get_chaos_features()
            
            # Store as flattened probability distribution
            self.chaos_field = edge_features
    
    def evaluate_fitness(self, positions):
        """Evaluate fitness for multiple positions in parallel"""
        # Use multiprocessing for fitness evaluation
        with Pool(min(cpu_count(), len(positions))) as pool:
            return pool.map(self.fitness_func, positions)
    
    def optimize(self, iterations=100):
        """Run the optimization process"""
        for iteration in range(iterations):
            self.iteration = iteration
            
            # Update chaos field
            if self.use_chaos and iteration % 5 == 0:
                self.update_chaos_field()
            
            # Get all agent positions
            positions = [agent.position for agent in self.agents]
            
            # Evaluate fitness in parallel
            fitness_values = self.evaluate_fitness(positions)
            
            # Update agent personal bests and global best
            for i, (position, fitness) in enumerate(zip(positions, fitness_values)):
                agent = self.agents[i]
                
                if fitness > agent.best_fitness:
                    agent.best_fitness = fitness
                    agent.best_position = position.copy()
                
                if fitness > self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_position = position.copy()
            
            # Update agents' positions
            for agent in self.agents:
                agent.update(self.global_best_position, self.chaos_field)
            
            # Evolutionary step - arrange agents in random pairs and evolve
            indices = list(range(self.swarm_size))
            np.random.shuffle(indices)
            
            for i in range(0, self.swarm_size - 1, 2):
                idx1, idx2 = indices[i], indices[i+1]
                agent1, agent2 = self.agents[idx1], self.agents[idx2]
                
                # Cross-evolve
                agent1.evolve(fitness_values[idx1], agent2)
                agent2.evolve(fitness_values[idx2], agent1)
            
            # Calculate diversity
            positions = np.array([agent.position for agent in self.agents])
            diversity = np.mean(np.std(positions, axis=0))
            
            # Record history
            self.convergence_history.append(self.global_best_fitness)
            self.diversity_history.append(diversity)
            
            # Dynamic parameter adjustment based on diversity
            if iteration > 10 and diversity < 0.01:
                # Inject chaos to escape local optima
                for i in range(self.swarm_size // 10):  # Reset 10% of agents
                    idx = np.random.randint(0, self.swarm_size)
                    self.agents[idx] = EvolutionaryAgent(self.dimensions, self.bounds)
            
            # Progress report every 10 iterations
            if iteration % 10 == 0:
                print(f"Iteration {iteration}: Best fitness = {self.global_best_fitness:.4f}, Diversity = {diversity:.4f}")
        
        return self.global_best_position, self.global_best_fitness
    
    def visualize_optimization(self):
        """Visualize the optimization process"""
        plt.figure(figsize=(15, 10))
        
        # Plot convergence
        plt.subplot(2, 2, 1)
        plt.plot(self.convergence_history)
        plt.title('Convergence History')
        plt.xlabel('Iteration')
        plt.ylabel('Best Fitness')
        
        # Plot diversity
        plt.subplot(2, 2, 2)
        plt.plot(self.diversity_history)
        plt.title('Swarm Diversity')
        plt.xlabel('Iteration')
        plt.ylabel('Diversity')
        
        # Plot final agent positions (first 2 dimensions)
        if self.dimensions >= 2:
            plt.subplot(2, 2, 3)
            positions = np.array([agent.position for agent in self.agents])
            plt.scatter(positions[:, 0], positions[:, 1], alpha=0.6)
            plt.scatter([self.global_best_position[0]], [self.global_best_position[1]], 
                       color='red', s=100, marker='*')
            plt.title('Agent Positions (dims 0-1)')
            plt.xlabel('Dimension 0')
            plt.ylabel('Dimension 1')
        
        # Plot chaos field if used
        if self.use_chaos:
            plt.subplot(2, 2, 4)
            plt.imshow(self.chaos_generator.grid, cmap='binary')
            plt.title('Chaos Field')
            plt.colorbar()
        
        plt.tight_layout()
        plt.show()
    
    def visualize_search_space(self, resolution=50):
        """Visualize the fitness landscape for 2D problems"""
        if self.dimensions != 2:
            print("Visualization only available for 2D problems")
            return
        
        # Create grid for visualization
        x = np.linspace(self.bounds[0][0], self.bounds[0][1], resolution)
        y = np.linspace(self.bounds[1][0], self.bounds[1][1], resolution)
        X, Y = np.meshgrid(x, y)
        
        # Evaluate fitness across the grid
        Z = np.zeros((resolution, resolution))
        for i in range(resolution):
            positions = []
            for j in range(resolution):
                positions.append(np.array([X[i, j], Y[i, j]]))
            
            Z[i, :] = self.evaluate_fitness(positions)
        
        # Plot the fitness landscape
        plt.figure(figsize=(12, 10))
        
        # Contour plot
        plt.contourf(X, Y, Z, 50, cmap='viridis', alpha=0.5)
        plt.colorbar(label='Fitness')
        
        # Plot agent positions
        positions = np.array([agent.position for agent in self.agents])
        plt.scatter(positions[:, 0], positions[:, 1], color='white', alpha=0.6, label='Agents')
        
        # Plot global best
        plt.scatter([self.global_best_position[0]], [self.global_best_position[1]], 
                   color='red', s=200, marker='*', label='Global Best')
        
        plt.title('Fitness Landscape and Agent Positions')
        plt.xlabel('Dimension 0')
        plt.ylabel('Dimension 1')
        plt.legend()
        plt.tight_layout()
        plt.show()


# Graph integration layer to operate with Quantum Neural Pathfinder
class SwarmGraphOptimizer:
    def __init__(self, graph, dimensions=3, swarm_size=100, use_chaos=True):
        self.graph = graph
        self.dimensions = dimensions
        self.swarm_size = swarm_size
        self.use_chaos = use_chaos
        
        # Extract graph properties for optimization
        self.node_positions = {}
        self.edge_weights = {}
        self.node_properties = {}
        
        # Initialize from graph
        self._extract_graph_properties()
        
        # Setup the optimizer
        self.optimizer = ChaosSwarmOptimizer(
            fitness_func=self._graph_fitness_function,
            dimensions=dimensions * len(self.graph.nodes),
            swarm_size=swarm_size,
            bounds=[(-5, 5) for _ in range(dimensions * len(self.graph.nodes))],
            use_chaos=use_chaos
        )
    
    def _extract_graph_properties(self):
        """Extract properties from the graph for optimization"""
        for node in self.graph.nodes:
            # Get or generate node positions
            if 'position' in self.graph.nodes[node]:
                self.node_positions[node] = self.graph.nodes[node]['position']
            else:
                self.node_positions[node] = np.random.normal(0, 1, self.dimensions)
            
            # Extract other node properties
            props = {}
            for key, value in self.graph.nodes[node].items():
                if key != 'position' and isinstance(value, (int, float)):
                    props[key] = value
            self.node_properties[node] = props
        
        # Extract edge weights
        for u, v in self.graph.edges:
            self.edge_weights[(u, v)] = self.graph.edges[u, v].get('weight', 1.0)
    
    def _graph_fitness_function(self, position_vector):
        """Fitness function for graph optimization"""
        # Reshape vector to node positions
        node_positions = {}
        n_nodes = len(self.graph.nodes)
        position_matrix = position_vector.reshape(n_nodes, self.dimensions)
        
        for i, node in enumerate(self.graph.nodes):
            node_positions[node] = position_matrix[i]
        
        # Calculate fitness based on several factors
        
        # 1. Edge length optimization - shorter is better
        edge_length_factor = 0
        for u, v in self.graph.edges:
            distance = np.linalg.norm(node_positions[u] - node_positions[v])
            weight = self.edge_weights.get((u, v), 1.0)
            edge_length_factor += weight * distance
        
        # 2. Node distribution - more evenly distributed is better
        distribution_factor = 0
        for i, u in enumerate(self.graph.nodes):
            for v in self.graph.nodes:
                if u != v:
                    distance = np.linalg.norm(node_positions[u] - node_positions[v])
                    distribution_factor += 1 / (distance + 0.1)  # Avoid division by zero
        
        # 3. Property-based relationships - nodes with similar properties should be closer
        property_factor = 0
        for u in self.graph.nodes:
            for v in self.graph.nodes:
                if u != v:
                    # Calculate property similarity
                    similarity = 0
                    count = 0
                    for key in set(self.node_properties[u].keys()) & set(self.node_properties[v].keys()):
                        diff = abs(self.node_properties[u][key] - self.node_properties[v][key])
                        max_val = max(abs(self.node_properties[u][key]), abs(self.node_properties[v][key]))
                        if max_val > 0:
                            similarity += 1 - (diff / max_val)
                            count += 1
                    
                    if count > 0:
                        similarity /= count
                        distance = np.linalg.norm(node_positions[u] - node_positions[v])
                        # Similar nodes should be closer
                        property_factor += similarity / (distance + 0.1)
        
        # Combine factors with appropriate weights
        fitness = -edge_length_factor - 0.1 * distribution_factor + 0.5 * property_factor
        
        return fitness
    
    def optimize_graph_layout(self, iterations=100):
        """Optimize the graph layout using swarm intelligence"""
        # Run the optimizer
        best_position, best_fitness = self.optimizer.optimize(iterations=iterations)
        
        # Update graph with optimized positions
        n_nodes = len(self.graph.nodes)
        position_matrix = best_position.reshape(n_nodes, self.dimensions)
        
        for i, node in enumerate(self.graph.nodes):
            self.graph.nodes[node]['position'] = position_matrix[i]
        
        return best_fitness
    
    def find_optimal_paths(self, source, target, n_paths=5):
        """Find multiple diverse optimal paths using swarm intelligence"""
        # Define a path fitness function
        def path_fitness(path_indices):
            # Convert continuous values to node indices
            n_nodes = len(self.graph.nodes)
            nodes = list(self.graph.nodes)
            
            # Extract path nodes
            path_length = min(len(path_indices) // 2, 20)  # Limit path length
            path = [source]
            
            for i in range(path_length):
                idx = int(path_indices[i] * n_nodes) % n_nodes
                node = nodes[idx]
                
                # Skip if node already in path or not connected
                if node in path or not self.graph.has_edge(path[-1], node):
                    continue
                
                path.append(node)
                
                # Stop if target reached
                if node == target:
                    break
            
            # If target not reached, penalize
            if path[-1] != target:
                path.append(target)
                return -1000  # Large penalty
            
            # Calculate path cost
            cost = 0
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                if self.graph.has_edge(u, v):
                    cost += self.graph.edges[u, v].get('weight', 1.0)
                else:
                    return -1000  # Invalid path
            
            # Calculate path diversity (for multiple paths)
            diversity_bonus = len(set(path))  # Favor paths with more unique nodes
            
            # Path fitness is negative cost plus diversity bonus
            return -cost + 0.1 * diversity_bonus
        
        # Setup a swarm optimizer for path finding
        path_optimizer = ChaosSwarmOptimizer(
            fitness_func=path_fitness,
            dimensions=40,  # Allow for paths up to 20 nodes
            swarm_size=self.swarm_size,
            bounds=[(0, 1) for _ in range(40)],
            use_chaos=self.use_chaos
        )
        
        # Run the optimizer
        best_position, _ = path_optimizer.optimize(iterations=50)
        
        # Extract the path from the best position
        n_nodes = len(self.graph.nodes)
        nodes = list(self.graph.nodes)
        
        path = [source]
        path_length = min(len(best_position) // 2, 20)
        
        for i in range(path_length):
            idx = int(best_position[i] * n_nodes) % n_nodes
            node = nodes[idx]
            
            if node not in path and self.graph.has_edge(path[-1], node):
                path.append(node)
                
                if node == target:
                    break
        
        if path[-1] != target:
            path.append(target)
        
        return path
    
    def optimize_network_flows(self, demand_pairs, iterations=50):
        """Optimize the entire network for multiple source-destination pairs"""
        # Define a fitness function for network flow optimization
        def network_flow_fitness(weight_factors):
            # Apply weight factors to the original edge weights
            modified_weights = {}
            for i, (u, v) in enumerate(self.graph.edges):
                idx = i % len(weight_factors)
                factor = 0.5 + weight_factors[idx]  # 0.5 to 1.5 range
                modified_weights[(u, v)] = self.edge_weights.get((u, v), 1.0) * factor
            
            # Calculate overall network performance
            total_cost = 0
            path_count = 0
            
            for source, target in demand_pairs:
                try