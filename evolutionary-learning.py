from dataclasses import dataclass, field
from typing import Dict, List, Set, Any, Optional
import numpy as np
from datetime import datetime
import networkx as nx
from copy import deepcopy
import random

@dataclass
class Evolution:
    """Tracks evolutionary changes in the network."""
    generation: int = 0
    mutations: List[Dict] = field(default_factory=list)
    fitness_history: List[float] = field(default_factory=list)
    adaptation_rate: float = 0.1

@dataclass
class Genome:
    """Represents the genetic configuration of a node."""
    traits: Dict[str, float]
    learning_rate: float
    plasticity: float
    mutation_rate: float
    parent_id: Optional[str] = None
    
    def mutate(self) -> 'Genome':
        """Creates a mutated copy of the genome."""
        mutated = deepcopy(self)
        
        # Mutate traits
        for trait in mutated.traits:
            if random.random() < self.mutation_rate:
                mutated.traits[trait] *= np.random.normal(1, 0.1)
                mutated.traits[trait] = np.clip(mutated.traits[trait], 0, 1)
        
        # Mutate learning parameters
        if random.random() < self.mutation_rate:
            mutated.learning_rate *= np.random.normal(1, 0.1)
            mutated.learning_rate = np.clip(mutated.learning_rate, 0.01, 1.0)
            
        if random.random() < self.mutation_rate:
            mutated.plasticity *= np.random.normal(1, 0.1)
            mutated.plasticity = np.clip(mutated.plasticity, 0.1, 1.0)
            
        return mutated

class EvolutionaryLearning:
    """Manages the evolutionary learning process of the network."""
    
    def __init__(self, population_size: int = 100):
        self.population_size = population_size
        self.evolution = Evolution()
        self.population = []
        self.fitness_cache = {}
        self.best_genome = None
        self.generation_metrics = []
        
    def initialize_population(self):
        """Initializes the population with random genomes."""
        self.population = []
        for _ in range(self.population_size):
            genome = Genome(
                traits={
                    'pattern_recognition': random.random(),
                    'knowledge_integration': random.random(),
                    'adaptation_speed': random.random(),
                    'memory_retention': random.random(),
                    'generalization': random.random()
                },
                learning_rate=random.uniform(0.01, 0.1),
                plasticity=random.uniform(0.1, 0.9),
                mutation_rate=random.uniform(0.01, 0.1)
            )
            self.population.append(genome)
    
    def evolve(self, network: nx.DiGraph, generations: int = 10):
        """Evolves the population over multiple generations."""
        metrics = []
        
        for generation in range(generations):
            # Evaluate fitness
            fitness_scores = self._evaluate_population(network)
            
            # Record metrics
            gen_metrics = self._record_generation_metrics(generation, fitness_scores)
            metrics.append(gen_metrics)
            
            # Select parents
            parents = self._select_parents(fitness_scores)
            
            # Create new population
            new_population = []
            while len(new_population) < self.population_size:
                parent1, parent2 = random.sample(parents, 2)
                child = self._crossover(parent1, parent2)
                child = child.mutate()
                new_population.append(child)
            
            self.population = new_population
            self.evolution.generation += 1
            
            # Update network based on best genome
            self._update_network(network)
            
        return metrics
    
    def _evaluate_population(self, network: nx.DiGraph) -> List[float]:
        """Evaluates the fitness of each genome in the population."""
        fitness_scores = []
        
        for genome in self.population:
            if str(genome.traits) in self.fitness_cache:
                fitness = self.fitness_cache[str(genome.traits)]
            else:
                fitness = self._calculate_fitness(genome, network)
                self.fitness_cache[str(genome.traits)] = fitness
            
            fitness_scores.append(fitness)
            
        return fitness_scores
    
    def _calculate_fitness(self, genome: Genome, network: nx.DiGraph) -> float:
        """Calculates fitness score for a genome."""
        # Pattern recognition score
        pattern_score = self._evaluate_pattern_recognition(genome, network)
        
        # Knowledge integration score
        integration_score = self._evaluate_knowledge_integration(genome, network)
        
        # Adaptation score
        adaptation_score = self._evaluate_adaptation(genome, network)
        
        # Combine scores with weights from genome traits
        fitness = (
            genome.traits['pattern_recognition'] * pattern_score +
            genome.traits['knowledge_integration'] * integration_score +
            genome.traits['adaptation_speed'] * adaptation_score
        ) / 3.0
        
        return fitness
    
    def _evaluate_pattern_recognition(self, genome: Genome, network: nx.DiGraph) -> float:
        """Evaluates pattern recognition capability."""
        # Calculate clustering coefficient as a measure of pattern recognition
        clustering = nx.average_clustering(network)
        
        # Consider network modularity
        modularity = self._calculate_modularity(network)
        
        # Combine metrics with genome traits
        score = (clustering + modularity) / 2
        return score * genome.traits['pattern_recognition']
    
    def _evaluate_knowledge_integration(self, genome: Genome, network: nx.DiGraph) -> float:
        """Evaluates knowledge integration capability."""
        # Calculate network density
        density = nx.density(network)
        
        # Calculate average path length
        try:
            avg_path_length = nx.average_shortest_path_length(network)
            path_length_score = 1.0 / (1.0 + avg_path_length)
        except nx.NetworkXError:
            path_length_score = 0.0
        
        # Combine metrics with genome traits
        score = (density + path_length_score) / 2
        return score * genome.traits['knowledge_integration']
    
    def _evaluate_adaptation(self, genome: Genome, network: nx.DiGraph) -> float:
        """Evaluates adaptation capability."""
        # Consider learning rate and plasticity
        adaptation_score = (genome.learning_rate + genome.plasticity) / 2
        
        # Consider network growth rate
        growth_rate = len(network.nodes()) / max(1, self.evolution.generation)
        growth_score = 1.0 / (1.0 + np.exp(-growth_rate))  # Sigmoid normalization
        
        # Combine metrics with genome traits
        score = (adaptation_score + growth_score) / 2
        return score * genome.traits['adaptation_speed']
    
    def _calculate_modularity(self, network: nx.DiGraph) -> float:
        """Calculates network modularity."""
        if not network.nodes():
            return 0.0
            
        # Convert to undirected for community detection
        undirected = network.to_undirected()
        
        # Find communities
        communities = nx.community.greedy_modularity_communities(undirected)
        
        # Calculate modularity score
        modularity = nx.community.modularity(undirected, communities)
        return modularity
    
    def _select_parents(self, fitness_scores: List[float]) -> List[Genome]:
        """Selects parent genomes for next generation using tournament selection."""
        tournament_size = 5
        num_parents = self.population_size // 2
        parents = []
        
        for _ in range(num_parents):
            # Select tournament participants
            tournament_indices = random.sample(range(len(self.population)), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            
            # Select winner
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            parents.append(self.population[winner_idx])
        
        return parents
    
    def _crossover(self, parent1: Genome, parent2: Genome) -> Genome:
        """Performs crossover between two parent genomes."""
        child_traits = {}
        
        # Trait crossover
        for trait in parent1.traits:
            if random.random() < 0.5:
                child_traits[trait] = parent1.traits[trait]
            else:
                child_traits[trait] = parent2.traits[trait]
        
        # Learning parameter crossover
        child = Genome(
            traits=child_traits,
            learning_rate=(parent1.learning_rate + parent2.learning_rate) / 2,
            plasticity=(parent1.plasticity + parent2.plasticity) / 2,
            mutation_rate=(parent1.mutation_rate + parent2.mutation_rate) / 2
        )
        
        return child
    
    def _update_network(self, network: nx.DiGraph):
        """Updates network based on best genome."""
        # Find best genome
        fitness_scores = self._evaluate_population(network)
        best_idx = np.argmax(fitness_scores)
        best_genome = self.population[best_idx]
        
        # Update network parameters based on best genome
        for node in network.nodes():
            network.nodes[node]['learning_rate'] = best_genome.learning_rate
            network.nodes[node]['plasticity'] = best_genome.plasticity
            
        # Update edge weights based on genome traits
        for edge in network.edges():
            weight = network.edges[edge]['weight']
            new_weight = weight * (1 + best_genome.traits['adaptation_speed'])
            network.edges[edge]['weight'] = np.clip(new_weight, 0, 1)
    
    def _record_generation_metrics(self, generation: int, fitness_scores: List[float]) -> Dict[str, Any]:
        """Records metrics for current generation."""
        metrics = {
            'generation': generation,
            'max_fitness': max(fitness_scores),
            'avg_fitness': np.mean(fitness_scores),
            'min_fitness': min(fitness_scores),
            'fitness_std': np.std(fitness_scores),
            'best_genome_traits': self.population[np.argmax(fitness_scores)].traits,
            'timestamp': datetime.now().isoformat()
        }
        
        self.generation_metrics.append(metrics)
        return metrics
    
    def get_evolution_summary(self) -> Dict[str, Any]:
        """Returns a summary of the evolutionary process."""
        if not self.generation_metrics:
            return {}
            
        return {
            'total_generations': len(self.generation_metrics),
            'final_max_fitness': self.generation_metrics[-1]['max_fitness'],
            'fitness_improvement': (
                self.generation_metrics[-1]['max_fitness'] -
                self.generation_metrics[0]['max_fitness']
            ),
            'best_genome_traits': self.generation_metrics[-1]['best_genome_traits'],
            'evolution_time': (
                datetime.fromisoformat(self.generation_metrics[-1]['timestamp']) -
                datetime.fromisoformat(self.generation_metrics[0]['timestamp'])
            ).total_seconds()
        }
