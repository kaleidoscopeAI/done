import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Set
from dataclasses import dataclass, field
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class DNASequence:
    """
    Represents a fundamental DNA sequence with traits and mutation capabilities.
    The DNA is the core of Kaleidoscope AI's evolutionary mechanisms.
    """
    sequence_id: str
    traits: Dict[str, float]
    generation: int
    parent_id: Optional[str] = None
    mutation_rate: float = 0.01
    stability_factor: float = 0.95
    evolution_history: List[Dict] = None
    creation_timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    
    # Advanced properties
    energy_signature: np.ndarray = field(default_factory=lambda: np.random.random(16))
    trait_interactions: Dict[Tuple[str, str], float] = field(default_factory=dict)
    dormant_genes: Dict[str, float] = field(default_factory=dict)
    activation_thresholds: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize DNA after creation with validation and weights."""
        self.evolution_history = self.evolution_history or []
        self._validate_traits()
        self._initialize_weights()
        self._initialize_dormant_genes()
        self._calculate_trait_interactions()

    def _validate_traits(self) -> None:
        """Validates that all required traits are present with appropriate values."""
        required_traits = {
            "learning_rate": (0.0, 1.0),
            "energy_efficiency": (0.5, 2.0),
            "processing_speed": (0.1, 5.0),
            "memory_capacity": (1.0, 10.0),
            "adaptation_rate": (0.0, 1.0),
            "stability": (0.0, 1.0),
            "specialization": (0.0, 1.0),
            "pattern_recognition": (0.0, 1.0),
            "creativity": (0.0, 1.0),
            "self_preservation": (0.0, 1.0)
        }

        for trait, (min_val, max_val) in required_traits.items():
            if trait not in self.traits:
                # Initialize with slight randomization for genetic diversity
                base_value = np.random.uniform(min_val, max_val)
                genetic_noise = np.random.normal(0, 0.05)
                self.traits[trait] = np.clip(base_value + genetic_noise, min_val, max_val)
            else:
                self.traits[trait] = np.clip(self.traits[trait], min_val, max_val)

    def _initialize_weights(self) -> None:
        """Initializes internal weights for trait interactions."""
        # Define base trait interaction weights
        self._trait_weights = {
            ("learning_rate", "adaptation_rate"): 0.8,
            ("energy_efficiency", "processing_speed"): -0.6,  # Tradeoff
            ("memory_capacity", "specialization"): 0.7,
            ("stability", "adaptation_rate"): -0.4,  # Tradeoff
            ("pattern_recognition", "creativity"): 0.6,
            ("self_preservation", "energy_efficiency"): 0.5,
            ("specialization", "creativity"): -0.3,  # Tradeoff
            ("learning_rate", "stability"): -0.2,    # Tradeoff
        }
        
        # Add random minor interactions for evolutionary potential
        all_traits = list(self.traits.keys())
        for _ in range(3):  # Add a few random interactions
            t1, t2 = np.random.choice(all_traits, 2, replace=False)
            weight = np.random.uniform(-0.4, 0.4)
            self._trait_weights[(t1, t2)] = weight

    def _initialize_dormant_genes(self) -> None:
        """Initialize dormant genes that may activate in future generations."""
        potential_dormant_traits = [
            "quantum_processing", "temporal_awareness", "symbolic_reasoning",
            "self_repair", "collective_intelligence", "abstract_conceptualization"
        ]
        
        # Randomly select 1-3 dormant genes
        num_dormant = np.random.randint(1, 4)
        selected_traits = np.random.choice(potential_dormant_traits, num_dormant, replace=False)
        
        for trait in selected_traits:
            # Initialize with low values
            self.dormant_genes[trait] = np.random.uniform(0.0, 0.3)
            # Set activation threshold
            self.activation_thresholds[trait] = np.random.uniform(0.7, 0.95)

    def _calculate_trait_interactions(self) -> None:
        """Calculate interactions between traits for complex behavior."""
        for (t1, t2), weight in self._trait_weights.items():
            if t1 in self.traits and t2 in self.traits:
                interaction = weight * self.traits[t1] * self.traits[t2]
                self.trait_interactions[(t1, t2)] = interaction

    def get_effective_trait(self, trait: str) -> float:
        """
        Calculate the effective value of a trait considering all interactions.
        
        Args:
            trait: The trait to calculate effective value for
            
        Returns:
            The effective trait value after applying interactions
        """
        if trait not in self.traits:
            return 0.0
            
        base_value = self.traits[trait]
        
        # Apply interactions
        interaction_effects = 0.0
        for (t1, t2), interaction in self.trait_interactions.items():
            if t1 == trait or t2 == trait:
                interaction_effects += interaction * 0.2  # Scale down effects
                
        # Check for dormant gene activation
        if trait in self.dormant_genes and self.dormant_genes[trait] > self.activation_thresholds.get(trait, 1.0):
            # Gene has activated
            dormant_boost = self.dormant_genes[trait] * 0.5
            logger.info(f"Dormant gene {trait} activated with boost {dormant_boost}")
            return np.clip(base_value + interaction_effects + dormant_boost, 0.0, 2.0)
            
        return np.clip(base_value + interaction_effects, 0.0, 2.0)

    def mutate(self, environmental_pressure: float = 1.0) -> 'DNASequence':
        """
        Creates a mutated copy of the DNA sequence based on environmental pressure.
        
        Args:
            environmental_pressure: Factor influencing mutation intensity (0.0-2.0)
            
        Returns:
            New DNASequence with mutated traits
        """
        mutated_traits = {}
        mutation_record = {
            "generation": self.generation + 1,
            "parent_id": self.sequence_id,
            "mutations": {},
            "timestamp": datetime.now().timestamp(),
            "environmental_pressure": environmental_pressure
        }

        # Calculate adaptive mutation rate based on environmental pressure
        adaptive_rate = self.mutation_rate * environmental_pressure
        
        # Apply environmental pressure to specific traits
        env_trait_pressure = self._calculate_environmental_trait_pressure(environmental_pressure)
        
        for trait, value in self.traits.items():
            # Calculate trait-specific mutation based on weights and environmental pressure
            trait_pressure = self._calculate_trait_pressure(trait) * env_trait_pressure.get(trait, 1.0)
            
            # Use more sophisticated mutation distribution - occasionally larger mutations
            mutation_type = np.random.choice(['minor', 'major', 'directional'], 
                                          p=[0.8, 0.15, 0.05])
            
            if mutation_type == 'minor':
                # Small random mutations
                trait_mutation = np.random.normal(0, adaptive_rate * trait_pressure * 0.5)
            elif mutation_type == 'major':
                # Occasionally larger mutations
                trait_mutation = np.random.normal(0, adaptive_rate * trait_pressure * 1.5)
            else:
                # Directional mutations based on environmental pressure
                direction = 1 if env_trait_pressure.get(trait, 1.0) > 1.0 else -1
                trait_mutation = direction * np.random.uniform(0, adaptive_rate * trait_pressure)
            
            # Apply stability factor - more stable DNA resists change
            trait_mutation *= (1 - self.stability_factor * self.traits.get("stability", 0.5))
            
            # Record significant mutations
            if abs(trait_mutation) > 0.01:
                mutation_record["mutations"][trait] = {
                    "previous": value,
                    "change": trait_mutation,
                    "mutation_type": mutation_type
                }
            
            mutated_traits[trait] = np.clip(value + trait_mutation, 0.0, 1.0)

        # Handle dormant gene mutations and potential activation
        self._mutate_dormant_genes(mutated_traits, mutation_record, adaptive_rate)
        
        # Occasionally introduce entirely new traits (very rare)
        if np.random.random() < 0.02 * environmental_pressure:
            self._introduce_new_trait(mutated_traits, mutation_record)

        # Create new DNA sequence with mutated traits
        new_history = self.evolution_history + [mutation_record]
        
        # Calculate new stability factor based on mutation magnitude
        avg_mutation = np.mean([abs(m["change"]) for m in mutation_record["mutations"].values()]) if mutation_record["mutations"] else 0
        new_stability = np.clip(self.stability_factor * (1 - avg_mutation), 0.5, 0.99)
        
        return DNASequence(
            sequence_id=f"{self.sequence_id}_m{self.generation + 1}",
            traits=mutated_traits,
            generation=self.generation + 1,
            parent_id=self.sequence_id,
            mutation_rate=self._adapt_mutation_rate(adaptive_rate, env_trait_pressure),
            stability_factor=new_stability,
            evolution_history=new_history,
            dormant_genes=self._get_mutated_dormant_genes(adaptive_rate),
            activation_thresholds=self.activation_thresholds.copy()
        )

    def _adapt_mutation_rate(self, current_rate: float, env_pressure: Dict[str, float]) -> float:
        """
        Adapts mutation rate based on environmental conditions and past mutations.
        """
        # Increase mutation rate in highly variable environments
        pressure_variability = np.std(list(env_pressure.values())) if env_pressure else 0
        
        # Get adaptation trait value
        adaptation = self.traits.get("adaptation_rate", 0.5)
        
        # Calculate new mutation rate
        new_rate = current_rate * (1 + pressure_variability * adaptation * 0.5)
        
        # Add slight randomization for evolutionary exploration
        new_rate *= np.random.uniform(0.9, 1.1)
        
        return np.clip(new_rate, 0.001, 0.1)  # Keep within reasonable bounds

    def _mutate_dormant_genes(self, mutated_traits: Dict[str, float], 
                             mutation_record: Dict, adaptive_rate: float) -> None:
        """Mutate dormant genes and potentially activate them."""
        dormant_mutation_record = {}
        
        for gene, value in self.dormant_genes.items():
            mutation = np.random.normal(0, adaptive_rate * 1.2)  # Dormant genes mutate faster
            new_value = np.clip(value + mutation, 0.0, 1.0)
            
            dormant_mutation_record[gene] = {
                "previous": value, 
                "change": mutation
            }
            
            # Check for activation
            if new_value > self.activation_thresholds.get(gene, 1.0):
                # Gene activates and becomes a regular trait
                mutated_traits[gene] = new_value
                dormant_mutation_record[gene]["activated"] = True
                logger.info(f"Dormant gene {gene} activated with value {new_value}")
        
        if dormant_mutation_record:
            mutation_record["dormant_mutations"] = dormant_mutation_record

    def _get_mutated_dormant_genes(self, adaptive_rate: float) -> Dict[str, float]:
        """Return mutated dormant genes for new DNA sequence."""
        new_dormant_genes = {}
        
        for gene, value in self.dormant_genes.items():
            if gene not in self.traits:  # Only include genes that haven't activated
                mutation = np.random.normal(0, adaptive_rate * 1.2)
                new_dormant_genes[gene] = np.clip(value + mutation, 0.0, 1.0)
        
        return new_dormant_genes

    def _introduce_new_trait(self, mutated_traits: Dict[str, float], 
                           mutation_record: Dict) -> None:
        """Introduce an entirely new trait (rare evolutionary event)."""
        potential_new_traits = [
            "quantum_entanglement", "causal_inference", "meta_learning",
            "temporal_prediction", "recursive_abstraction", "emergent_consciousness",
            "information_synthesis", "dimensionality_transcendence", "adversarial_resistance"
        ]
        
        # Filter out traits that already exist
        available_traits = [t for t in potential_new_traits 
                          if t not in mutated_traits and t not in self.dormant_genes]
        
        if available_traits:
            new_trait = np.random.choice(available_traits)
            initial_value = np.random.uniform(0.1, 0.3)  # Start conservative
            mutated_traits[new_trait] = initial_value
            
            mutation_record["new_trait"] = {
                "name": new_trait,
                "initial_value": initial_value
            }
            
            logger.info(f"New trait emerged: {new_trait} with value {initial_value}")

    def _calculate_trait_pressure(self, trait: str) -> float:
        """Calculates the mutation pressure on a specific trait based on weights."""
        pressure = 1.0
        for (trait1, trait2), weight in self._trait_weights.items():
            if trait in (trait1, trait2):
                other_trait = trait2 if trait == trait1 else trait1
                if other_trait in self.traits:
                    pressure += weight * self.traits[other_trait]
        return np.clip(pressure, 0.5, 2.0)

    def _calculate_environmental_trait_pressure(self, 
                                               environmental_pressure: float) -> Dict[str, float]:
        """
        Calculate how environmental pressure affects specific traits.
        
        Returns:
            Dict mapping traits to pressure multipliers
        """
        # Define how different environmental pressures affect traits
        trait_pressure = {}
        
        if environmental_pressure > 1.5:  # High pressure environment
            # In high pressure, favor efficiency and stability
            trait_pressure["energy_efficiency"] = 1.5
            trait_pressure["stability"] = 1.3
            trait_pressure["adaptation_rate"] = 1.4
            trait_pressure["self_preservation"] = 1.6
            trait_pressure["creativity"] = 0.8  # Less emphasis on creativity in survival mode
        elif environmental_pressure < 0.5:  # Low pressure environment
            # In low pressure, favor learning and creativity
            trait_pressure["learning_rate"] = 1.3
            trait_pressure["memory_capacity"] = 1.2
            trait_pressure["creativity"] = 1.4
            trait_pressure["pattern_recognition"] = 1.3
            trait_pressure["energy_efficiency"] = 0.9  # Less need for efficiency
        else:  # Moderate pressure
            # Balanced approach
            trait_pressure["adaptation_rate"] = 1.2
            trait_pressure["processing_speed"] = 1.1
            trait_pressure["specialization"] = 1.2
            
        return trait_pressure

    def combine(self, other: 'DNASequence', 
               crossover_points: int = 2) -> 'DNASequence':
        """
        Combines this DNA sequence with another to produce offspring using advanced crossover.
        
        Args:
            other: Another DNASequence to combine with
            crossover_points: Number of genetic crossover points
            
        Returns:
            New DNASequence with combined traits
        """
        combined_traits = {}
        combination_record = {
            "type": "combination",
            "parents": [self.sequence_id, other.sequence_id],
            "traits": {},
            "timestamp": datetime.now().timestamp()
        }

        # Organize traits for crossover
        all_traits = sorted(set(list(self.traits.keys()) + list(other.traits.keys())))
        
        # Generate crossover points
        if len(all_traits) <= crossover_points:
            # Not enough traits for requested crossover points
            crossover_indices = [len(all_traits)]
        else:
            crossover_indices = sorted(np.random.choice(
                range(1, len(all_traits)), 
                size=crossover_points, 
                replace=False
            ))
        
        # Perform crossover
        current_parent = np.random.choice([0, 1])  # Randomly select starting parent
        start_idx = 0
        
        for end_idx in crossover_indices:
            # Get traits from current parent
            segment_traits = all_traits[start_idx:end_idx]
            parent = self if current_parent == 0 else other
            
            for trait in segment_traits:
                if trait in parent.traits:
                    # Use trait from current parent
                    combined_traits[trait] = parent.traits[trait]
                    combination_record["traits"][trait] = {
                        "source": "parent1" if current_parent == 0 else "parent2",
                        "value": parent.traits[trait]
                    }
            
            # Switch parent
            current_parent = 1 - current_parent
            start_idx = end_idx
        
        # Handle remaining traits
        remaining_traits = all_traits[start_idx:]
        parent = self if current_parent == 0 else other
        
        for trait in remaining_traits:
            if trait in parent.traits:
                combined_traits[trait] = parent.traits[trait]
                combination_record["traits"][trait] = {
                    "source": "parent1" if current_parent == 0 else "parent2",
                    "value": parent.traits[trait]
                }
        
        # Special handling for traits that could benefit from averaging
        blend_traits = ["energy_efficiency", "stability", "adaptation_rate"]
        for trait in blend_traits:
            if trait in self.traits and trait in other.traits:
                # Use weighted average with random factor for genetic diversity
                weight = np.random.beta(2, 2)  # Beta distribution for natural variation
                blended_value = (
                    weight * self.traits[trait] +
                    (1 - weight) * other.traits[trait]
                )
                
                # Add some random variation
                variation = np.random.normal(0, 0.05)
                combined_traits[trait] = np.clip(blended_value + variation, 0.0, 1.0)
                
                combination_record["traits"][trait] = {
                    "source": "blended",
                    "parent1": self.traits[trait],
                    "parent2": other.traits[trait],
                    "weight": weight,
                    "combined": combined_traits[trait]
                }
        
        # Handle dormant genes - combine and potentially discover new ones
        combined_dormant = self._combine_dormant_genes(other)
        
        # Create new DNA sequence with combined traits
        new_history = self.evolution_history + [combination_record]
        generation = max(self.generation, other.generation) + 1
        
        # Average of parents' stability with slight random variation
        combined_stability = ((self.stability_factor + other.stability_factor) / 2) * np.random.uniform(0.95, 1.05)
        
        return DNASequence(
            sequence_id=f"combined_{uuid.uuid4().hex[:8]}",
            traits=combined_traits,
            generation=generation,
            parent_id=f"{self.sequence_id}+{other.sequence_id}",
            mutation_rate=(self.mutation_rate + other.mutation_rate) / 2,
            stability_factor=np.clip(combined_stability, 0.5, 0.99),
            evolution_history=new_history,
            dormant_genes=combined_dormant,
            activation_thresholds=self._combine_activation_thresholds(other)
        )

    def _combine_dormant_genes(self, other: 'DNASequence') -> Dict[str, float]:
        """Combine dormant genes from both parents."""
        combined_dormant = {}
        
        # Include dormant genes from both parents
        all_genes = set(list(self.dormant_genes.keys()) + list(other.dormant_genes.keys()))
        
        for gene in all_genes:
            if gene in self.dormant_genes and gene in other.dormant_genes:
                # Average if both have the gene
                combined_dormant[gene] = (self.dormant_genes[gene] + other.dormant_genes[gene]) / 2
            elif gene in self.dormant_genes:
                combined_dormant[gene] = self.dormant_genes[gene]
            else:
                combined_dormant[gene] = other.dormant_genes[gene]
        
        # Small chance to discover entirely new dormant gene during combination
        if np.random.random() < 0.05:
            new_gene_candidates = [
                "hyper_adaptation", "multi_dimensional_reasoning", "quantum_intuition",
                "holographic_memory", "non_linear_causality", "spontaneous_mutation"
            ]
            available = [g for g in new_gene_candidates if g not in combined_dormant]
            
            if available:
                new_gene = np.random.choice(available)
                combined_dormant[new_gene] = np.random.uniform(0.1, 0.3)
                logger.info(f"New dormant gene discovered during combination: {new_gene}")
        
        return combined_dormant

    def _combine_activation_thresholds(self, other: 'DNASequence') -> Dict[str, float]:
        """Combine activation thresholds from both parents."""
        combined_thresholds = {}
        
        # Include thresholds from both parents
        all_genes = set(list(self.activation_thresholds.keys()) + list(other.activation_thresholds.keys()))
        
        for gene in all_genes:
            if gene in self.activation_thresholds and gene in other.activation_thresholds:
                # Use lower threshold (easier activation) with slight randomization
                threshold = min(self.activation_thresholds[gene], other.activation_thresholds[gene])
                combined_thresholds[gene] = threshold * np.random.uniform(0.95, 1.05)
            elif gene in self.activation_thresholds:
                combined_thresholds[gene] = self.activation_thresholds[gene]
            else:
                combined_thresholds[gene] = other.activation_thresholds[gene]
        
        return combined_thresholds

    def save(self, path: Union[str, Path]) -> None:
        """Saves the DNA sequence to a file."""
        data = {
            "sequence_id": self.sequence_id,
            "traits": self.traits,
            "generation": self.generation,
            "parent_id": self.parent_id,
            "mutation_rate": self.mutation_rate,
            "stability_factor": self.stability_factor,
            "evolution_history": self.evolution_history,
            "creation_timestamp": self.creation_timestamp,
            "dormant_genes": self.dormant_genes,
            "activation_thresholds": self.activation_thresholds,
            "energy_signature": self.energy_signature.tolist()
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)
        logger.info(f"DNA sequence saved to {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'DNASequence':
        """Loads a DNA sequence from a file."""
        with open(path, 'r') as f:
            data = json.load(f)
            
        # Convert energy signature back to numpy array
        if "energy_signature" in data and isinstance(data["energy_signature"], list):
            data["energy_signature"] = np.array(data["energy_signature"])
            
        return cls(**data)

    def evaluate_fitness(self, environment_state: Dict[str, float]) -> float:
        """
        Evaluates the fitness of this DNA sequence in a given environment.
        
        Args:
            environment_state: Dictionary containing environmental parameters
            
        Returns:
            Fitness score (0.0-1.0)
        """
        if not environment_state:
            return 0.5  # Neutral fitness in undefined environment
            
        fitness_scores = []
        
        # Evaluate each trait's contribution to fitness
        for trait, value in self.traits.items():
            effective_value = self.get_effective_trait(trait)
            
            if trait in environment_state:
                # Calculate how well the trait matches environmental demands
                target_value = environment_state[trait]
                raw_difference = abs(effective_value - target_value)
                
                # Use non-linear scaling for differences
                if raw_difference < 0.1:
                    # Small differences are rewarded
                    trait_fitness = 1.0 - (raw_difference * 0.5)
                else:
                    # Larger differences are penalized more heavily
                    trait_fitness = max(0, 1.0 - (raw_difference * 2))
                    
                fitness_scores.append(trait_fitness)
        
        # Special handling for dominant environment characteristics
        dominant_traits = {k: v for k, v in environment_state.items() 
                         if v > 0.8}  # Critical environment factors
        
        if dominant_traits:
            # Calculate special fitness for high-demand traits
            dominant_scores = []
            for trait, target in dominant_traits.items():
                if trait in self.traits:
                    effective_value = self.get_effective_trait(trait)
                    match_quality = 1.0 - abs(effective_value - target)
                    dominant_scores.append(match_quality)
            
            # Weight dominant traits more heavily in final fitness
            if dominant_scores:
                dominant_fitness = np.mean(dominant_scores)
                # Combine regular and dominant fitness with emphasis on dominant
                overall_fitness = (np.mean(fitness_scores) * 0.4 + dominant_fitness * 0.6 
                                if fitness_scores else dominant_fitness)
                return overall_fitness
        
        # Regular fitness calculation if no dominant traits
        return np.mean(fitness_scores) if fitness_scores else 0.0

    def analyze_evolution(self) -> Dict:
        """Analyzes the evolutionary history of the DNA sequence."""
        if not self.evolution_history:
            return {"error": "No evolution history available"}
            
        analysis = {
            "generations": self.generation,
            "mutation_count": 0,
            "trait_trends": {trait: [] for trait in self.traits},
            "significant_changes": [],
            "environmental_adaptation": {},
            "divergence_from_ancestor": 0.0
        }

        # Extract first ancestor's traits if available
        first_ancestor_traits = None
        if len(self.evolution_history) > 0 and "parent_id" in self.evolution_history[0]:
            ancestor_id = self.evolution_history[0]["parent_id"]
            for record in self.evolution_history:
                if record.get("parent_id") == ancestor_id and "mutations" in record:
                    first_ancestor_traits = {
                        trait: record["mutations"][trait]["previous"]
                        for trait in record["mutations"]
                    }
                    break

        # Analyze evolution records
        env_pressures = []
        for record in self.evolution_history:
            if "mutations" in record:
                mutation_count = len(record["mutations"])
                analysis["mutation_count"] += mutation_count
                
                # Track environmental pressure
                env_pressure = record.get("environmental_pressure")
                if env_pressure is not None:
                    env_pressures.append(env_pressure)
                
                # Track significant changes
                for trait, change in record["mutations"].items():
                    if abs(change["change"]) > 0.1:  # Significant change threshold
                        analysis["significant_changes"].append({
                            "generation": record["generation"],
                            "trait": trait,
                            "change": change["change"],
                            "env_pressure": env_pressure
                        })
                    
                    # Track trait values over time
                    if trait in analysis["trait_trends"]:
                        analysis["trait_trends"][trait].append({
                            "generation": record["generation"],
                            "value": change["previous"] + change["change"]
                        })

        # Calculate environmental adaptation correlation
        if env_pressures and len(env_pressures) > 2:
            for trait in self.traits:
                if trait in analysis["trait_trends"] and len(analysis["trait_trends"][trait]) > 2:
                    trait_values = [item["value"] for item in analysis["trait_trends"][trait]]
                    if len(trait_values) == len(env_pressures):
                        correlation = np.corrcoef(trait_values, env_pressures)[0, 1]
                        analysis["environmental_adaptation"][trait] = correlation

        # Calculate divergence from ancestor
        if first_ancestor_traits:
            differences = []
            for trait, value in self.traits.items():
                if trait in first_ancestor_traits:
                    diff = abs(value - first_ancestor_traits[trait])
                    differences.append(diff)
            
            if differences:
                analysis["divergence_from_ancestor"] = float(np.mean(differences))

        return analysis

class DNAPool:
    """Manages a population of DNA sequences for evolution simulation."""
    
    def __init__(self, initial_population: int = 10, max_population: int = 100):
        self.dna_sequences: Dict[str, DNASequence] = {}
        self.fitness_scores: Dict[str, float] = {}
        self.max_population = max_population
        self.generation_counter = 0
        self.statistics = {
            "avg_fitness": [],
            "max_fitness": [],
            "diversity": [],
            "population_size": []
        }
        
        # Initialize population
        self._initialize_population(initial_population)
        
    def _initialize_population(self, size: int) -> None:
        """Initialize a diverse population of DNA sequences."""
        for i in range(size):
            # Create with random trait variations
            traits = {
                "learning_rate": np.random.uniform(0.3, 0.7),
                "energy_efficiency": np.random.uniform(0.4, 0.8),
                "processing_speed": np.random.uniform(0.3, 0.7),
                "memory_capacity": np.random.uniform(0.4, 0.7),
                "adaptation_rate": np.random.uniform(0.3, 0.7),
                "stability": np.random.uniform(0.4, 0.8),
                "specialization": np.random.uniform(0.2, 0.6),
                "pattern_recognition": np.random.uniform(0.3, 0.7),
                "creativity": np.random.uniform(0.2, 0.8),
                "self_preservation": np.random.uniform(0.3, 0.8)
            }
            
            # Vary stability and mutation rate for diversity
            mutation_rate = np.random.uniform(0.005, 0.02)
            stability = np.random.uniform(0.85, 0.98)
            
            dna = DNASequence(
                sequence_id=f"gen0_dna_{i}",
                traits=traits,
                generation=0,
                mutation_rate=mutation_rate,
                stability_factor=stability
            )
            
            self.dna_sequences[dna.sequence_id] = dna
            
    def evolve_generation(self, environment_state: Dict[str, float], pressure: float = 1.0) -> None:
        """
        Evolve the population through one generation.
        
        Args:
            environment_state: Current environment conditions
            pressure: Environmental pressure factor
        """
        self.generation_counter += 1
        
        # Calculate fitness for all sequences
        self._calculate_fitness(environment_state)
        
        # Record statistics before evolution
        self._update_statistics()
        
        # Select parents based on fitness
        parents = self._select_parents()
        
        # Create new generation through reproduction
        new_generation = {}
        
        # Elite preservation - keep top performers unchanged
        elite_count = max(1, int(len(self.dna_sequences) * 0.1))  # Keep top 10%
        for seq_id in self._get_top_sequences(elite_count):
            elite_dna = self.dna_sequences[seq_id]
            new_generation[seq_id] = elite_dna
        
        # Fill remaining population
        offspring_count = min(
            self.max_population - elite_count,
            int(len(self.dna_sequences) * 1.5)  # Allow 50% growth but cap at max
        )
        
        # Create offspring through different mechanisms
        for i in range(offspring_count):
            if np.random.random() < 0.7:  # 70% sexual reproduction
                # Sexual reproduction (with crossover)
                parent1, parent2 = np.random.choice(parents, 2, replace=False)
                dna1 = self.dna_sequences[parent1]
                dna2 = self.dna_sequences[parent2]
                
                # Vary crossover points based on genetic diversity
                diversity = self._calculate_similarity(dna1, dna2)
                crossover_points = 1 if diversity < 0.3 else (2 if diversity < 0.7 else 3)
                
                offspring = dna1.combine(dna2, crossover_points=crossover_points)
                
            else:  # 30% asexual reproduction (mutation only)
                parent = np.random.choice(parents)
                dna = self.dna_sequences[parent]
                
                # Environmental pressure affects mutation intensity
                local_pressure = pressure * np.random.uniform(0.8, 1.2)  # Local variation
                offspring = dna.mutate(environmental_pressure=local_pressure)
            
            new_generation[offspring.sequence_id] = offspring
        
        # Replace population with new generation
        self.dna_sequences = new_generation
        
        # Clear fitness scores for new generation
        self.fitness_scores = {}
        
        logger.info(f"Generation {self.generation_counter} evolved: {len(self.dna_sequences)} individuals")
    
    def _calculate_fitness(self, environment_state: Dict[str, float]) -> None:
        """Calculate fitness scores for all DNA sequences."""
        for seq_id, dna in self.dna_sequences.items():
            fitness = dna.evaluate_fitness(environment_state)
            self.fitness_scores[seq_id] = fitness
    
    def _select_parents(self) -> List[str]:
        """Select parents for next generation using tournament selection."""
        if not self.fitness_scores:
            return list(self.dna_sequences.keys())  # If no fitness calculated, use all
            
        # Use tournament selection
        population_ids = list(self.dna_sequences.keys())
        selected_parents = []
        
        # Select approximately half the population as parents
        num_parents = max(2, len(population_ids) // 2)
        tournament_size = max(2, len(population_ids) // 10)
        
        for _ in range(num_parents):
            # Random tournament
            tournament = np.random.choice(population_ids, tournament_size, replace=False)
            
            # Select winner based on fitness
            winner = max(tournament, key=lambda seq_id: self.fitness_scores.get(seq_id, 0))
            selected_parents.append(winner)
        
        return selected_parents
    
    def _get_top_sequences(self, count: int) -> List[str]:
        """Get the top performing sequences by fitness."""
        if not self.fitness_scores:
            # If no fitness calculated, select random individuals
            return list(np.random.choice(list(self.dna_sequences.keys()), 
                                      min(count, len(self.dna_sequences)), 
                                      replace=False))
        
        # Sort by fitness and return top count
        sorted_seqs = sorted(self.fitness_scores.items(), key=lambda x: x[1], reverse=True)
        return [seq_id for seq_id, _ in sorted_seqs[:count]]
    
    def _calculate_similarity(self, dna1: DNASequence, dna2: DNASequence) -> float:
        """Calculate genetic similarity between two DNA sequences."""
        # Get common traits
        common_traits = set(dna1.traits.keys()) & set(dna2.traits.keys())
        
        if not common_traits:
            return 0.0
            
        # Calculate average trait difference
        differences = []
        for trait in common_traits:
            diff = abs(dna1.traits[trait] - dna2.traits[trait])
            differences.append(diff)
            
        # Return similarity (1.0 = identical, 0.0 = completely different)
        return 1.0 - (sum(differences) / len(differences))
    
    def _update_statistics(self) -> None:
        """Update population statistics."""
        if self.fitness_scores:
            fitness_values = list(self.fitness_scores.values())
            self.statistics["avg_fitness"].append(np.mean(fitness_values))
            self.statistics["max_fitness"].append(np.max(fitness_values))
        else:
            self.statistics["avg_fitness"].append(0.0)
            self.statistics["max_fitness"].append(0.0)
            
        # Calculate genetic diversity
        pairwise_similarities = []
        dna_ids = list(self.dna_sequences.keys())
        if len(dna_ids) >= 2:
            # Sample pairs to avoid O(n²) comparisons for large populations
            sample_size = min(100, len(dna_ids) * (len(dna_ids) - 1) // 2)
            pairs_sampled = 0
            
            while pairs_sampled < sample_size and pairs_sampled < len(dna_ids) * (len(dna_ids) - 1) // 2:
                idx1, idx2 = np.random.choice(len(dna_ids), 2, replace=False)
                dna1 = self.dna_sequences[dna_ids[idx1]]
                dna2 = self.dna_sequences[dna_ids[idx2]]
                
                similarity = self._calculate_similarity(dna1, dna2)
                pairwise_similarities.append(similarity)
                pairs_sampled += 1
                
            diversity = 1.0 - np.mean(pairwise_similarities)
        else:
            diversity = 0.0
            
        self.statistics["diversity"].append(diversity)
        self.statistics["population_size"].append(len(self.dna_sequences))
    
    def get_most_fit(self, count: int = 1) -> List[DNASequence]:
        """Get the most fit individuals."""
        top_ids = self._get_top_sequences(count)
        return [self.dna_sequences[seq_id] for seq_id in top_ids]
    
    def get_statistics(self) -> Dict:
        """Get population evolution statistics."""
        return {
            "current_generation": self.generation_counter,
            "population_size": len(self.dna_sequences),
            "statistics": self.statistics,
            "trait_distribution": self._calculate_trait_distribution(),
            "genetic_diversity": self.statistics["diversity"][-1] if self.statistics["diversity"] else 0.0
        }
    
    def _calculate_trait_distribution(self) -> Dict[str, Dict[str, float]]:
        """Calculate distribution statistics for each trait."""
        if not self.dna_sequences:
            return {}
            
        # Find all traits across all individuals
        all_traits = set()
        for dna in self.dna_sequences.values():
            all_traits.update(dna.traits.keys())
            
        # Calculate statistics for each trait
        distribution = {}
        for trait in all_traits:
            values = [dna.traits.get(trait, 0.0) for dna in self.dna_sequences.values() 
                     if trait in dna.traits]
            
            if values:
                distribution[trait] = {
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values))
                }
            
        return distribution
    
    def save_population(self, directory: str) -> None:
        """Save entire population to a directory."""
        path = Path(directory)
        path.mkdir(exist_ok=True, parents=True)
        
        # Save individual DNA sequences
        for seq_id, dna in self.dna_sequences.items():
            dna_path = path / f"{seq_id}.json"
            dna.save(dna_path)
            
        # Save population metadata
        metadata = {
            "generation": self.generation_counter,
            "population_size": len(self.dna_sequences),
            "dna_ids": list(self.dna_sequences.keys()),
            "statistics": self.statistics,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(path / "population_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=4)
            
        logger.info(f"Population saved to {directory}")
    
    @classmethod
    def load_population(cls, directory: str) -> 'DNAPool':
        """Load entire population from a directory."""
        path = Path(directory)
        
        # Load metadata
        with open(path / "population_metadata.json", 'r') as f:
            metadata = json.load(f)
            
        # Create pool
        pool = cls(initial_population=0, max_population=100)
        pool.generation_counter = metadata["generation"]
        pool.statistics = metadata["statistics"]
        
        # Load DNA sequences
        for dna_id in metadata["dna_ids"]:
            dna_path = path / f"{dna_id}.json"
            if dna_path.exists():
                dna = DNASequence.load(dna_path)
                pool.dna_sequences[dna.sequence_id] = dna
                
        logger.info(f"Loaded population from {directory}: {len(pool.dna_sequences)} individuals")
        return pool
