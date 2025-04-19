# core/GrowthLaws.py
import logging
import random
import math
from typing import List, TYPE_CHECKING, Dict, Any

# Avoid circular import for type hinting
if TYPE_CHECKING:
    from .BaseNode import BaseNode

logger = logging.getLogger(__name__)

class GrowthLaws:
    """
    Defines the rules governing node evolution, including energy dynamics,
    stress response, replication triggers, and mutation during replication.
    """

    def __init__(self,
                 replication_energy_threshold: float = 150.0,
                 replication_stress_factor: float = 0.8, # Stress below this factor encourages replication
                 energy_gain_base: float = 2.0,         # Base energy gain per cycle
                 stress_impact_on_gain: float = 0.5,    # How much stress reduces energy gain
                 decay_base_rate: float = 0.1,          # Base energy decay per cycle
                 connection_reward_energy: float = 0.5, # Energy bonus per connection
                 mutation_rate_multiplier: float = 1.0): # Multiplier for DNA mutation rate
        """
        Initializes the GrowthLaws with configurable parameters.

        Args:
            replication_energy_threshold: Energy level needed to consider replication.
            replication_stress_factor: Node stress level relative to this affects replication chance.
                                       Lower stress might make replication more likely.
            energy_gain_base: Base energy gain per cycle before modifiers.
            stress_impact_on_gain: Factor determining how stress reduces energy gain.
            decay_base_rate: Base rate for natural energy decay.
            connection_reward_energy: Small energy bonus per active connection.
            mutation_rate_multiplier: Adjusts the base mutation rate defined in node DNA.
        """
        self.repl_energy_thresh = replication_energy_threshold
        self.repl_stress_factor = replication_stress_factor
        self.energy_gain_base = energy_gain_base
        self.stress_impact_gain = stress_impact_on_gain
        self.decay_base_rate = decay_base_rate
        self.conn_reward = connection_reward_energy
        self.mutation_multiplier = mutation_rate_multiplier

        logger.info("GrowthLaws initialized.")

    def apply(self, nodes: List['BaseNode'], global_stress: float = 0.0):
        """
        Applies the growth laws to a list of nodes for one time step.

        Args:
            nodes: A list of BaseNode instances to process.
            global_stress: A general environmental stress level affecting all nodes.
        """
        if not nodes:
            return

        start_time = time.time()
        node_count = len(nodes)
        replication_signals = 0

        for node in nodes:
            if node.state.energy <= 0: # Skip dead nodes
                 continue

            # 1. Apply Decay and Stress Recovery (inherent node process)
            node.decay(self.decay_base_rate)
            if node.state.energy <= 0: # Check again after decay
                 continue

            # 2. Apply Global Stress
            node.apply_stress(global_stress) # Node's resistance applies internally

            # 3. Simulate Energy Gain (influenced by stress and connections)
            self._simulate_energy_gain(node)

            # 4. Check Replication Conditions
            can_replicate = self._check_replication(node)
            node.state.ready_to_replicate = can_replicate # Set the flag on the node
            if can_replicate:
                replication_signals += 1

        duration = time.time() - start_time
        avg_energy = sum(n.state.energy for n in nodes if n.state.energy > 0) / node_count if node_count > 0 else 0
        logger.debug(f"GrowthLaws applied to {node_count} nodes in {duration:.4f}s. Avg Energy: {avg_energy:.2f}. Replication signals: {replication_signals}.")


    def _simulate_energy_gain(self, node: 'BaseNode'):
        """Calculates and applies energy gain for a node."""
        # Base gain modified by efficiency
        base_gain = self.energy_gain_base * node.state.dna.get("energy_efficiency", 1.0)

        # Stress impact: Higher stress reduces gain
        stress_factor = max(0, 1.0 - (node.state.stress_level * self.stress_impact_gain)) # Can reduce gain to 0

        # Connection reward: More connections slightly boost gain
        connection_bonus = len(node.connections) * self.conn_reward

        total_gain = (base_gain * stress_factor) + connection_bonus
        node.state.energy += total_gain
        # logger.debug(f"Node {node.id} energy gain: {total_gain:.2f} (Base: {base_gain:.2f}, StressFactor: {stress_factor:.2f}, ConnBonus: {connection_bonus:.2f}). New Energy: {node.state.energy:.2f}")


    def _check_replication(self, node: 'BaseNode') -> bool:
        """
        Determines if a node should signal readiness for replication.
        Conditions: High enough energy, low enough stress (relative).
        """
        if node.state.energy >= self.repl_energy_thresh:
            # Replication is more likely if internal stress is relatively low
            stress_modifier = max(0.1, 1.0 - (node.state.stress_level / (self.repl_energy_thresh * 0.5))) # Example scaling
            replication_chance = stress_modifier * 0.5 # Base 50% chance if energy is met and stress is zero

            if random.random() < replication_chance:
                 logger.info(f"Node {node.id} meets replication criteria. Energy: {node.state.energy:.2f}, StressMod: {stress_modifier:.2f}")
                 return True
        return False

    def get_mutated_dna_for_child(self, parent_node: 'BaseNode') -> Dict[str, Any]:
        """
        Creates a mutated DNA copy for a new child node.
        Uses the parent's DNA and applies mutations based on parent's mutation rate
        and the GrowthLaws' global multiplier.
        """
        parent_dna = parent_node.state.dna
        new_dna = parent_dna.copy()
        base_mutation_rate = parent_dna.get("mutation_rate", 0.01)
        effective_mutation_rate = base_mutation_rate * self.mutation_multiplier

        logger.debug(f"Generating child DNA from {parent_node.id}. BaseRate: {base_mutation_rate:.4f}, EffectiveRate: {effective_mutation_rate:.4f}")

        for gene, value in new_dna.items():
            if isinstance(value, (int, float)) and random.random() < effective_mutation_rate:
                # Apply mutation: +/- percentage variation
                variation = 1.0 + random.uniform(-0.15, 0.15) # Up to 15% change
                new_dna[gene] = value * variation
                # Ensure logical bounds if necessary (e.g., efficiency > 0)
                if gene == "energy_efficiency" or gene == "stress_resistance":
                    new_dna[gene] = max(0.1, new_dna[gene]) # Keep resistance/efficiency positive
                logger.info(f"Mutation in child DNA (Gene: {gene}): {value:.3f} -> {new_dna[gene]:.3f}")

        # Chance to slightly change mutation rate itself
        if random.random() < effective_mutation_rate * 0.1: # Lower chance for meta-mutation
            variation = 1.0 + random.uniform(-0.2, 0.2)
            new_dna["mutation_rate"] = max(0.001, parent_dna["mutation_rate"] * variation) # Clamp min mutation rate
            logger.info(f"Mutation rate itself mutated: {parent_dna['mutation_rate']:.4f} -> {new_dna['mutation_rate']:.4f}")

        return new_dna

    def mutate(self) -> 'GrowthLaws':
        """
        Creates a slightly mutated version of these GrowthLaws.
        Useful if the environment itself can change.
        """
        new_laws = GrowthLaws(
            replication_energy_threshold=self.repl_energy_thresh * random.uniform(0.95, 1.05),
            replication_stress_factor=self.repl_stress_factor * random.uniform(0.95, 1.05),
            energy_gain_base=self.energy_gain_base * random.uniform(0.95, 1.05),
            stress_impact_on_gain=self.stress_impact_gain * random.uniform(0.95, 1.05),
            decay_base_rate=self.decay_base_rate * random.uniform(0.95, 1.05),
            connection_reward_energy=self.conn_reward * random.uniform(0.95, 1.05),
            mutation_rate_multiplier=self.mutation_multiplier * random.uniform(0.95, 1.05)
        )
        logger.info("GrowthLaws themselves mutated.")
        return new_laws
