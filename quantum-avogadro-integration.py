import os
import sys
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import threading
import queue
import time

# Qt imports for visualization
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QComboBox, QLabel, QSplitter, QTabWidget,
    QTableWidget, QTableWidgetItem, QDockWidget, QFileDialog
)
from PySide6.QtCore import Qt, Signal, Slot, QTimer, QThread, QMutex
from PySide6.QtGui import QColor, QPalette

# Avogadro imports
import avogadro
from avogadro.core import Molecule, Atom, Bond
from avogadro.qtgui import MoleculeViewWidget, ToolPluginFactory
try:
    from avogadro.io import FileFormatManager
except ImportError:
    # Handle different Avogadro import structures
    pass

# RDKit imports
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

# Import our quantum simulator components
# This assumes we have access to the QuantumDrugSimulator class from our previous implementations
# In a real application, these would be proper imports from modules

class QuantumState:
    """Represents the quantum state of a molecule for visualization"""
    def __init__(self, 
                wavefunction: np.ndarray, 
                energy: float, 
                electron_density: Optional[np.ndarray] = None,
                entanglement_map: Optional[Dict[int, List[int]]] = None):
        self.wavefunction = wavefunction
        self.energy = energy
        self.electron_density = electron_density
        self.entanglement_map = entanglement_map or {}
        self.timestamp = time.time()

class QuantumRenderer(QThread):
    """Thread for rendering quantum states"""
    stateUpdated = Signal(object)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.mutex = QMutex()
        self.state_queue = queue.Queue(maxsize=10)
        self.running = True
        
    def enqueue_state(self, state: QuantumState):
        try:
            # Remove oldest state if queue is full
            if self.state_queue.full():
                self.state_queue.get_nowait()
            self.state_queue.put_nowait(state)
        except queue.Full:
            pass
            
    def run(self):
        while self.running:
            try:
                state = self.state_queue.get(timeout=0.1)
                self.stateUpdated.emit(state)
            except queue.Empty:
                time.sleep(0.01)
                
    def stop(self):
        self.running = False
        self.wait()

class QuantumMoleculeWidget(MoleculeViewWidget):
    """Enhanced Avogadro molecule viewer with quantum visualization"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.quantum_overlay_enabled = True
        self.current_state = None
        self.atom_visualizations = {}
        self.setup_renderer()
        
    def setup_renderer(self):
        self.renderer = QuantumRenderer()
        self.renderer.stateUpdated.connect(self.update_quantum_overlay)
        self.renderer.start()
        
        # Set update timer for smooth animations
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update)
        self.update_timer.start(33)  # ~30 FPS
        
    @Slot(object)
    def update_quantum_overlay(self, state: QuantumState):
        """Update quantum overlay with new state"""
        self.current_state = state
        if self.molecule() and state:
            self._calculate_atom_visualizations(state)
        
    def _calculate_atom_visualizations(self, state: QuantumState):
        """Calculate visualization parameters for each atom"""
        if not self.molecule():
            return
            
        n_atoms = self.molecule().atomCount()
        if n_atoms == 0:
            return
            
        # Reset visualizations
        self.atom_visualizations = {}
        
        # Map quantum state to atoms
        wf = state.wavefunction
        state_dim = len(wf)
        chunk_size = max(1, state_dim // n_atoms)
        
        for i in range(n_atoms):
            # Extract relevant part of wavefunction
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, state_dim)
            
            if start_idx >= state_dim:
                break
                
            # Calculate quantum properties for visualization
            amplitudes = wf[start_idx:end_idx]
            probability = np.sum(np.abs(amplitudes)**2)
            phase = np.angle(np.sum(amplitudes))
            
            # Store visualization parameters
            self.atom_visualizations[i] = {
                'radius': 0.3 + 0.7 * probability,  # Scale to visible range
                'color': self._phase_to_color(phase),
                'opacity': min(1.0, 0.3 + 0.7 * probability),
                'probability': probability
            }
    
    def _phase_to_color(self, phase: float) -> QColor:
        """Convert quantum phase to color using HSV"""
        # Map phase from [-π, π] to [0, 1] for hue
        hue = (phase + np.pi) / (2 * np.pi)
        return QColor.fromHsvF(hue, 1.0, 1.0)
        
    def paintEvent(self, event):
        """Override paint event to add quantum overlay"""
        # Let Avogadro do the standard rendering
        super().paintEvent(event)
        
        # Add our quantum overlay
        if self.quantum_overlay_enabled and self.atom_visualizations and self.molecule():
            self._draw_quantum_overlay()
    
    def _draw_quantum_overlay(self):
        """Draw quantum visualization overlay"""
        try:
            # Get painter from avogadro
            painter = self.painter()
            
            # Draw overlays for each atom
            for atom_idx, vis_props in self.atom_visualizations.items():
                if atom_idx >= self.molecule().atomCount():
                    continue
                    
                # Get atom position
                atom = self.molecule().atom(atom_idx)
                if not atom:
                    continue
                    
                # Convert to screen coordinates
                pos3d = atom.position3d()
                screen_pos = self.camera().project(pos3d)
                
                # Draw quantum state representation
                radius = vis_props['radius'] * 20  # Scale up for visibility
                color = vis_props['color']
                color.setAlphaF(vis_props['opacity'])
                
                # Draw colored circle
                painter.setPen(Qt.NoPen)
                painter.setBrush(color)
                painter.drawEllipse(
                    screen_pos.x() - radius,
                    screen_pos.y() - radius,
                    radius * 2,
                    radius * 2
                )
                
                # Draw entanglement lines if available
                if self.current_state and self.current_state.entanglement_map:
                    if atom_idx in self.current_state.entanglement_map:
                        for entangled_idx in self.current_state.entanglement_map[atom_idx]:
                            if entangled_idx >= self.molecule().atomCount():
                                continue
                                
                            entangled_atom = self.molecule().atom(entangled_idx)
                            if not entangled_atom:
                                continue
                                
                            entangled_pos = self.camera().project(entangled_atom.position3d())
                            
                            # Draw entanglement line
                            painter.setPen(QColor(255, 255, 255, 100))  # Transparent white
                            painter.drawLine(screen_pos, entangled_pos)
        except Exception as e:
            # Catch any errors to prevent crashing the application
            logging.error(f"Error in quantum overlay: {str(e)}")

class AvogadroVisualizer:
    """Interface between our quantum system and Avogadro visualization"""
    
    def __init__(self, simulator=None):
        self.app = QApplication.instance() or QApplication(sys.argv)
        self.window = QMainWindow()
        self.simulator = simulator
        self.current_molecule_name = None
        self.molecule_mapping = {}  # Maps molecule names to Avogadro molecules
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the UI components"""
        self.window.setWindowTitle("Quantum Drug Discovery Visualizer")
        self.window.resize(1280, 800)
        
        # Create central widget
        central = QWidget()
        main_layout = QVBoxLayout(central)
        
        # Create toolbar
        toolbar = QHBoxLayout()
        
        # Add molecule selector
        self.molecule_combo = QComboBox()
        self.molecule_combo.addItem("Select Molecule...")
        self.molecule_combo.currentTextChanged.connect(self.on_molecule_selected)
        toolbar.addWidget(QLabel("Molecule:"))
        toolbar.addWidget(self.molecule_combo)
        
        # Add visualization controls
        self.view_combo = QComboBox()
        self.view_combo.addItems(["Quantum State", "Electron Density", "Binding Affinity"])
        self.view_combo.currentTextChanged.connect(self.on_view_changed)
        toolbar.addWidget(QLabel("View:"))
        toolbar.addWidget(self.view_combo)
        
        # Add buttons
        self.optimize_btn = QPushButton("Optimize Structure")
        self.optimize_btn.clicked.connect(self.on_optimize_clicked)
        toolbar.addWidget(self.optimize_btn)
        
        self.analyze_btn = QPushButton("Analyze")
        self.analyze_btn.clicked.connect(self.on_analyze_clicked)
        toolbar.addWidget(self.analyze_btn)
        
        # Add to main layout
        main_layout.addLayout(toolbar)
        
        # Create molecule viewer
        self.viewer = QuantumMoleculeWidget()
        main_layout.addWidget(self.viewer)
        
        # Set central widget
        self.window.setCentralWidget(central)
        
        # Add docks
        self.setup_docks()
        
    def setup_docks(self):
        """Set up dock widgets"""
        # Properties dock
        self.properties_dock = QDockWidget("Properties", self.window)
        self.properties_table = QTableWidget()
        self.properties_table.setColumnCount(2)
        self.properties_table.setHorizontalHeaderLabels(["Property", "Value"])
        self.properties_dock.setWidget(self.properties_table)
        self.window.addDockWidget(Qt.RightDockWidgetArea, self.properties_dock)
        
        # Quantum states dock
        self.quantum_dock = QDockWidget("Quantum State", self.window)
        self.quantum_table = QTableWidget()
        self.quantum_table.setColumnCount(2)
        self.quantum_table.setHorizontalHeaderLabels(["Parameter", "Value"])
        self.quantum_dock.setWidget(self.quantum_table)
        self.window.addDockWidget(Qt.RightDockWidgetArea, self.quantum_dock)
    
    def update_molecule_list(self):
        """Update the molecule selector with available molecules"""
        if not self.simulator:
            return
            
        # Save current selection
        current_text = self.molecule_combo.currentText()
        
        # Clear and repopulate list
        self.molecule_combo.clear()
        self.molecule_combo.addItem("Select Molecule...")
        
        # Add molecules from simulator
        for name in self.simulator.molecular_database.keys():
            self.molecule_combo.addItem(name)
            
        # Restore selection if possible
        idx = self.molecule_combo.findText(current_text)
        if idx >= 0:
            self.molecule_combo.setCurrentIndex(idx)
    
    def set_simulator(self, simulator):
        """Set the quantum simulator to use"""
        self.simulator = simulator
        self.update_molecule_list()
    
    def on_molecule_selected(self, molecule_name: str):
        """Handle molecule selection"""
        if molecule_name == "Select Molecule..." or not self.simulator:
            return
            
        # Check if molecule exists
        if molecule_name not in self.simulator.molecular_database:
            logging.warning(f"Molecule {molecule_name} not found in database")
            return
            
        # Save current molecule name
        self.current_molecule_name = molecule_name
        
        # Get molecule from simulator
        mol_data = self.simulator.molecular_database[molecule_name]
        rdkit_mol = mol_data['molecule']
        
        # Convert to Avogadro molecule
        avo_mol = self.rdkit_to_avogadro(rdkit_mol)
        
        # Store in mapping
        self.molecule_mapping[molecule_name] = avo_mol
        
        # Set in viewer
        self.viewer.setMolecule(avo_mol)
        
        # Update properties display
        self.update_properties(molecule_name)
        
        # Update quantum state
        self.update_quantum_state(molecule_name)
    
    def rdkit_to_avogadro(self, rdkit_mol) -> Molecule:
        """Convert RDKit molecule to Avogadro format"""
        try:
            # Create Avogadro molecule
            avo_mol = Molecule()
            
            # Convert atoms
            conf = rdkit_mol.GetConformer()
            for i in range(rdkit_mol.GetNumAtoms()):
                atom = rdkit_mol.GetAtomWithIdx(i)
                pos = conf.GetAtomPosition(i)
                
                # Add atom to Avogadro molecule
                avo_atom = avo_mol.addAtom(atom.GetAtomicNum())
                
                # Add position data
                avo_mol.atomPositions3d()[avo_atom.index()] = [pos.x, pos.y, pos.z]
                
            # Convert bonds
            for bond in rdkit_mol.GetBonds():
                begin_idx = bond.GetBeginAtomIdx()
                end_idx = bond.GetEndAtomIdx()
                order = int(bond.GetBondTypeAsDouble())
                avo_mol.addBond(begin_idx, end_idx, order)
                
            return avo_mol
            
        except Exception as e:
            logging.error(f"Error converting molecule: {str(e)}")
            # Return empty molecule on error
            return Molecule()
    
    def update_properties(self, molecule_name: str):
        """Update properties display for selected molecule"""
        if not self.simulator or molecule_name not in self.simulator.molecular_database:
            return
            
        # Get molecule data
        mol_data = self.simulator.molecular_database[molecule_name]
        descriptors = mol_data.get('descriptors', {})
        
        # Clear current properties
        self.properties_table.setRowCount(0)
        
        # Add basic properties
        self.add_property("Name", molecule_name)
        self.add_property("SMILES", mol_data['state'].smiles)
        self.add_property("Energy", f"{mol_data['state'].conformer_energy:.4f}")
        self.add_property("Stability", f"{mol_data['state'].stability_index:.4f}")
        
        # Add descriptors
        for name, value in descriptors.items():
            if value is not None:
                self.add_property(name, f"{value:.4f}" if isinstance(value, float) else str(value))
    
    def add_property(self, name: str, value: str):
        """Add a property to the properties table"""
        row = self.properties_table.rowCount()
        self.properties_table.insertRow(row)
        self.properties_table.setItem(row, 0, QTableWidgetItem(name))
        self.properties_table.setItem(row, 1, QTableWidgetItem(value))
    
    def update_quantum_state(self, molecule_name: str):
        """Update quantum state display for selected molecule"""
        if not self.simulator or molecule_name not in self.simulator.molecular_database:
            return
            
        # Get molecule data
        mol_data = self.simulator.molecular_database[molecule_name]
        
        # Extract quantum features
        quantum_features = mol_data['state'].quantum_features
        
        # Create quantum state object for visualization
        state = QuantumState(
            wavefunction=quantum_features.flatten(),
            energy=mol_data['state'].conformer_energy,
            electron_density=mol_data['state'].electron_density
        )
        
        # Update entanglement map
        state.entanglement_map = self._calculate_entanglement_map(quantum_features)
        
        # Send to renderer
        self.viewer.renderer.enqueue_state(state)
        
        # Update quantum state info table
        self.quantum_table.setRowCount(0)
        
        # Add quantum state info
        self.add_quantum_info("Energy", f"{state.energy:.6f}")
        self.add_quantum_info("State Dimension", str(len(state.wavefunction)))
        
        # Add more quantum information
        norm = np.linalg.norm(state.wavefunction)
        entropy = self._calculate_quantum_entropy(state.wavefunction)
        
        self.add_quantum_info("State Norm", f"{norm:.6f}")
        self.add_quantum_info("Quantum Entropy", f"{entropy:.6f}")
        
        # Calculate coherence measure
        coherence = self._calculate_coherence(state.wavefunction)
        self.add_quantum_info("Coherence", f"{coherence:.6f}")
    
    def add_quantum_info(self, name: str, value: str):
        """Add quantum information to the quantum table"""
        row = self.quantum_table.rowCount()
        self.quantum_table.insertRow(row)
        self.quantum_table.setItem(row, 0, QTableWidgetItem(name))
        self.quantum_table.setItem(row, 1, QTableWidgetItem(value))
    
    def _calculate_entanglement_map(self, quantum_features: np.ndarray) -> Dict[int, List[int]]:
        """Calculate atom-atom entanglement from quantum features"""
        # Simplified approximation of entanglement
        n = quantum_features.shape[0]
        entanglement_map = {}
        
        # Calculate correlation matrix
        flat_features = quantum_features.reshape(n, -1)
        if flat_features.shape[1] > 0:
            corr_matrix = np.abs(np.corrcoef(flat_features))
            
            # Threshold for significant entanglement
            threshold = 0.7
            
            # Find entangled pairs
            for i in range(n):
                entangled = []
                for j in range(n):
                    if i != j and corr_matrix[i, j] > threshold:
                        entangled.append(j)
                if entangled:
                    entanglement_map[i] = entangled
                    
        return entanglement_map
    
    def _calculate_quantum_entropy(self, wavefunction: np.ndarray) -> float:
        """Calculate von Neumann entropy of quantum state"""
        # Calculate density matrix
        density = np.outer(wavefunction, np.conj(wavefunction))
        
        # Calculate eigenvalues
        eigenvalues = np.linalg.eigvalsh(density)
        
        # Filter out negligible eigenvalues (numerical stability)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        
        # Calculate entropy
        entropy = -np.sum(eigenvalues * np.log2(eigenvalues))
        return np.real(entropy)
    
    def _calculate_coherence(self, wavefunction: np.ndarray) -> float:
        """Calculate quantum coherence measure"""
        # Simplified l1-norm of coherence
        density = np.outer(wavefunction, np.conj(wavefunction))
        abs_sum = np.sum(np.abs(density)) - np.sum(np.abs(np.diag(density)))
        return abs_sum
    
    def on_view_changed(self, view_name: str):
        """Handle view type change"""
        # Implement different visualization modes
        if view_name == "Quantum State":
            self.viewer.quantum_overlay_enabled = True
            if self.current_molecule_name:
                self.update_quantum_state(self.current_molecule_name)
                
        elif view_name == "Electron Density":
            # Switch to electron density visualization
            self.viewer.quantum_overlay_enabled = True
            if self.current_molecule_name and self.simulator:
                mol_data = self.simulator.molecular_database[self.current_molecule_name]
                
                # Create quantum state with electron density focus
                state = QuantumState(
                    wavefunction=mol_data['state'].quantum_features.flatten(),
                    energy=mol_data['state'].conformer_energy,
                    electron_density=mol_data['state'].electron_density
                )
                
                # Adjust visualization to highlight electron density
                self.viewer.renderer.enqueue_state(state)
                
        elif view_name == "Binding Affinity":
            # Implement binding affinity visualization
            self.viewer.quantum_overlay_enabled = True
            if self.current_molecule_name and self.simulator:
                mol_data = self.simulator.molecular_database[self.current_molecule_name]
                
                # Map binding affinities to visualization
                if mol_data['state'].binding_affinities:
                    # Create custom visualization based on binding data
                    state = self._create_binding_visualization(mol_data)
                    self.viewer.renderer.enqueue_state(state)
    
    def _create_binding_visualization(self, mol_data: Dict) -> QuantumState:
        """Create visualization state focused on binding affinities"""
        # Get quantum features as base
        wavefunction = mol_data['state'].quantum_features.flatten()
        
        # Get binding affinities
        binding_affinities = mol_data['state'].binding_affinities
        
        # Create state
        state = QuantumState(
            wavefunction=wavefunction,
            energy=mol_data['state'].conformer_energy
        )
        
        # Map binding data to atom-specific entanglement
        binding_map = {}
        if binding_affinities:
            # Create simplified visualization showing "binding hot spots"
            # In a real system, this would use actual binding site information
            rdkit_mol = mol_data['molecule']
            n_atoms = rdkit_mol.GetNumAtoms()
            
            # Find atoms likely involved in binding (simplified example)
            potential_binding_atoms = []
            for i in range(n_atoms):
                atom = rdkit_mol.GetAtomWithIdx(i)
                # Atoms with H-bond potential often involved in binding
                if atom.GetSymbol() in ['O', 'N'] or atom.GetFormalCharge() != 0:
                    potential_binding_atoms.append(i)
            
            # Connect these atoms in the visualization
            if potential_binding_atoms:
                for i in potential_binding_atoms:
                    binding_map[i] = [j for j in potential_binding_atoms if j != i]
        
        # Set entanglement map to reflect binding
        state.entanglement_map = binding_map
        
        return state
    
    async def on_optimize_clicked(self):
        """Handle structure optimization"""
        if not self.simulator or not self.current_molecule_name:
            return
            
        try:
            # Run quantum optimization
            results = await self.simulator.simulate_quantum_optimization(self.current_molecule_name)
            
            # Update molecule in viewer
            self.on_molecule_selected(self.current_molecule_name)
            
            # Show results
            QMessageBox.information(
                self.window,
                "Optimization Complete",
                f"Energy improved by {results['energy_improvement']:.4f}\n" +
                f"Final energy: {results['optimized_energy']:.4f}"
            )
            
        except Exception as e:
            QMessageBox.warning(
                self.window,
                "Optimization Failed",
                f"Error: {str(e)}"
            )
    
    async def on_analyze_clicked(self):
        """Handle analysis button click"""
        if not self.simulator or not self.current_molecule_name:
            return
            
        try:
            # Run screening against targets
            results = await self.simulator.screen_against_targets(self.current_molecule_name)
            
            # Show results in table format
            result_text = "Binding Affinities:\n\n"
            for target, scores in results.items():
                result_text += f"{target}:\n"
                result_text += f"  Binding Score: {scores['binding_score']:.4f}\n"
                result_text += f"  Combined Score: {scores['combined_score']:.4f}\n\n"
                
            QMessageBox.information(
                self.window,
                "Analysis Results",
                result_text
            )
            
            # Update visualization to reflect new binding data
            if self.view_combo.currentText() == "Binding Affinity":
                self.on_view_changed("Binding Affinity")
            
        except Exception as e:
            QMessageBox.warning(
                self.window,
                "Analysis Failed",
                f"Error: {str(e)}"
            )
    
    def show(self):
        """Show the visualizer window"""
        self.window.show()
    
    def close(self):
        """Close the visualizer"""
        self.window.close()
        
    def run(self):
        """Run the application event loop"""
        self.show()
        return self.app.exec()

# Example of how to use the visualizer with our simulator
async def launch_visualizer(simulator):
    """Launch the Avogadro visualizer with our quantum simulator"""
    # Create visualizer
    visualizer = AvogadroVisualizer(simulator)
    
    # Start visualization
    visualizer.show()
    
    # Return visualizer for further use
    return visualizer
