import numpy as np
import tensorflow as tf
from skimage.measure import marching_cubes
import plotly.graph_objects as go
from typing import Tuple, List, Dict
from scipy.ndimage import gaussian_filter

class MolecularStructure:
    def __init__(self, resolution: int = 64):
        self.resolution = resolution
        self.grid_spacing = 2.0 / resolution
        self.electron_density = np.zeros((resolution, resolution, resolution))
        self.atoms = []  # List to store atomic positions and types
        
    def add_atom(self, position: Tuple[float, float, float], 
                 atom_type: str, radius: float = 1.0):
        """Add an atom to the molecular structure."""
        self.atoms.append({
            'position': position,
            'type': atom_type,
            'radius': radius
        })
        
    def compute_electron_density(self):
        """Compute electron density field from atomic positions."""
        # Create coordinate grid centered in the cube
        x, y, z = np.mgrid[-0.5:0.5:self.resolution*1j, 
                          -0.5:0.5:self.resolution*1j, 
                          -0.5:0.5:self.resolution*1j]
        
        # Initialize electron density field
        self.electron_density = np.zeros_like(x)
        
        # Atomic radii and electron contributions (simplified model)
        atom_properties = {
            'H': {'radius': 0.1, 'electrons': 1},
            'C': {'radius': 0.25, 'electrons': 6},
            'N': {'radius': 0.23, 'electrons': 7},
            'O': {'radius': 0.22, 'electrons': 8},
        }
        
        # Scale factor to ensure molecule fits in cube
        scale_factor = 0.3  # Adjust this to change molecule size
        
        # Compute electron density as sum of Gaussian distributions
        for atom in self.atoms:
            pos = tuple(p * scale_factor for p in atom['position'])
            atom_type = atom['type']
            props = atom_properties.get(atom_type, {'radius': 0.2, 'electrons': 1})
            
            # Calculate distance from each point to atom center
            r2 = (x - pos[0])**2 + (y - pos[1])**2 + (z - pos[2])**2
            
            # Add Gaussian electron density contribution
            sigma = props['radius'] * scale_factor
            amplitude = props['electrons']
            self.electron_density += amplitude * np.exp(-r2 / (2 * sigma**2))
            
        # Normalize electron density
        self.electron_density /= np.max(self.electron_density)
    
    def visualize_molecule(self, density_levels: int = 3, 
                          show_atoms: bool = True):
        """Visualize the molecular structure with electron density isosurfaces."""
        fig = go.Figure()
        
        # Add cube edges to show boundaries
        cube_vertices = np.array([
            [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5]
        ])
        
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # bottom face
            [4, 5], [5, 6], [6, 7], [7, 4],  # top face
            [0, 4], [1, 5], [2, 6], [3, 7]   # vertical edges
        ]
        
        for edge in edges:
            fig.add_trace(go.Scatter3d(
                x=[cube_vertices[edge[0]][0], cube_vertices[edge[1]][0]],
                y=[cube_vertices[edge[0]][1], cube_vertices[edge[1]][1]],
                z=[cube_vertices[edge[0]][2], cube_vertices[edge[1]][2]],
                mode='lines',
                line=dict(color='gray', width=1),
                showlegend=False
            ))
        
        # Add electron density isosurfaces
        if np.max(self.electron_density) > 0:
            levels = np.linspace(0.1, 0.8, density_levels)
            
            for level in levels:
                try:
                    verts, faces, _, _ = marching_cubes(
                        self.electron_density,
                        level=level,
                        spacing=(self.grid_spacing, self.grid_spacing, self.grid_spacing)
                    )
                    
                    # Scale vertices to fit inside cube
                    verts = verts - 1.0  # Center at origin
                    verts = verts * self.grid_spacing  # Scale to proper size
                    
                    opacity = 0.3 * level
                    
                    fig.add_trace(go.Mesh3d(
                        x=verts[:, 0],
                        y=verts[:, 1],
                        z=verts[:, 2],
                        i=faces[:, 0],
                        j=faces[:, 1],
                        k=faces[:, 2],
                        opacity=opacity,
                        colorscale="Viridis",
                        intensity=np.ones(len(verts)) * level,
                        name=f'Density {level:.2f}'
                    ))
                except Exception as e:
                    print(f"Warning: Could not generate isosurface at level {level}: {str(e)}")
        
        # Add atoms as spheres
        if show_atoms:
            atom_colors = {
                'H': 'white',
                'C': 'grey',
                'N': 'blue',
                'O': 'red'
            }
            
            scale_factor = 0.3  # Match scale factor from compute_electron_density
            
            for atom in self.atoms:
                pos = tuple(p * scale_factor for p in atom['position'])
                atom_type = atom['type']
                radius = atom['radius'] * scale_factor
                color = atom_colors.get(atom_type, 'purple')
                
                fig.add_trace(go.Scatter3d(
                    x=[pos[0]],
                    y=[pos[1]],
                    z=[pos[2]],
                    mode='markers',
                    marker=dict(
                        size=15 * radius,
                        color=color,
                        symbol='circle',
                        line=dict(
                            color='white',
                            width=1
                        )
                    ),
                    name=f'{atom_type} atom'
                ))
        
        # Update layout
        fig.update_layout(
            scene=dict(
                aspectmode='cube',
                camera=dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=1.5, y=1.5, z=1.5)
                ),
                xaxis=dict(range=[-0.6, 0.6]),
                yaxis=dict(range=[-0.6, 0.6]),
                zaxis=dict(range=[-0.6, 0.6]),
                xaxis_title='X (Å)',
                yaxis_title='Y (Å)',
                zaxis_title='Z (Å)'
            ),
            title='Molecular Structure Visualization',
            showlegend=True
        )
        
        fig.show()

# Example usage: Create a water molecule (H2O)
if __name__ == "__main__":
    try:
        # Create molecular structure
        molecule = MolecularStructure(resolution=64)
        
        # Add atoms for water molecule (H2O)
        # Scale all positions to fit within [-1, 1] cube
        # Oxygen at center
        molecule.add_atom((0.0, 0.0, 0.0), 'O')
        
        # Hydrogens at typical H2O bond angles (104.5°)
        bond_length = 1.0  # Will be scaled down by scale_factor in computation
        angle = 104.5 * np.pi / 180  # Convert to radians
        
        h1_x = bond_length * np.sin(angle/2)
        h1_y = bond_length * np.cos(angle/2)
        h2_x = -bond_length * np.sin(angle/2)
        h2_y = bond_length * np.cos(angle/2)
        
        molecule.add_atom((h1_x, h1_y, 0.0), 'H')
        molecule.add_atom((h2_x, h2_y, 0.0), 'H')
        
        # Compute electron density
        print("Computing electron density...")
        molecule.compute_electron_density()
        
        # Visualize the molecule
        print("Generating visualization...")
        molecule.visualize_molecule(density_levels=4, show_atoms=True)
        print("Visualization complete")
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        import traceback
        traceback.print_exc()=True)
        print("Visualization complete")
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        import traceback
        traceback.print_exc()
