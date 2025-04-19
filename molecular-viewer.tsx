import React, { useState, useEffect, useRef } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Activity } from 'lucide-react';

const MolecularViewer = () => {
  const canvasRef = useRef(null);
  const [selectedMolecule, setSelectedMolecule] = useState('ethanol');
  const [showBonds, setShowBonds] = useState(true);
  const [rotationSpeed, setRotationSpeed] = useState(0.01);
  const [rotation, setRotation] = useState({ x: 0, y: 0, z: 0 });

  // Parse PDB data
  const parsePDB = (pdbContent) => {
    const atoms = [];
    const bonds = [];
    
    pdbContent.split('\n').forEach(line => {
      if (line.startsWith('ATOM') || line.startsWith('HETATM')) {
        atoms.push({
          x: parseFloat(line.substring(30, 38)),
          y: parseFloat(line.substring(38, 46)),
          z: parseFloat(line.substring(46, 54)),
          element: line.substring(76, 78).trim(),
          serial: parseInt(line.substring(6, 11))
        });
      } else if (line.startsWith('CONECT')) {
        const values = line.trim().split(/\s+/);
        const from = parseInt(values[1]);
        for (let i = 2; i < values.length; i++) {
          const to = parseInt(values[i]);
          bonds.push([from, to]);
        }
      }
    });

    return { atoms, bonds };
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    // Project 3D to 2D
    const project = (x, y, z) => {
      // Apply rotation
      const rx = x * Math.cos(rotation.y) - z * Math.sin(rotation.y);
      const ry = y;
      const rz = x * Math.sin(rotation.y) + z * Math.cos(rotation.y);
      
      // Simple perspective projection
      const scale = 100;
      const distance = 500;
      const perspective = distance / (distance + rz);
      
      return {
        x: width/2 + rx * scale * perspective,
        y: height/2 + ry * scale * perspective,
        z: rz
      };
    };

    // Draw molecule
    const drawMolecule = (structure) => {
      ctx.clearRect(0, 0, width, height);

      // Draw bonds
      if (showBonds) {
        structure.bonds.forEach(([from, to]) => {
          const fromAtom = structure.atoms.find(a => a.serial === from);
          const toAtom = structure.atoms.find(a => a.serial === to);
          if (fromAtom && toAtom) {
            const p1 = project(fromAtom.x, fromAtom.y, fromAtom.z);
            const p2 = project(toAtom.x, toAtom.y, toAtom.z);
            ctx.beginPath();
            ctx.moveTo(p1.x, p1.y);
            ctx.lineTo(p2.x, p2.y);
            ctx.strokeStyle = 'rgba(100, 100, 100, 0.7)';
            ctx.stroke();
          }
        });
      }

      // Draw atoms
      structure.atoms.forEach(atom => {
        const p = project(atom.x, atom.y, atom.z);
        ctx.beginPath();
        ctx.arc(p.x, p.y, 10, 0, Math.PI * 2);
        
        // Color by element
        const colors = {
          'C': '#808080',
          'O': '#FF0000',
          'H': '#FFFFFF',
          'N': '#0000FF',
          'P': '#FFA500',
          'S': '#FFFF00'
        };
        
        ctx.fillStyle = colors[atom.element] || '#808080';
        ctx.strokeStyle = 'black';
        ctx.fill();
        ctx.stroke();
        
        // Draw element label
        ctx.fillStyle = 'black';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(atom.element, p.x, p.y);
      });
    };

    // Animation loop
    let animationFrame;
    const animate = () => {
      setRotation(prev => ({
        ...prev,
        y: prev.y + rotationSpeed
      }));
      animationFrame = requestAnimationFrame(animate);
    };
    animate();

    // Cleanup
    return () => {
      cancelAnimationFrame(animationFrame);
    };
  }, [rotation, showBonds, rotationSpeed]);

  return (
    <div className="p-4">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              <Activity className="w-6 h-6 mr-2" />
              Molecular Structure
            </CardTitle>
          </CardHeader>
          <CardContent>
            <canvas
              ref={canvasRef}
              width={400}
              height={400}
              className="border border-gray-300 rounded"
            />
            <div className="mt-4 space-y-2">
              <div className="flex items-center gap-2">
                <Button onClick={() => setShowBonds(!showBonds)}>
                  {showBonds ? 'Hide Bonds' : 'Show Bonds'}
                </Button>
                <div>
                  <label className="block text-sm font-medium">Rotation Speed</label>
                  <input
                    type="range"
                    min="0"
                    max="0.05"
                    step="0.001"
                    value={rotationSpeed}
                    onChange={(e) => setRotationSpeed(parseFloat(e.target.value))}
                    className="w-32"
                  />
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Molecule Properties</CardTitle>
          </CardHeader>
          <CardContent>
            <Tabs defaultValue="structure">
              <TabsList>
                <TabsTrigger value="structure">Structure</TabsTrigger>
                <TabsTrigger value="bonds">Bonds</TabsTrigger>
                <TabsTrigger value="properties">Properties</TabsTrigger>
              </TabsList>

              <TabsContent value="structure">
                <div className="space-y-2">
                  <p><strong>Atom Count:</strong> {selectedMolecule === 'ethanol' ? 9 : 8}</p>
                  <p><strong>Formula:</strong> {selectedMolecule === 'ethanol' ? 'C2H5OH' : 'C2H8'}</p>
                  <p><strong>Molecular Mass:</strong> {selectedMolecule === 'ethanol' ? '46.07 g/mol' : '30.07 g/mol'}</p>
                </div>
              </TabsContent>

              <TabsContent value="bonds">
                <div className="space-y-2">
                  <p><strong>Total Bonds:</strong> {selectedMolecule === 'ethanol' ? 8 : 7}</p>
                  <p><strong>Single Bonds:</strong> {selectedMolecule === 'ethanol' ? 8 : 7}</p>
                  <p><strong>Average Bond Length:</strong> 1.54 Å</p>
                </div>
              </TabsContent>

              <TabsContent value="properties">
                <div className="space-y-2">
                  <p><strong>Symmetry:</strong> C1</p>
                  <p><strong>Dipole Moment:</strong> {selectedMolecule === 'ethanol' ? '1.69 D' : '0 D'}</p>
                  <p><strong>Charge:</strong> Neutral</p>
                </div>
              </TabsContent>
            </Tabs>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default MolecularViewer;