import React, { useState, useEffect, useRef } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Activity, Upload } from 'lucide-react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Slider } from '@/components/ui/slider';
import { Badge } from '@/components/ui/badge';

const MolecularAnalysisApp = () => {
  // Basic state
  const [molecularData, setMolecularData] = useState(null);
  const [error, setError] = useState(null);
  const canvasRef = useRef(null);

  // Visualization controls
  const [showBonds, setShowBonds] = useState(true);
  const [rotationSpeed, setRotationSpeed] = useState([0.01]);
  const [rotation, setRotation] = useState({ x: 0, y: 0, z: 0 });
  const [zoom, setZoom] = useState([1]);

  // Drug discovery analysis state
  const [drugProperties, setDrugProperties] = useState(null);

  // Handle file upload
  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    try {
      const content = await file.text();
      const structure = parsePDBStructure(content);
      const properties = analyzeMolecule(structure);
      
      setMolecularData({ structure, properties });
      analyzeDrugLikeProperties(structure);
    } catch (err) {
      setError(`Error processing file: ${err.message}`);
    }
  };

  // Parse PDB structure
  const parsePDBStructure = (content) => {
    const atoms = [];
    const bonds = [];
    
    content.split('\n').forEach(line => {
      if (line.startsWith('ATOM') || line.startsWith('HETATM')) {
        atoms.push({
          serial: parseInt(line.substring(6, 11)),
          element: line.substring(76, 78).trim() || 'C',
          x: parseFloat(line.substring(30, 38)),
          y: parseFloat(line.substring(38, 46)),
          z: parseFloat(line.substring(46, 54))
        });
      } else if (line.startsWith('CONECT')) {
        const values = line.trim().split(/\s+/);
        const fromAtom = parseInt(values[1]);
        for (let i = 2; i < values.length; i++) {
          bonds.push({
            from: fromAtom,
            to: parseInt(values[i])
          });
        }
      }
    });

    return { atoms, bonds };
  };

  // Basic molecular analysis
  const analyzeMolecule = (structure) => {
    const { atoms, bonds } = structure;
    
    // Count elements
    const elementCounts = atoms.reduce((counts, atom) => {
      counts[atom.element] = (counts[atom.element] || 0) + 1;
      return counts;
    }, {});

    // Calculate molecular mass
    const atomicMasses = {
      'H': 1.008, 'C': 12.011, 'N': 14.007, 'O': 15.999,
      'P': 30.974, 'S': 32.065, 'F': 18.998, 'Cl': 35.45
    };

    const molecularMass = atoms.reduce((mass, atom) => 
      mass + (atomicMasses[atom.element] || 0), 0);

    return {
      elementCounts,
      molecularMass: molecularMass.toFixed(3),
      atomCount: atoms.length,
      bondCount: bonds.length
    };
  };

  // Drug-like property analysis
  const analyzeDrugLikeProperties = (structure) => {
    const { atoms } = structure;
    const properties = analyzeMolecule(structure);

    // Lipinski's Rule of Five analysis
    const lipinski = {
      molecularWeight: parseFloat(properties.molecularMass),
      hBondDonors: countHBondDonors(atoms),
      hBondAcceptors: countHBondAcceptors(atoms),
      logP: estimateLogP(atoms),
      violations: 0
    };

    // Count Lipinski violations
    if (lipinski.molecularWeight > 500) lipinski.violations++;
    if (lipinski.hBondDonors > 5) lipinski.violations++;
    if (lipinski.hBondAcceptors > 10) lipinski.violations++;
    if (lipinski.logP > 5) lipinski.violations++;

    setDrugProperties(lipinski);
  };

  // Helper functions for drug analysis
  const countHBondDonors = (atoms) => 
    atoms.filter(a => ['O', 'N'].includes(a.element)).length;

  const countHBondAcceptors = (atoms) => 
    atoms.filter(a => ['O', 'N'].includes(a.element)).length;

  const estimateLogP = (atoms) => {
    // Simplified LogP estimation
    const elementContributions = {
      'C': 0.5, 'O': -0.5, 'N': -0.5, 'S': 0.3,
      'H': 0.1, 'Cl': 0.8, 'F': 0.5
    };

    return atoms.reduce((logP, atom) => 
      logP + (elementContributions[atom.element] || 0), 0);
  };

  // 3D Visualization
  useEffect(() => {
    if (!molecularData || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    const project = (x, y, z) => {
      const scale = 100 * zoom[0];
      const distance = 500;
      
      // Apply rotation
      const rx = x * Math.cos(rotation.y) - z * Math.sin(rotation.y);
      const ry = y;
      const rz = x * Math.sin(rotation.y) + z * Math.cos(rotation.y);
      
      // Apply perspective
      const perspective = distance / (distance + rz);
      
      return {
        x: width/2 + rx * scale * perspective,
        y: height/2 + ry * scale * perspective,
        z: rz
      };
    };

    const drawMolecule = () => {
      ctx.clearRect(0, 0, width, height);
      const { atoms, bonds } = molecularData.structure;

      // Draw bonds
      if (showBonds) {
        bonds.forEach(bond => {
          const fromAtom = atoms.find(a => a.serial === bond.from);
          const toAtom = atoms.find(a => a.serial === bond.to);
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
      atoms.forEach(atom => {
        const p = project(atom.x, atom.y, atom.z);
        ctx.beginPath();
        ctx.arc(p.x, p.y, 10, 0, Math.PI * 2);
        
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
        
        ctx.fillStyle = 'black';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(atom.element, p.x, p.y);
      });
    };

    let animationFrame;
    const animate = () => {
      setRotation(prev => ({
        ...prev,
        y: prev.y + rotationSpeed[0]
      }));
      drawMolecule();
      animationFrame = requestAnimationFrame(animate);
    };
    animate();

    return () => cancelAnimationFrame(animationFrame);
  }, [molecularData, rotation, showBonds, zoom, rotationSpeed]);

  return (
    <div className="p-4 space-y-4">
      <Card>
        <CardHeader>
          <CardTitle>Drug Discovery Analysis</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {/* File Upload */}
            <div>
              <Button onClick={() => document.getElementById('file-upload').click()}>
                <Upload className="w-4 h-4 mr-2" />
                Upload Structure
              </Button>
              <input
                id="file-upload"
                type="file"
                accept=".pdb,.mol,.mol2,.sdf"
                className="hidden"
                onChange={handleFileUpload}
              />
            </div>

            {error && (
              <Alert variant="destructive">
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}

            {/* Molecule Viewer and Analysis */}
            {molecularData && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* 3D Viewer */}
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center">
                      <Activity className="w-6 h-6 mr-2" />
                      Structure View
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
                      <Button 
                        onClick={() => setShowBonds(!showBonds)}
                        size="sm"
                      >
                        {showBonds ? 'Hide Bonds' : 'Show Bonds'}
                      </Button>
                      <div className="space-y-1">
                        <label className="text-sm font-medium">Rotation Speed</label>
                        <Slider
                          defaultValue={[0.01]}
                          value={rotationSpeed}
                          onValueChange={setRotationSpeed}
                          min={0}
                          max={0.05}
                          step={0.001}
                        />
                      </div>
                      <div className="space-y-1">
                        <label className="text-sm font-medium">Zoom</label>
                        <Slider
                          defaultValue={[1]}
                          value={zoom}
                          onValueChange={setZoom}
                          min={0.5}
                          max={2}
                          step={0.1}
                        />
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {/* Analysis Results */}
                <Card>
                  <CardHeader>
                    <CardTitle>Analysis Results</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <Tabs defaultValue="properties">
                      <TabsList>
                        <TabsTrigger value="properties">Properties</TabsTrigger>
                        <TabsTrigger value="druglike">Drug-like</TabsTrigger>
                      </TabsList>

                      <TabsContent value="properties" className="space-y-2">
                        <p><strong>Molecular Mass:</strong> {molecularData.properties.molecularMass} g/mol</p>
                        <p><strong>Atoms:</strong> {molecularData.properties.atomCount}</p>
                        <p><strong>Bonds:</strong> {molecularData.properties.bondCount}</p>
                        <div>
                          <p className="font-medium">Element Composition:</p>
                          <div className="flex flex-wrap gap-2 mt-1">
                            {Object.entries(molecularData.properties.elementCounts).map(([element, count]) => (
                              <Badge key={element} variant="secondary">
                                {element}: {count}
                              </Badge>
                            ))}
                          </div>
                        </div>
                      </TabsContent>

                      <TabsContent value="druglike" className="space-y-2">
                        {drugProperties && (
                          <>
                            <p><strong>Molecular Weight:</strong> {drugProperties.molecularWeight.toFixed(1)}</p>
                            <p><strong>LogP (est.):</strong> {drugProperties.logP.toFixed(2)}</p>
                            <p><strong>H-Bond Donors:</strong> {drugProperties.hBondDonors}</p>
                            <p><strong>H-Bond Acceptors:</strong> {drugProperties.hBondAcceptors}</p>
                            <p>
                              <strong>Lipinski Violations:</strong>{' '}
                              <Badge variant={drugProperties.violations <= 1 ? 'success' : 'destructive'}>
                                {drugProperties.violations}
                              </Badge>
                            </p>
                          </>
                        )}
                      </TabsContent>
                    </Tabs>
                  </CardContent>
                </Card>
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default MolecularAnalysisApp;