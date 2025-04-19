import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';
import { Terminal } from 'lucide-react';

const Dashboard = () => {
  const [systemMetrics, setSystemMetrics] = useState([]);
  
  useEffect(() => {
    // Simulated metrics data
    const data = [
      {timestamp: '00:00', processing: 85, memory: 65, efficiency: 90},
      {timestamp: '00:05', processing: 88, memory: 70, efficiency: 87},
      {timestamp: '00:10', processing: 92, memory: 75, efficiency: 85},
      {timestamp: '00:15', processing: 90, memory: 72, efficiency: 88},
      {timestamp: '00:20', processing: 87, memory: 68, efficiency: 91}
    ];
    setSystemMetrics(data);
  }, []);

  return (
    <div className="p-6 bg-gray-100 min-h-screen">
      {/* Header */}
      <div className="mb-8 flex justify-between items-center">
        <h1 className="text-2xl font-bold text-gray-800">Quantum System Dashboard</h1>
        <div className="flex items-center space-x-4">
          <div className="px-4 py-2 bg-green-500 text-white rounded">System Active</div>
          <Terminal className="w-6 h-6 text-gray-600" />
        </div>
      </div>

      {/* Main Grid */}
      <div className="grid grid-cols-2 gap-6">
        {/* Metrics Chart */}
        <div className="bg-white p-6 rounded-lg shadow-lg">
          <h2 className="text-xl font-semibold mb-4">System Performance</h2>
          <LineChart width={600} height={300} data={systemMetrics}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="timestamp" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey="processing" stroke="#8884d8" />
            <Line type="monotone" dataKey="memory" stroke="#82ca9d" />
            <Line type="monotone" dataKey="efficiency" stroke="#ffc658" />
          </LineChart>
        </div>

        {/* Status Panel */}
        <div className="bg-white p-6 rounded-lg shadow-lg">
          <h2 className="text-xl font-semibold mb-4">System Status</h2>
          <div className="space-y-4">
            <div className="flex justify-between items-center p-3 bg-gray-50 rounded">
              <span>Processing Nodes</span>
              <span className="px-3 py-1 bg-green-100 text-green-800 rounded">Active</span>
            </div>
            <div className="flex justify-between items-center p-3 bg-gray-50 rounded">
              <span>Memory Network</span>
              <span className="px-3 py-1 bg-green-100 text-green-800 rounded">Optimal</span>
            </div>
            <div className="flex justify-between items-center p-3 bg-gray-50 rounded">
              <span>Quantum Optimization</span>
              <span className="px-3 py-1 bg-blue-100 text-blue-800 rounded">Running</span>
            </div>
          </div>
        </div>

        {/* Control Panel */}
        <div className="bg-white p-6 rounded-lg shadow-lg">
          <h2 className="text-xl font-semibold mb-4">System Controls</h2>
          <div className="grid grid-cols-2 gap-4">
            <button className="p-3 bg-blue-500 text-white rounded hover:bg-blue-600">
              Optimize System
            </button>
            <button className="p-3 bg-gray-500 text-white rounded hover:bg-gray-600">
              Reset Parameters
            </button>
            <button className="p-3 bg-green-500 text-white rounded hover:bg-green-600">
              Start Processing
            </button>
            <button className="p-3 bg-red-500 text-white rounded hover:bg-red-600">
              Emergency Stop
            </button>
          </div>
        </div>

        {/* Console Output */}
        <div className="bg-black p-6 rounded-lg shadow-lg">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-xl font-semibold text-white">System Console</h2>
            <Terminal className="w-6 h-6 text-white" />
          </div>
          <div className="font-mono text-green-500 text-sm">
            <p>> System initialized</p>
            <p>> Processing nodes: OK</p>
            <p>> Memory network: OPTIMAL</p>
            <p>> Quantum optimization: RUNNING</p>
            <p>> _</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
