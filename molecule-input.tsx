import React, { useState, useCallback } from 'react';
import { Upload, X, Search, AlertCircle } from 'lucide-react';
import { Alert, AlertDescription } from '@/components/ui/alert';
import Papa from 'papaparse';

const MoleculeInput = ({ onAnalyze }) => {
  const [smiles, setSmiles] = useState('');
  const [file, setFile] = useState(null);
  const [error, setError] = useState(null);
  const [targets, setTargets] = useState([]);
  const [loading, setLoading] = useState(false);

  const handleFileUpload = useCallback(async (event) => {
    const uploadedFile = event.target.files[0];
    if (!uploadedFile) return;

    try {
      setLoading(true);
      setError(null);

      const fileContent = await uploadedFile.text();
      Papa.parse(fileContent, {
        header: true,
        dynamicTyping: true,
        complete: (results) => {
          if (results.data && results.data.length > 0) {
            if (!results.data[0].SMILES) {
              setError('File must contain a SMILES column');
              return;
            }
            setFile(results.data);
            if (results.data[0].Target) {
              setTargets(Array.from(new Set(results.data.map(row => row.Target).filter(Boolean))));
            }
          }
        },
        error: (error) => {
          setError(`Failed to parse file: ${error.message}`);
        }
      });
    } catch (err) {
      setError(`Failed to read file: ${err.message}`);
    } finally {
      setLoading(false);
    }
  }, []);

  const handleSubmit = async () => {
    try {
      setLoading(true);
      setError(null);

      if (file) {
        // Batch analysis
        const results = await Promise.all(
          file.map(row => 
            fetch('/api/analyze', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                smiles: row.SMILES,
                targets: row.Target ? [row.Target] : undefined
              })
            }).then(res => res.json())
          )
        );
        onAnalyze(results);
      } else if (smiles) {
        // Single molecule analysis
        const response = await fetch('/api/analyze', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ smiles, targets })
        });
        const result = await response.json();
        onAnalyze([result]);
      } else {
        setError('Please input a SMILES string or upload a file');
      }
    } catch (err) {
      setError(`Analysis failed: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-4 p-4 border rounded-lg bg-white">
      {/* SMILES Input */}
      <div>
        <label className="block text-sm font-medium mb-1">
          SMILES String
        </label>
        <div className="flex gap-2">
          <input
            type="text"
            className="flex-1 px-3 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            value={smiles}
            onChange={(e) => setSmiles(e.target.value)}
            placeholder="Enter SMILES string..."
            disabled={loading || !!file}
          />
          {smiles && (
            <button
              className="p-2 text-gray-400 hover:text-gray-600"
              onClick={() => setSmiles('')}
            >
              <X className="h-5 w-5" />
            </button>
          )}
        </div>
      </div>

      {/* File Upload */}
      <div>
        <label className="block text-sm font-medium mb-1">
          Or Upload File
        </label>
        <div className="flex items-center gap-2">
          <label className="flex-1 cursor-pointer">
            <div className="px-4 py-2 border rounded-lg text-center hover:bg-gray-50">
              <Upload className="h-5 w-5 inline-block mr-2" />
              {file ? file[0].name : 'Choose CSV file'}
            </div>
            <input
              type="file"
              className="hidden"
              accept=".csv"
              onChange={handleFileUpload}
              disabled={loading || !!smiles}
            />
          </label>
          {file && (
            <button
              className="p-2 text-gray-400 hover:text-gray-600"
              onClick={() => setFile(null)}
            >
              <X className="h-5 w-5" />
            </button>
          )}
        </div>
      </div>

      {/* Target Selection */}
      {targets.length > 0 && (
        <div>
          <label className="block text-sm font-medium mb-1">
            Targets
          </label>
          <div className="flex flex-wrap gap-2">
            {targets.map(target => (
              <div
                key={target}
                className="px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-sm"
              >
                {target}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Error Display */}
      {error && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {/* Submit Button */}
      <button
        className={`w-full py-2 px-4 rounded-lg text-white font-medium 
          ${loading ? 'bg-gray-400' : 'bg-blue-500 hover:bg-blue-600'}
          transition duration-200`}
        onClick={handleSubmit}
        disabled={loading || (!smiles && !file)}
      >
        {loading ? (
          <div className="flex items-center justify-center">
            <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-2" />
            Analyzing...
          </div>
        ) : (
          <div className="flex items-center justify-center">
            <Search className="h-5 w-5 mr-2" />
            Analyze
          </div>
        )}
      </button>
    </div>
  );
};

export default MoleculeInput;
