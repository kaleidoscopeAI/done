#!/bin/bash

# Production Build Script for AI System
# Created: April 17, 2025

set -e  # Exit on any error

echo "===================== AI SYSTEM PRODUCTION BUILD ====================="
echo "Starting production build process..."

# Create build directories if they don't exist
mkdir -p ./build
mkdir -p ./dist

# Step 1: Build C components with optimized flags
echo "Building C components for production..."
make clean
make release  # Uses the release target defined in Makefile

# Step 2: Build the shared library
echo "Building shared library..."
make libkaleidoscope.so

# Step 3: Set up Python environment
echo "Setting up Python environment..."
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python -m venv .venv
fi
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Step 4: Build frontend components
echo "Building frontend components..."
if [ -d "./frontend" ]; then
    cd ./frontend
    
    # Check if npm is installed
    if ! command -v npm &> /dev/null; then
        echo "ERROR: npm is not installed. Please install Node.js and npm."
        exit 1
    fi
    
    # Install dependencies and build production version
    npm install
    npm run build
    
    # Copy build artifacts to the dist directory
    cp -r build/* ../dist/
    cd ..
else
    # If no dedicated frontend folder, look for individual TSX files
    echo "No frontend directory found. Checking for individual React components..."
    
    # Create a temporary frontend build directory
    mkdir -p ./temp_frontend
    
    # Find and copy all frontend-related files
    find . -name "*.tsx" -o -name "*.jsx" -o -name "*.js" -o -name "*.html" -o -name "*.css" | grep -v "node_modules" | while read file; do
        cp --parents "$file" ./temp_frontend/
    done
    
    # If we found frontend files, set up a minimal build system
    if [ -n "$(find ./temp_frontend -name '*.tsx' -o -name '*.jsx')" ]; then
        echo "Found React components. Setting up temporary build environment..."
        
        cd ./temp_frontend
        
        # Create package.json if it doesn't exist
        if [ ! -f "package.json" ]; then
            echo '{
                "name": "ai-system-frontend",
                "version": "1.0.0",
                "scripts": {
                    "build": "webpack --mode production"
                },
                "dependencies": {
                    "react": "^18.2.0",
                    "react-dom": "^18.2.0",
                    "recharts": "^2.7.2"
                },
                "devDependencies": {
                    "@babel/core": "^7.22.9",
                    "@babel/preset-env": "^7.22.9",
                    "@babel/preset-react": "^7.22.5",
                    "@babel/preset-typescript": "^7.22.5",
                    "@types/react": "^18.2.15",
                    "@types/react-dom": "^18.2.7",
                    "babel-loader": "^9.1.3",
                    "css-loader": "^6.8.1",
                    "html-webpack-plugin": "^5.5.3",
                    "style-loader": "^3.3.3",
                    "typescript": "^5.1.6",
                    "webpack": "^5.88.1",
                    "webpack-cli": "^5.1.4"
                }
            }' > package.json
        fi
        
        # Create webpack.config.js if it doesn't exist
        if [ ! -f "webpack.config.js" ]; then
            echo "const path = require('path');
            const HtmlWebpackPlugin = require('html-webpack-plugin');

            module.exports = {
              entry: './molecular-quantum-system.tsx',
              output: {
                filename: 'bundle.js',
                path: path.resolve(__dirname, 'dist'),
              },
              resolve: {
                extensions: ['.tsx', '.ts', '.js', '.jsx'],
              },
              module: {
                rules: [
                  {
                    test: /\.(ts|tsx|js|jsx)$/,
                    exclude: /node_modules/,
                    use: {
                      loader: 'babel-loader',
                      options: {
                        presets: [
                          '@babel/preset-env',
                          '@babel/preset-react',
                          '@babel/preset-typescript'
                        ]
                      }
                    }
                  },
                  {
                    test: /\.css$/,
                    use: ['style-loader', 'css-loader'],
                  }
                ]
              },
              plugins: [
                new HtmlWebpackPlugin({
                  template: 'index.html',
                  filename: 'index.html',
                })
              ]
            };" > webpack.config.js
        fi
        
        # Create index.html if it doesn't exist
        if [ ! -f "index.html" ]; then
            echo '<!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>AI System Quantum Visualization</title>
            </head>
            <body>
                <div id="root"></div>
            </body>
            </html>' > index.html
        fi
        
        # Create a basic entry point if it doesn't exist
        if [ ! -f "molecular-quantum-system.tsx" ]; then
            echo "import React from 'react';
            import ReactDOM from 'react-dom/client';
            import QuantumMolecularVisualization from './molecular-quantum-vis';

            const root = ReactDOM.createRoot(document.getElementById('root'));
            root.render(
              <React.StrictMode>
                <QuantumMolecularVisualization />
              </React.StrictMode>
            );" > molecular-quantum-system.tsx
        fi
        
        # Install dependencies and build
        npm install
        npm run build
        
        # Copy build output to main dist directory
        cp -r dist/* ../../dist/
        
        cd ..
        # Clean up temporary directory
        rm -rf ./temp_frontend
    else
        echo "No frontend components found to build."
    fi
fi

# Step 5: Combine backend and frontend for deployment
echo "Preparing final deployment package..."

# Create the production directory structure
mkdir -p ./production
mkdir -p ./production/bin
mkdir -p ./production/lib
mkdir -p ./production/web
mkdir -p ./production/config

# Copy C binary and library
cp ./ai_system ./production/bin/
cp ./libkaleidoscope.so ./production/lib/

# Copy Python files
find . -name "*.py" | grep -v "__pycache__" | grep -v ".venv" | while read file; do
    cp --parents "$file" ./production/
done

# Copy frontend files
cp -r ./dist/* ./production/web/

# Copy configuration files
cp -r ./config.py ./production/config/
if [ -f ".env.production" ]; then
    cp .env.production ./production/.env
fi

# Create production launcher script
cat > ./production/start_system.sh << 'EOF'
#!/bin/bash
export LD_LIBRARY_PATH=./lib:$LD_LIBRARY_PATH
cd "$(dirname "$0")"
source .venv/bin/activate
python main.py --production
EOF
chmod +x ./production/start_system.sh

# Create a simple README
cat > ./production/README.md << 'EOF'
# AI System - Production Build

This is a production build of the AI System created on April 17, 2025.

## Directory Structure
- bin/ - Contains compiled C binaries
- lib/ - Contains shared libraries
- web/ - Contains frontend web components
- config/ - Contains configuration files

## Getting Started
1. Ensure you have Python 3.9+ installed
2. Run `python -m venv .venv` to create a virtual environment
3. Run `source .venv/bin/activate` to activate the environment
4. Run `pip install -r requirements.txt` to install dependencies
5. Run `./start_system.sh` to start the system

## Troubleshooting
- If you encounter any issues, check the logs in the logs/ directory
- Ensure all dependencies are installed correctly
- Verify that the system has sufficient permissions
EOF

# Create a requirements.txt file if it doesn't exist
if [ ! -f "./production/requirements.txt" ]; then
    pip freeze > ./production/requirements.txt
fi

echo "Creating production archive..."
tar -czf ai_system_production.tar.gz ./production

echo "Production build complete! Archive created: ai_system_production.tar.gz"
echo "=================================================================="
echo "To deploy, extract the archive on the target system and run start_system.sh"
echo "=================================================================="