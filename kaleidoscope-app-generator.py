#!/usr/bin/env python3
"""
Kaleidoscope AI - Application Generator
=======================================
Transforms natural language app descriptions into full-featured applications.
Integrates with the core Kaleidoscope AI architecture to generate complete,
production-ready codebases from high-level descriptions.

Usage:
    python app_generator.py --description "Create a web application for tracking inventory"
    python app_generator.py --description-file app_description.txt --output-dir ./generated_app
"""

import os
import sys
import json
import logging
import asyncio
import re
import shutil
import subprocess
import argparse
import hashlib
import time
import platform
import traceback
import venv
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor

# Set up virtual environment if it doesn't exist
def setup_virtual_environment():
    """Setup and activate a virtual environment for the application."""
    venv_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "venv")
    
    # Create virtual environment if it doesn't exist
    if not os.path.exists(venv_dir):
        print(f"Creating virtual environment in {venv_dir}...")
        venv.create(venv_dir, with_pip=True)
    
    # Determine the path to the activation script
    if platform.system() == "Windows":
        activate_script = os.path.join(venv_dir, "Scripts", "activate.bat")
        activate_cmd = f'"{activate_script}"'
    else:
        activate_script = os.path.join(venv_dir, "bin", "activate")
        activate_cmd = f'source "{activate_script}"'
    
    # Check if we're already in the virtual environment
    if sys.prefix == sys.base_prefix:
        print(f"To activate the virtual environment, run: {activate_cmd}")
        print("After activation, install dependencies with: pip install -r requirements.txt")
        print("Then run this script again.")
        
        # Create requirements.txt if it doesn't exist
        req_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements.txt")
        if not os.path.exists(req_file):
            with open(req_file, "w") as f:
                f.write("aiohttp==3.8.4\n")
                f.write("openai==0.27.8\n")
                f.write("anthropic==0.3.0\n")
                f.write("jinja2==3.1.2\n")
                f.write("pyyaml==6.0\n")
                f.write("termcolor==2.3.0\n")
        
        sys.exit(0)

# Check if we're in a virtual environment
if sys.prefix == sys.base_prefix:
    setup_virtual_environment()

# Now that we're in the virtual env, import the rest of our dependencies
try:
    import aiohttp
    import jinja2
    import yaml
    from termcolor import colored
except ImportError:
    print("Required packages not found. Please activate the virtual environment and install dependencies:")
    print("pip install -r requirements.txt")
    sys.exit(1)

# Import Kaleidoscope components
try:
    # Add parent directory to path for importing the Kaleidoscope modules
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from kaleidoscope_core.error_handling import ErrorManager, ErrorCategory, ErrorSeverity, EnhancedError, ErrorContext
    from kaleidoscope_core.code_reusability import UnifiedAST, LanguageAdapterRegistry, MimicryPipeline
    from kaleidoscope_core.llm_integration import LLMIntegration, LLMProvider, OpenAIProvider, AnthropicProvider
except ImportError:
    print("Kaleidoscope core modules not found. Using standalone mode.")
    
    # Define minimal implementations of required classes for standalone operation
    class ErrorSeverity:
        DEBUG = "DEBUG"
        INFO = "INFO"
        WARNING = "WARNING"
        ERROR = "ERROR"
        CRITICAL = "CRITICAL"
        FATAL = "FATAL"
    
    class ErrorCategory:
        SYSTEM = "SYSTEM"
        NETWORK = "NETWORK"
        API = "API"
        PARSING = "PARSING"
        ANALYSIS = "ANALYSIS"
        DECOMPILATION = "DECOMPILATION"
        SPECIFICATION = "SPECIFICATION"
        RECONSTRUCTION = "RECONSTRUCTION"
        MIMICRY = "MIMICRY"
        LLM = "LLM"
        SECURITY = "SECURITY"
        RESOURCE = "RESOURCE"
        VALIDATION = "VALIDATION"
        RECOVERY = "RECOVERY"
        UNKNOWN = "UNKNOWN"
    
    class ErrorManager:
        """Simplified error manager for standalone operation."""
        
        def __init__(self):
            self.errors = []
        
        def handle_exception(self, exception, category=None, severity=None, operation=None, **context_kwargs):
            error = {
                "exception": exception,
                "category": category or ErrorCategory.UNKNOWN,
                "severity": severity or ErrorSeverity.ERROR,
                "operation": operation,
                "context": context_kwargs,
                "timestamp": time.time()
            }
            self.errors.append(error)
            logging.error(f"Error in {operation}: {str(exception)}")
            return error
    
    class LLMIntegration:
        """Simplified LLM integration for standalone operation."""
        
        def __init__(self, providers=None):
            self.providers = providers or {}
        
        async def generate_completion(self, prompt, model=None, provider=None, max_tokens=1000, temperature=0.7):
            """
            Generate completion using the specified provider.
            Falls back to OpenAI's GPT-3.5 API with a warning.
            """
            import openai
            logging.warning("Using OpenAI API fallback in standalone mode")
            
            try:
                response = await openai.ChatCompletion.acreate(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                return response.choices[0].message.content
            except Exception as e:
                logging.error(f"Error using OpenAI API: {str(e)}")
                return "Error generating completion. Please check your OpenAI API key."

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("app_generator.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# LLM Provider Configuration
LLM_CONFIG = {
    "openai": {
        "api_key": os.environ.get("OPENAI_API_KEY"),
        "models": {
            "default": "gpt-3.5-turbo",
            "advanced": "gpt-4"
        }
    },
    "anthropic": {
        "api_key": os.environ.get("ANTHROPIC_API_KEY"),
        "models": {
            "default": "claude-instant-1",
            "advanced": "claude-2"
        }
    },
    "local": {
        "url": os.environ.get("LOCAL_LLM_URL", "http://localhost:11434"),
        "models": {
            "default": "llama2",
            "advanced": "llama2-70b"
        }
    }
}

@dataclass
class AppComponent:
    """Represents a component of the application"""
    name: str
    type: str
    description: str
    dependencies: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)
    files: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "type": self.type,
            "description": self.description,
            "dependencies": self.dependencies,
            "properties": self.properties,
            "files": self.files
        }

@dataclass
class AppArchitecture:
    """Represents the architecture of an application"""
    name: str
    description: str
    type: str
    language: str
    framework: str
    components: List[AppComponent] = field(default_factory=list)
    dependencies: Dict[str, str] = field(default_factory=dict)
    database: Optional[str] = None
    apis: List[Dict[str, Any]] = field(default_factory=list)
    deployment: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "description": self.description,
            "type": self.type,
            "language": self.language,
            "framework": self.framework,
            "components": [c.to_dict() for c in self.components],
            "dependencies": self.dependencies,
            "database": self.database,
            "apis": self.apis,
            "deployment": self.deployment
        }
    
    def add_component(self, component: AppComponent) -> 'AppArchitecture':
        """Add a component to the architecture"""
        self.components.append(component)
        return self

class AppDescriptionAnalyzer:
    """Analyzes app descriptions to extract requirements and architecture"""
    
    def __init__(self, llm_integration: LLMIntegration):
        """
        Initialize the analyzer
        
        Args:
            llm_integration: LLM integration for analysis
        """
        self.llm = llm_integration
        self.error_manager = ErrorManager()
    
    async def analyze_description(self, description: str) -> AppArchitecture:
        """
        Analyze an app description
        
        Args:
            description: App description
            
        Returns:
            Extracted app architecture
        """
        logger.info("Analyzing app description")
        
        try:
            # Step 1: Extract basic app properties
            app_props = await self._extract_app_properties(description)
            
            # Step 2: Determine technology stack
            tech_stack = await self._determine_tech_stack(description, app_props)
            
            # Step 3: Identify core components
            components = await self._identify_components(description, app_props, tech_stack)
            
            # Step 4: Build architecture model
            architecture = AppArchitecture(
                name=app_props.get("name", "GeneratedApp"),
                description=app_props.get("description", description),
                type=app_props.get("type", "web"),
                language=tech_stack.get("language", "python"),
                framework=tech_stack.get("framework", ""),
                database=tech_stack.get("database"),
                dependencies=tech_stack.get("dependencies", {}),
                deployment=tech_stack.get("deployment", {})
            )
            
            # Add components
            for comp in components:
                architecture.add_component(AppComponent(
                    name=comp["name"],
                    type=comp["type"],
                    description=comp["description"],
                    dependencies=comp.get("dependencies", []),
                    properties=comp.get("properties", {})
                ))
            
            return architecture
        except Exception as e:
            error_ctx = ErrorContext(
                operation="analyze_description",
                additional_info={"description_length": len(description)}
            )
            self.error_manager.handle_exception(
                e,
                category=ErrorCategory.ANALYSIS,
                severity=ErrorSeverity.ERROR,
                operation="analyze_app_description",
                error_context=error_ctx
            )
            # Provide a fallback architecture
            return self._create_fallback_architecture(description)
    
    def _create_fallback_architecture(self, description: str) -> AppArchitecture:
        """Create a fallback architecture when analysis fails"""
        # Extract a simple name from the description
        words = re.findall(r'\b[A-Za-z][a-z]+\b', description)
        if len(words) >= 2:
            name = f"{words[0].capitalize()}{words[1].capitalize()}"
        else:
            name = "FallbackApp"
        
        # Create a basic web app architecture
        architecture = AppArchitecture(
            name=name,
            description=description[:100] + "..." if len(description) > 100 else description,
            type="web",
            language="python",
            framework="flask",
            database="sqlite"
        )
        
        # Add basic components
        architecture.add_component(AppComponent(
            name="app",
            type="application",
            description="Main application module"
        ))
        
        architecture.add_component(AppComponent(
            name="models",
            type="data",
            description="Data models"
        ))
        
        architecture.add_component(AppComponent(
            name="views",
            type="interface",
            description="User interface views"
        ))
        
        return architecture
    
    async def _extract_app_properties(self, description: str) -> Dict[str, Any]:
        """
        Extract basic app properties from description
        
        Args:
            description: App description
            
        Returns:
            Dictionary of app properties
        """
        prompt = f"""
Analyze the following app description and extract key properties.
Return a JSON object with the following fields:
- name: A suitable name for the app
- description: A concise description of the app
- type: The app type (web, mobile, desktop, cli, api, etc.)
- features: List of main features
- target_users: Target user base
- complexity: Estimated complexity (simple, moderate, complex)

App Description:
{description}
"""
        try:
            response = await self.llm.generate_completion(prompt)
            properties = json.loads(response)
            logger.info("Extracted app properties")
            return properties
        except Exception as e:
            error = self.error_manager.handle_exception(
                e, 
                category=ErrorCategory.ANALYSIS,
                severity=ErrorSeverity.ERROR,
                operation="extract_app_properties"
            )
            logger.error(f"Failed to extract app properties: {str(e)}")
            return {
                "name": self._extract_fallback_name(description),
                "description": description[:100] + "..." if len(description) > 100 else description,
                "type": "web",
                "features": [],
                "target_users": "general",
                "complexity": "moderate"
            }
    
    def _extract_fallback_name(self, description: str) -> str:
        """Extract a fallback name from the description when LLM fails"""
        # Look for capitalized words that might be good app names
        capitalized = re.findall(r'\b[A-Z][a-zA-Z]+\b', description)
        if capitalized:
            return "".join(capitalized[:2])
        
        # Otherwise take the first few words
        words = description.split()
        if len(words) >= 2:
            return "".join([w.capitalize() for w in words[:2]])
        
        # Last resort
        return "GeneratedApp"
    
    async def _determine_tech_stack(self, description: str, properties: Dict[str, Any]) -> Dict[str, Any]:
        """
        Determine appropriate technology stack
        
        Args:
            description: App description
            properties: App properties
            
        Returns:
            Dictionary of technology stack details
        """
        prompt = f"""
Based on the following app description and properties, recommend an appropriate technology stack.
Return a JSON object with the following fields:
- language: Primary programming language
- framework: Main framework to use
- database: Database technology (if needed)
- frontend: Frontend framework/library (if applicable)
- backend: Backend technology (if applicable)
- mobile: Mobile development approach (if applicable)
- dependencies: Key libraries and dependencies
- deployment: Recommended deployment approach

App Description:
{description}

App Properties:
{json.dumps(properties, indent=2)}
"""
        try:
            response = await self.llm.generate_completion(prompt)
            tech_stack = json.loads(response)
            logger.info(f"Determined tech stack: {tech_stack['language']}/{tech_stack['framework']}")
            return tech_stack
        except Exception as e:
            error = self.error_manager.handle_exception(
                e, 
                category=ErrorCategory.ANALYSIS,
                severity=ErrorSeverity.ERROR,
                operation="determine_tech_stack"
            )
            logger.error(f"Failed to determine tech stack: {str(e)}")
            
            # Provide a sensible default based on app type
            app_type = properties.get("type", "web").lower()
            if app_type == "web":
                return {
                    "language": "python",
                    "framework": "flask",
                    "database": "sqlite",
                    "frontend": "bootstrap",
                    "backend": "rest",
                    "dependencies": {},
                    "deployment": {"type": "docker"}
                }
            elif app_type == "mobile":
                return {
                    "language": "javascript",
                    "framework": "react-native",
                    "database": "sqlite",
                    "dependencies": {},
                    "deployment": {"type": "app-store"}
                }
            elif app_type == "desktop":
                return {
                    "language": "python",
                    "framework": "pyqt",
                    "database": "sqlite",
                    "dependencies": {},
                    "deployment": {"type": "executable"}
                }
            else:
                return {
                    "language": "python",
                    "framework": "flask",
                    "database": "sqlite",
                    "dependencies": {},
                    "deployment": {"type": "docker"}
                }
    
    async def _identify_components(
        self, 
        description: str, 
        properties: Dict[str, Any], 
        tech_stack: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Identify core components of the app
        
        Args:
            description: App description
            properties: App properties
            tech_stack: Technology stack
            
        Returns:
            List of component dictionaries
        """
        prompt = f"""
Identify the core components needed for this application.
Return a JSON array of component objects, each with:
- name: Component name
- type: Component type (model, view, controller, service, utility, etc.)
- description: What the component does
- dependencies: List of other components this depends on
- properties: Additional component-specific properties

App Description:
{description}

App Properties:
{json.dumps(properties, indent=2)}

Technology Stack:
{json.dumps(tech_stack, indent=2)}
"""
        try:
            response = await self.llm.generate_completion(prompt)
            components = json.loads(response)
            logger.info(f"Identified {len(components)} components")
            return components
        except Exception as e:
            error = self.error_manager.handle_exception(
                e, 
                category=ErrorCategory.ANALYSIS,
                severity=ErrorSeverity.ERROR,
                operation="identify_components"
            )
            logger.error(f"Failed to identify components: {str(e)}")
            
            # Generate basic components based on tech stack
            return self._generate_basic_components(tech_stack)
    
    def _generate_basic_components(self, tech_stack: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate basic components based on technology stack"""
        language = tech_stack.get("language", "").lower()
        framework = tech_stack.get("framework", "").lower()
        
        if language == "python":
            if framework == "flask":
                return [
                    {
                        "name": "app",
                        "type": "application",
                        "description": "Main Flask application",
                        "dependencies": []
                    },
                    {
                        "name": "models",
                        "type": "model",
                        "description": "Data models",
                        "dependencies": []
                    },
                    {
                        "name": "views",
                        "type": "view",
                        "description": "Web views and routes",
                        "dependencies": ["models"]
                    }
                ]
            elif framework == "django":
                return [
                    {
                        "name": "app",
                        "type": "application",
                        "description": "Main Django application",
                        "dependencies": []
                    },
                    {
                        "name": "models",
                        "type": "model",
                        "description": "Django models",
                        "dependencies": []
                    },
                    {
                        "name": "views",
                        "type": "view",
                        "description": "Django views",
                        "dependencies": ["models"]
                    },
                    {
                        "name": "urls",
                        "type": "router",
                        "description": "URL routing",
                        "dependencies": ["views"]
                    }
                ]
        elif language == "javascript" or language == "typescript":
            if framework == "react":
                return [
                    {
                        "name": "App",
                        "type": "component",
                        "description": "Main React application component",
                        "dependencies": []
                    },
                    {
                        "name": "components",
                        "type": "ui",
                        "description": "UI components",
                        "dependencies": []
                    },
                    {
                        "name": "services",
                        "type": "service",
                        "description": "API services",
                        "dependencies": []
                    }
                ]
            elif framework == "express":
                return [
                    {
                        "name": "app",
                        "type": "application",
                        "description": "Express application",
                        "dependencies": []
                    },
                    {
                        "name": "routes",
                        "type": "router",
                        "description": "API routes",
                        "dependencies": []
                    },
                    {
                        "name": "models",
                        "type": "model",
                        "description": "Data models",
                        "dependencies": []
                    }
                ]
        
        # Generic components
        return [
            {
                "name": "app",
                "type": "application",
                "description": "Main application",
                "dependencies": []
            },
            {
                "name": "data",
                "type": "model",
                "description": "Data handling",
                "dependencies": []
            },
            {
                "name": "ui",
                "type": "interface",
                "description": "User interface",
                "dependencies": ["data"]
            }
        ]

class CodeGenerator:
    """Generates code for application components with advanced capabilities"""
    
    def __init__(self, llm_integration: LLMIntegration):
        """
        Initialize the code generator
        
        Args:
            llm_integration: LLM integration for code generation
        """
        self.llm = llm_integration
        self.error_manager = ErrorManager()
        self.template_engine = jinja2.Environment(
            loader=jinja2.FileSystemLoader(
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
            ),
            autoescape=jinja2.select_autoescape(['html', 'xml']),
            trim_blocks=True,
            lstrip_blocks=True
        )
        self.ensure_template_dir()
    
    def ensure_template_dir(self):
        """Ensure the templates directory exists with basic templates"""
        template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
        os.makedirs(template_dir, exist_ok=True)
        
        # Create a basic Flask app template if it doesn't exist
        flask_app_template = os.path.join(template_dir, "flask_app.py.j2")
        if not os.path.exists(flask_app_template):
            with open(flask_app_template, 'w') as f:
                f.write("""# {{ app.name }} - {{ app.description }}
from flask import Flask, render_template, request, jsonify
{%- if app.database == 'sqlite' %}
import sqlite3
{%- endif %}
{%- if app.database == 'mongodb' %}
from pymongo import MongoClient
{%- endif %}
import os

app = Flask(__name__)

# Configuration
{%- if app.database == 'sqlite' %}
app.config['DATABASE'] = os.path.join(app.root_path, 'database.db')
{%- endif %}
{%- if app.database == 'mongodb' %}
app.config['MONGO_URI'] = os.environ.get('MONGO_URI', 'mongodb://localhost:27017/{{ app.name|lower }}')
{%- endif %}

# Routes
@app.route('/')
def index():
    return render_template('index.html')

{%- for component in app.components %}
{%- if component.type == 'view' %}
@app.route('/{{ component.name|lower }}')
def {{ component.name|lower }}():
    return render_template('{{ component.name|lower }}.html')
{%- endif %}
{%- endfor %}

if __name__ == '__main__':
    app.run(debug=True)
""")
    
    async def generate_component_code(
        self, 
        component: AppComponent, 
        architecture: AppArchitecture
    ) -> Dict[str, str]:
        """
        Generate code for a component
        
        Args:
            component: App component
            architecture: App architecture
            
        Returns:
            Dictionary mapping file paths to code
        """
        logger.info(f"Generating code for component: {component.name}")
        
        try:
            # Determine files needed for this component
            files = await self._determine_component_files(component, architecture)
            
            # Generate code for each file
            results = {}
            
            # Use ThreadPoolExecutor for parallel code generation
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = []
                for file_info in files:
                    file_path = file_info["path"]
                    futures.append(
                        executor.submit(
                            self._generate_file_code_sync,
                            file_path, 
                            file_info, 
                            component, 
                            architecture
                        )
                    )
                
                # Collect results
                for future in futures:
                    try:
                        file_path, code = future.result()
                        results[file_path] = code
                        logger.info(f"Generated code for {file_path}")
                    except Exception as e:
                        logger.error(f"Error in code generation thread: {str(e)}")
            
            return results
            
        except Exception as e:
            error = self.error_manager.handle_exception(
                e, 
                category=ErrorCategory.GENERATION,
                severity=ErrorSeverity.ERROR,
                operation="generate_component_code",
                component=component.name,
            )
            logger.error(f"Failed to generate code for component {component.name}: {str(e)}")
            
            # Return a minimal placeholder for the component
            return self._generate_placeholder_component(component, architecture)
    
    def _generate_placeholder_component(self, component: AppComponent, architecture: AppArchitecture) -> Dict[str, str]:
        """Generate placeholder code for a component when generation fails"""
        results = {}
        
        # Determine the file extension for the language
        extension = self._get_extension_for_language(architecture.language)
        
        # Create a placeholder file
        file_path = f"{component.name}{extension}"
        
        placeholder_code = f"""
# {component.name} - {component.description}
# This is a placeholder file. Code generation encountered an error.
# TODO: Implement this component

"""
        
        if architecture.language.lower() == "python":
            placeholder_code += """
def main():
    print("This component needs to be implemented")
    
if __name__ == "__main__":
    main()
"""
        elif architecture.language.lower() in ["javascript", "typescript"]:
            placeholder_code += """
function main() {
    console.log("This component needs to be implemented");
}

main();
"""
        
        results[file_path] = placeholder_code
        return results
    
    def _generate_file_code_sync(self, file_path: str, file_info: Dict[str, Any], 
                            component: AppComponent, architecture: AppArchitecture) -> Tuple[str, str]:
        """Synchronous wrapper for generate_file_code to use with ThreadPoolExecutor"""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            code = loop.run_until_complete(self._generate_file_code(
                file_path, file_info, component, architecture
            ))
            loop.close()
            return file_path, code
        except Exception as e:
            logger.error(f"Error generating code for {file_path}: {str(e)}")
            return file_path, f"# Error generating this file: {str(e)}\n# TODO: Implement this file"
    
    async def _determine_component_files(
        self, 
        component: AppComponent, 
        architecture: AppArchitecture
    ) -> List[Dict[str, Any]]:
        """
        Determine files needed for a component
        
        Args:
            component: App component
            architecture: App architecture
            
        Returns:
            List of file information dictionaries
        """
        prompt = f"""
Determine the files needed for this component in the application.
Return a JSON array of file objects, each with:
- path: Relative file path
- purpose: Purpose of this file
- dependencies: Other files this depends on
- content_type: Type of content (code, config, static, etc.)

Component:
{json.dumps(component.to_dict(), indent=2)}

Application Architecture:
{json.dumps(architecture.to_dict(), indent=2)}
"""
        try:
            response = await self.llm.generate_completion(prompt)
            files = json.loads(response)
            return files
        except Exception as e:
            error = self.error_manager.handle_exception(
                e, 
                category=ErrorCategory.ANALYSIS,
                severity=ErrorSeverity.ERROR,
                operation="determine_component_files"
            )
            logger.error(f"Failed to determine component files: {str(e)}")
            
            # Return a basic file for the component
            language = architecture.language.lower()
            extension = self._get_extension_for_language(language)
            
            # Generate a sensible file path based on component type and architecture
            file_path = self._generate_fallback_file_path(component, architecture, extension)
            
            return [{
                "path": file_path,
                "purpose": component.description,
                "dependencies": [],
                "content_type": "code"
            }]
    
    def _generate_fallback_file_path(self, component: AppComponent, architecture: AppArchitecture, extension: str) -> str:
        """Generate a sensible fallback file path when file determination fails"""
        framework = architecture.framework.lower()
        component_type = component.type.lower()
        component_name = component.name.lower()
        
        if framework == "flask":
            if component_type == "model":
                return f"models/{component_name}{extension}"
            elif component_type == "view":
                return f"routes/{component_name}{extension}"
            elif component_type == "template":
                return f"templates/{component_name}.html"
            elif component_type == "static":
                return f"static/{component_name}.css" if "css" in component.description.lower() else f"static/{component_name}.js"
        elif framework == "django":
            if component_type == "model":
                return f"{component_name}/models{extension}"
            elif component_type == "view":
                return f"{component_name}/views{extension}"
            elif component_type == "template":
                return f"{component_name}/templates/{component_name}.html"
        elif framework == "react":
            if component_type == "component":
                return f"src/components/{component.name.charAt(0).toUpperCase() + component.name.slice(1)}{extension}"
            elif component_type == "service":
                return f"src/services/{component_name}.service{extension}"
        
        # Default case
        return f"{component_name}{extension}"
    
    def _get_extension_for_language(self, language: str) -> str:
        """Get the file extension for a language"""
        extensions = {
            "python": ".py",
            "javascript": ".js",
            "typescript": ".ts",
            "java": ".java",
            "c#": ".cs",
            "ruby": ".rb",
            "php": ".php",
            "go": ".go",
            "rust": ".rs",
            "c++": ".cpp",
            "c": ".c"
        }
        return extensions.get(language.lower(), ".txt")
    
    async def _generate_file_code(
        self, 
        file_path: str, 
        file_info: Dict[str, Any], 
        component: AppComponent, 
        architecture: AppArchitecture
    ) -> str:
        """
        Generate code for a file
        
        Args:
            file_path: File path
            file_info: File information
            component: App component
            architecture: App architecture
            
        Returns:
            Generated code
        """
        # First, check if we have a template for this type of file
        template_code = self._try_generate_from_template(file_path, component, architecture)
        if template_code:
            return template_code
            
        # Otherwise, use the LLM to generate code
        file_ext = os.path.splitext(file_path)[1].lower()
        language = self._get_language_for_extension(file_ext) or architecture.language
        
        prompt = f"""
Generate complete, production-ready code for this file in {language}.
The code should be fully functional, well-structured, and follow best practices.
Include comprehensive error handling, logging, validation, and comments.
Do not use placeholder functions or methods - implement everything completely.

File Path: {file_path}
Purpose: {file_info['purpose']}

Component:
{json.dumps(component.to_dict(), indent=2)}

Application Architecture:
{json.dumps(architecture.to_dict(), indent=2)}

Generate only the code for this file, with no additional explanation.
"""
        response = await self.llm.generate_completion(prompt)
        
        # Clean up the response
        code = self._clean_generated_code(response, language)
        
        return code
    
    def _try_generate_from_template(self, file_path: str, component: AppComponent, architecture: AppArchitecture) -> Optional[str]:
        """Try to generate code from a template"""
        # Map file paths to template names
        template_map = {
            # Flask templates
            "app.py": "flask_app.py.j2",
            "__init__.py": "flask_init.py.j2",
            "models.py": "flask_models.py.j2",
            "views.py": "flask_views.py.j2",
            "routes.py": "flask_routes.py.j2",
            
            # Django templates
            "models.py": "django_models.py.j2",
            "views.py": "django_views.py.j2",
            "urls.py": "django_urls.py.j2",
            "admin.py": "django_admin.py.j2",
            
            # React templates
            "App.js": "react_app.js.j2",
            "index.js": "react_index.js.j2"
        }
        
        # Get the filename
        filename = os.path.basename(file_path)
        
        # Check if we have a template for this file
        if filename in template_map:
            template_name = template_map[filename]
            
            try:
                template = self.template_engine.get_template(template_name)
                return template.render(
                    app=architecture.to_dict(),
                    component=component.to_dict()
                )
            except (jinja2.exceptions.TemplateNotFound, jinja2.exceptions.TemplateError):
                # Template not found or error in template, fall back to LLM
                return None
        
        return None
    
    def _get_language_for_extension(self, ext: str) -> Optional[str]:
        """Get the language for a file extension"""
        extensions = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".jsx": "react",
            ".tsx": "react-typescript",
            ".java": "java",
            ".cs": "c#",
            ".rb": "ruby",
            ".php": "php",
            ".go": "go",
            ".rs": "rust",
            ".cpp": "c++",
            ".c": "c",
            ".html": "html",
            ".css": "css",
            ".scss": "scss",
            ".sql": "sql",
            ".json": "json",
            ".yml": "yaml",
            ".yaml": "yaml",
            ".md": "markdown",
            ".sh": "bash"
        }
        return extensions.get(ext)
    
    def _clean_generated_code(self, code: str, language: str) -> str:
        """
        Clean up generated code
        
        Args:
            code: Generated code
            language: Programming language
            
        Returns:
            Cleaned code
        """
        # Remove markdown code blocks if present
        code = re.sub(r'```[a-z]*\n', '', code)
        code = re.sub(r'```\n?, '', code)
        
        # Remove unnecessary comments about implementation
        code = re.sub(r'# TODO: Implement.*?\n', '', code)
        code = re.sub(r'// TODO: Implement.*?\n', '', code)
        
        # Ensure proper line endings
        code = code.replace('\r\n', '\n')
        
        # Add some import statements for common modules if they're not already there
        if language.lower() == "python":
            if "import " not in code and "from " not in code:
                imports = "import os\nimport sys\nimport logging\n\n"
                code = imports + code
        
        return code.strip()

class AppStructureGenerator:
    """Generates the overall application structure with improved reliability"""
    
    def __init__(self, llm_integration: LLMIntegration):
        """
        Initialize the app structure generator
        
        Args:
            llm_integration: LLM integration for structure generation
        """
        self.llm = llm_integration
        self.error_manager = ErrorManager()
        self.code_generator = CodeGenerator(llm_integration)
    
    async def generate_app_structure(
        self, 
        architecture: AppArchitecture, 
        output_dir: str
    ) -> Dict[str, Any]:
        """
        Generate the complete app structure
        
        Args:
            architecture: App architecture
            output_dir: Output directory
            
        Returns:
            Generation results
        """
        logger.info(f"Generating app structure for {architecture.name}")
        
        try:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate project structure information
            structure = await self._generate_project_structure(architecture)
            
            # Create directories
            for dir_path in structure.get("directories", []):
                full_path = os.path.join(output_dir, dir_path)
                os.makedirs(full_path, exist_ok=True)
                logger.info(f"Created directory: {dir_path}")
            
            # Generate configuration files
            config_files = await self._generate_config_files(architecture, structure)
            
            # Write configuration files
            for file_path, content in config_files.items():
                full_path = os.path.join(output_dir, file_path)
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                with open(full_path, 'w') as f:
                    f.write(content)
                logger.info(f"Created configuration file: {file_path}")
            
            # Generate code for each component with parallel processing
            component_files = {}
            
            # Use ThreadPoolExecutor for parallel component generation
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = {}
                for component in architecture.components:
                    future = executor.submit(
                        self._process_component_sync,
                        component,
                        architecture
                    )
                    futures[future] = component.name
                
                # Collect results
                for future in futures:
                    try:
                        component_name, files = future.result()
                        component_files[component_name] = files
                        
                        # Write component files
                        for file_path, content in files.items():
                            full_path = os.path.join(output_dir, file_path)
                            os.makedirs(os.path.dirname(full_path), exist_ok=True)
                            with open(full_path, 'w') as f:
                                f.write(content)
                            logger.info(f"Created component file: {file_path}")
                    except Exception as e:
                        logger.error(f"Error processing component {futures[future]}: {str(e)}")
            
            # Generate README and documentation
            docs = await self._generate_documentation(architecture, structure)
            
            # Write documentation files
            for file_path, content in docs.items():
                full_path = os.path.join(output_dir, file_path)
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                with open(full_path, 'w') as f:
                    f.write(content)
                logger.info(f"Created documentation file: {file_path}")
            
            # Generate deployment files (Docker, etc.)
            deployment_files = await self._generate_deployment_files(architecture, output_dir)
            
            # Return results
            return {
                "structure": structure,
                "config_files": config_files,
                "component_files": component_files,
                "documentation": docs,
                "deployment": deployment_files
            }
        except Exception as e:
            error = self.error_manager.handle_exception(
                e, 
                category=ErrorCategory.GENERATION,
                severity=ErrorSeverity.ERROR,
                operation="generate_app_structure"
            )
            logger.error(f"Failed to generate app structure: {str(e)}")
            
            # Return partial results
            return {
                "error": str(e),
                "structure": structure if 'structure' in locals() else {},
                "config_files": config_files if 'config_files' in locals() else {},
                "component_files": component_files if 'component_files' in locals() else {},
                "documentation": docs if 'docs' in locals() else {},
                "deployment": {}
            }
    
    def _process_component_sync(self, component: AppComponent, architecture: AppArchitecture) -> Tuple[str, Dict[str, str]]:
        """Synchronous wrapper for component processing to use with ThreadPoolExecutor"""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            files = loop.run_until_complete(self.code_generator.generate_component_code(component, architecture))
            loop.close()
            return component.name, files
        except Exception as e:
            logger.error(f"Error in component processing thread for {component.name}: {str(e)}")
            # Return empty files
            return component.name, {}
    
    async def _generate_project_structure(self, architecture: AppArchitecture) -> Dict[str, Any]:
        """
        Generate project structure information
        
        Args:
            architecture: App architecture
            
        Returns:
            Project structure information
        """
        prompt = f"""
Generate the complete project structure for this application.
Return a JSON object with:
- directories: Array of directory paths to create
- root_files: Array of files in the root directory
- standard_files: Common files for this technology stack
- entry_point: Main entry point file
- package_file: Package management file

Application Architecture:
{json.dumps(architecture.to_dict(), indent=2)}
"""
        try:
            response = await self.llm.generate_completion(prompt)
            structure = json.loads(response)
            logger.info(f"Generated project structure with {len(structure.get('directories', []))} directories")
            return structure
        except Exception as e:
            error = self.error_manager.handle_exception(
                e, 
                category=ErrorCategory.GENERATION,
                severity=ErrorSeverity.ERROR,
                operation="generate_project_structure"
            )
            logger.error(f"Failed to generate project structure: {str(e)}")
            
            # Return a basic structure based on the framework
            return self._generate_fallback_structure(architecture)
    
    def _generate_fallback_structure(self, architecture: AppArchitecture) -> Dict[str, Any]:
        """Generate a fallback project structure when the LLM fails"""
        framework = architecture.framework.lower()
        language = architecture.language.lower()
        
        if language == "python":
            if framework == "flask":
                return {
                    "directories": ["app", "app/models", "app/routes", "app/templates", "app/static", "tests"],
                    "root_files": ["requirements.txt", "app.py", ".gitignore", "README.md"],
                    "standard_files": ["app/__init__.py", "app/models/__init__.py", "app/routes/__init__.py"],
                    "entry_point": "app.py",
                    "package_file": "requirements.txt"
                }
            elif framework == "django":
                app_name = architecture.name.lower().replace(" ", "_")
                return {
                    "directories": [app_name, f"{app_name}/core", f"{app_name}/templates", f"{app_name}/static", "tests"],
                    "root_files": ["requirements.txt", "manage.py", ".gitignore", "README.md"],
                    "standard_files": [f"{app_name}/__init__.py", f"{app_name}/settings.py", f"{app_name}/urls.py", f"{app_name}/wsgi.py"],
                    "entry_point": "manage.py",
                    "package_file": "requirements.txt"
                }
            elif framework == "fastapi":
                return {
                    "directories": ["app", "app/api", "app/core", "app/models", "app/schemas", "tests"],
                    "root_files": ["requirements.txt", "main.py", ".gitignore", "README.md"],
                    "standard_files": ["app/__init__.py", "app/api/__init__.py", "app/models/__init__.py"],
                    "entry_point": "main.py",
                    "package_file": "requirements.txt"
                }
        elif language == "javascript" or language == "typescript":
            is_ts = language == "typescript"
            ext = ".ts" if is_ts else ".js"
            
            if framework == "react":
                return {
                    "directories": ["src", "src/components", "src/services", "src/styles", "public"],
                    "root_files": ["package.json", ".gitignore", "README.md", "tsconfig.json" if is_ts else ""],
                    "standard_files": [f"src/index{ext}", f"src/App{ext}", "public/index.html"],
                    "entry_point": f"src/index{ext}",
                    "package_file": "package.json"
                }
            elif framework == "express":
                return {
                    "directories": ["src", "src/routes", "src/models", "src/controllers", "src/middleware", "tests"],
                    "root_files": ["package.json", ".gitignore", "README.md", "tsconfig.json" if is_ts else ""],
                    "standard_files": [f"src/index{ext}", f"src/app{ext}"],
                    "entry_point": f"src/index{ext}",
                    "package_file": "package.json"
                }
            elif framework == "next":
                return {
                    "directories": ["pages", "components", "styles", "public", "lib", "api"],
                    "root_files": ["package.json", ".gitignore", "README.md", "next.config.js", "tsconfig.json" if is_ts else ""],
                    "standard_files": [f"pages/index{ext}", f"pages/_app{ext}"],
                    "entry_point": f"pages/index{ext}",
                    "package_file": "package.json"
                }
        
        # Generic structure
        return {
            "directories": ["src", "tests", "docs"],
            "root_files": ["README.md", ".gitignore"],
            "standard_files": ["src/main.py" if language == "python" else "src/index.js"],
            "entry_point": "src/main.py" if language == "python" else "src/index.js",
            "package_file": "requirements.txt" if language == "python" else "package.json"
        }
    
    async def _generate_config_files(
        self, 
        architecture: AppArchitecture, 
        structure: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Generate configuration files
        
        Args:
            architecture: App architecture
            structure: Project structure information
            
        Returns:
            Dictionary mapping file paths to content
        """
        config_files = {}
        
        # Generate each standard file
        files_to_generate = []
        files_to_generate.extend(structure.get("standard_files", []))
        files_to_generate.extend(structure.get("root_files", []))
        
        # Use ThreadPoolExecutor for parallel config file generation
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {}
            for file_path in files_to_generate:
                if not file_path:  # Skip empty entries
                    continue
                    
                future = executor.submit(
                    self._generate_config_file_sync,
                    file_path,
                    architecture
                )
                futures[future] = file_path
            
            # Collect results
            for future in futures:
                try:
                    file_path, content = future.result()
                    config_files[file_path] = content
                    logger.info(f"Generated content for {file_path}")
                except Exception as e:
                    logger.error(f"Error in config file generation thread for {futures[future]}: {str(e)}")
        
        return config_files
    
    def _generate_config_file_sync(self, file_path: str, architecture: AppArchitecture) -> Tuple[str, str]:
        """Synchronous wrapper for config file generation to use with ThreadPoolExecutor"""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            content = loop.run_until_complete(self._generate_config_file_content(file_path, architecture))
            loop.close()
            return file_path, content
        except Exception as e:
            logger.error(f"Error generating config file {file_path}: {str(e)}")
            # Return a placeholder
            return file_path, f"# Error generating this file: {str(e)}\n# TODO: Implement this file\n"
    
    async def _generate_config_file_content(self, file_path: str, architecture: AppArchitecture) -> str:
        """Generate content for a configuration file"""
        prompt = f"""
Generate complete, production-ready content for this configuration file.
The content should be properly formatted and follow best practices for this file type.

File Path: {file_path}

Application Architecture:
{json.dumps(architecture.to_dict(), indent=2)}

Generate only the file content, with no additional explanation.
"""
        try:
            content = await self.llm.generate_completion(prompt)
            
            # Clean up the content
            content = self._clean_file_content(content, file_path)
            
            return content
        except Exception as e:
            # Generate a fallback configuration based on the file path
            return self._generate_fallback_config(file_path, architecture)
    
    def _generate_fallback_config(self, file_path: str, architecture: AppArchitecture) -> str:
        """Generate fallback configuration files when the LLM fails"""
        filename = os.path.basename(file_path)
        
        if filename == "requirements.txt" and architecture.language.lower() == "python":
            return """# Basic requirements for a Flask web application
flask>=2.0.0
Werkzeug>=2.0.0
Jinja2>=3.0.0
itsdangerous>=2.0.0
click>=8.0.0
markupsafe>=2.0.0
# Database
sqlalchemy>=1.4.0
# Forms
wtforms>=3.0.0
flask-wtf>=1.0.0
# Testing
pytest>=6.0.0
# Security
flask-login>=0.5.0
# Deployment
gunicorn>=20.0.4
"""
        elif filename == "package.json" and architecture.language.lower() in ["javascript", "typescript"]:
            return """{
  "name": "%s",
  "version": "1.0.0",
  "description": "%s",
  "main": "index.js",
  "scripts": {
    "start": "node index.js",
    "dev": "nodemon index.js",
    "test": "jest"
  },
  "keywords": [],
  "author": "",
  "license": "MIT",
  "dependencies": {
    "express": "^4.17.1",
    "body-parser": "^1.19.0",
    "dotenv": "^10.0.0"
  },
  "devDependencies": {
    "nodemon": "^2.0.12",
    "jest": "^27.0.6"
  }
}
""" % (architecture.name.lower().replace(" ", "-"), architecture.description)
        elif filename == ".gitignore":
            return """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Node.js
node_modules/
npm-debug.log
yarn-debug.log
yarn-error.log
package-lock.json

# Virtual Environment
venv/
env/
ENV/

# IDE
.idea/
.vscode/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Application
logs/
*.log
.env
.env.local
.env.development.local
.env.test.local
.env.production.local
"""
        # Generic fallback
        return f"# Configuration file for {file_path}\n# TODO: Add proper configuration\n"
    
    def _clean_file_content(self, content: str, file_path: str) -> str:
        """
        Clean up generated file content
        
        Args:
            content: Generated content
            file_path: File path
            
        Returns:
            Cleaned content
        """
        # Remove markdown code blocks if present
        content = re.sub(r'```[a-z]*\n', '', content)
        content = re.sub(r'```\n?, '', content)
        
        # Ensure proper line endings
        content = content.replace('\r\n', '\n')
        
        # Special handling for specific file types
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.json':
            # Ensure valid JSON
            try:
                parsed = json.loads(content)
                content = json.dumps(parsed, indent=2)
            except json.JSONDecodeError:
                # If it's not valid JSON, try to fix common issues
                content = re.sub(r'(?m)^\s*//.*, '', content)  # Remove comments
                content = re.sub(r',\s*}', '\n}', content)  # Remove trailing commas
                
                try:
                    parsed = json.loads(content)
                    content = json.dumps(parsed, indent=2)
                except json.JSONDecodeError:
                    # If still not valid, leave it as is
                    pass
        
        return content.strip()
    
    async def _generate_documentation(
        self, 
        architecture: AppArchitecture, 
        structure: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Generate documentation files
        
        Args:
            architecture: App architecture
            structure: Project structure information
            
        Returns:
            Dictionary mapping file paths to content
        """
        docs = {}
        
        # Generate README
        prompt = f"""
Generate a comprehensive README.md file for this application.
Include sections for:
1. Project overview and description
2. Features
3. Technology stack
4. Installation instructions
5. Usage/getting started
6. Project structure
7. API documentation (if applicable)
8. Testing
9. Deployment
10. License information

Application Architecture:
{json.dumps(architecture.to_dict(), indent=2)}

Project Structure:
{json.dumps(structure, indent=2)}

Generate only the README content, formatted in Markdown.
"""
        try:
            readme_content = await self.llm.generate_completion(prompt)
            docs["README.md"] = readme_content
            
            # Generate additional documentation
            if "directories" in structure and "docs" in structure["directories"]:
                docs["docs/INSTALLATION.md"] = await self._generate_installation_guide(architecture)
                docs["docs/ARCHITECTURE.md"] = await self._generate_architecture_doc(architecture)
                docs["docs/API.md"] = await self._generate_api_doc(architecture)
            
            return docs
        except Exception as e:
            error = self.error_manager.handle_exception(
                e, 
                category=ErrorCategory.GENERATION,
                severity=ErrorSeverity.ERROR,
                operation="generate_documentation"
            )
            logger.error(f"Failed to generate documentation: {str(e)}")
            
            # Generate a minimal README as fallback
            return {
                "README.md": self._generate_fallback_readme(architecture, structure)
            }
    
    def _generate_fallback_readme(self, architecture: AppArchitecture, structure: Dict[str, Any]) -> str:
        """Generate a fallback README when the LLM fails"""
        app_name = architecture.name
        description = architecture.description
        language = architecture.language
        framework = architecture.framework
        
        components = "\n".join([f"- {c.name}: {c.description}" for c in architecture.components])
        
        # Determine installation steps based on language/framework
        if language.lower() == "python":
            installation = """
## Installation

1. Clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the virtual environment:
   - Windows: `venv\\Scripts\\activate`
   - Unix/MacOS: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Run the application: `python app.py`
"""
        elif language.lower() in ["javascript", "typescript"]:
            installation = """
## Installation

1. Clone the repository
2. Install dependencies: `npm install`
3. Run the application: 
   - Development mode: `npm run dev`
   - Production mode: `npm start`
"""
        else:
            installation = """
## Installation

1. Clone the repository
2. Follow the standard installation process for this type of project
3. Run the application
"""
        
        # Create the README
        return f"""# {app_name}

{description}

## Technology Stack

- Language: {language}
- Framework: {framework}
- Database: {architecture.database or "None specified"}

## Main Components

{components}

{installation}

## Project Structure

This project follows a standard {framework} application structure.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
"""
    
    async def _generate_installation_guide(self, architecture: AppArchitecture) -> str:
        """Generate an installation guide"""
        prompt = f"""
Generate a detailed installation guide for this application.
Include sections for:
1. System requirements
2. Prerequisites
3. Step-by-step installation instructions
4. Configuration
5. Troubleshooting common issues

Application Architecture:
{json.dumps(architecture.to_dict(), indent=2)}

Generate only the installation guide content, formatted in Markdown.
"""
        try:
            return await self.llm.generate_completion(prompt)
        except Exception:
            # Fallback installation guide
            language = architecture.language.lower()
            framework = architecture.framework.lower()
            
            if language == "python":
                return f"""# Installation Guide

## System Requirements

- Python 3.8 or higher
- pip package manager
- Git

## Prerequisites

Before installing the application, ensure you have the following:

- A suitable Python environment
- Basic knowledge of {framework}
- (Optional) A virtual environment tool (venv, virtualenv, or conda)

## Installation Steps

1. Clone the repository:
   ```
   git clone <repository-url>
   cd {architecture.name.lower().replace(' ', '-')}
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   # On Windows
   venv\\Scripts\\activate
   # On Unix/MacOS
   source venv/bin/activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Run the application:
   ```
   python app.py
   ```

## Configuration

- Copy `.env.example` to `.env` and update the values as needed
- Configure your database settings in the config file

## Troubleshooting

- If you encounter dependency issues, try updating pip: `pip install --upgrade pip`
- For database connection issues, ensure your database server is running
- Check the logs for detailed error messages
"""
            elif language in ["javascript",