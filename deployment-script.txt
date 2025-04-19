#!/usr/bin/env python3
"""
Quantum Kaleidoscope Deployment Script (Enhanced & Integrated venv)
================================================================

This script automates the deployment of the Enhanced Autonomous Quantum Kaleidoscope system.
It handles installation, configuration, virtual environment setup, dependency management,
PATH modification, and launching the system.

Modified to include venv setup, requirements.txt generation, and PATH updates.
"""

import os
import sys
import time
import subprocess
import argparse
import logging
import shutil
import urllib.request
import zipfile
import json
import random
import platform
import socket
import signal
import stat # Needed for checking execute permissions more reliably
from typing import List, Dict, Any, Optional, Tuple

# --- Configuration ---
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("KaleidoscopeDeployer")

# Define the list of required Python packages
# (Consolidated from previous analysis - VERIFY THIS LIST IS COMPLETE)
REQUIRED_PACKAGES = [
    "numpy", "torch", "websockets", "fastapi", "uvicorn",
    "flask", "flask-socketio", "requests", "networkx", "matplotlib",
    "scipy", "pennylane", "plotly", "paramiko", "docker", "kubernetes",
    "streamlit", "transformers", "huggingface_hub", "llama-cpp-python",
    "tokenizers", "ctransformers", "spacy", "colorlog", "eventlet",
    "pandas", "Pillow", "psutil"
]

# --- Error Handling Helper ---
def error_exit(message, code=1):
    logger.critical(message)
    sys.exit(code)

# --- Deployment Class ---
class KaleidoscopeDeployer:
    """Handles the deployment of the Quantum Kaleidoscope system."""

    def __init__(self, install_dir: str = "./quantum_kaleidoscope",
                 data_dir: str = "./data",
                 config_file: Optional[str] = None,
                 ports: Tuple[int, int] = (8000, 8080)):

        self.install_dir = os.path.abspath(install_dir)
        self.data_dir = os.path.abspath(data_dir)
        self.config_file = config_file
        self.main_port, self.visualizer_port = ports
        self.venv_dir = os.path.join(self.install_dir, "venv")
        self.requirements_file = os.path.join(self.install_dir, "requirements.txt")
        self.pid_file = os.path.join(self.data_dir, "kaleidoscope.pid")

        # Configuration defaults (can be overridden by config file)
        self.config = {
            "system": {
                "dimension": 128,
                "auto_discovery": True,
                "max_active_nodes": 1000
            },
            "crawler": {
                "max_threads": 5,
                "rate_limit": 1.0,
                "respect_robots": True,
                "user_agent": "QuantumKaleidoscope/1.0"
            },
            "seed_urls": [
                "https://en.wikipedia.org/wiki/Quantum_computing",
                "https://en.wikipedia.org/wiki/Machine_learning",
                "https://en.wikipedia.org/wiki/Data_visualization"
            ],
            "paths": { # Added paths section for clarity
                "install_dir": self.install_dir,
                "data_dir": self.data_dir,
                "log_dir": os.path.join(self.data_dir, "logs"),
                "model_cache": os.path.expanduser("~/.cache/unravel-ai/models") # Example
            }
        }

        logger.info(f"Deployer initialized with install_dir={self.install_dir}, data_dir={self.data_dir}")

    def _run_command(self, cmd: List[str], cwd: Optional[str] = None, check: bool = True) -> bool:
        """Runs a shell command and logs output."""
        try:
            logger.info(f"Running command: {' '.join(cmd)} {'in '+cwd if cwd else ''}")
            process = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, check=check)
            if process.stdout:
                logger.info(f"Command stdout:\n{process.stdout.strip()}")
            if process.stderr:
                # Log stderr as warning unless check=True and it failed (which raises exception)
                log_level = logging.ERROR if check and process.returncode != 0 else logging.WARNING
                logger.log(log_level, f"Command stderr:\n{process.stderr.strip()}")
            return process.returncode == 0
        except FileNotFoundError:
            logger.error(f"Command not found: {cmd[0]}")
            return False
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed with code {e.returncode}: {' '.join(cmd)}")
            if e.stdout: logger.error(f"Stdout: {e.stdout.strip()}")
            if e.stderr: logger.error(f"Stderr: {e.stderr.strip()}")
            return False
        except Exception as e:
            logger.error(f"Error running command {' '.join(cmd)}: {e}", exc_info=True)
            return False

    def _is_port_available(self, port: int) -> bool:
        """Check if a port is available to use."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(0.5) # Short timeout
                s.bind(("127.0.0.1", port))
            return True
        except socket.error:
            return False
        except Exception as e:
            logger.warning(f"Unexpected error checking port {port}: {e}")
            return False # Assume not available on error

    def check_prerequisites(self) -> bool:
        """Check if the system meets all prerequisites."""
        logger.info("Checking prerequisites...")
        
        
        # <<< FIX START: Create necessary directories before checking disk space >>>
        try:
        # Ensure install_dir and data_dir are attributes of self
        install_dir_path = getattr(self, 'install_dir', './quantum_kaleidoscope')
        data_dir_path = getattr(self, 'data_dir', './data')
        os.makedirs(install_dir_path, exist_ok=True)
        os.makedirs(data_dir_path, exist_ok=True)
        logger.info(f"Ensured installation directories exist: {install_dir_path}, {data_dir_path}")
        except AttributeError as ae:
        logger.error(f"FATAL: Deployer object missing 'install_dir' or 'data_dir' attribute: {ae}")
        return False
        except Exception as e:
        logger.error(f"FATAL: Error creating required directories: {e}")
        return False
        # <<< FIX END >>>
        
        # <<< FIX START: Create necessary directories before checking disk space >>>
        try:
        # Ensure install_dir and data_dir are attributes of self
        install_dir_path = getattr(self, 'install_dir', './quantum_kaleidoscope')
        data_dir_path = getattr(self, 'data_dir', './data')
        os.makedirs(install_dir_path, exist_ok=True)
        os.makedirs(data_dir_path, exist_ok=True)
        logger.info(f"Ensured installation directories exist: {install_dir_path}, {data_dir_path}")
        except AttributeError as ae:
        logger.error(f"FATAL: Deployer object missing 'install_dir' or 'data_dir' attribute: {ae}")
        return False
        except Exception as e:
        logger.error(f"FATAL: Error creating required directories: {e}")
        return False
        # <<< FIX END >>>
        
        prereqs_ok = True

        # --- FIX START: Create necessary directories before checking disk space ---
        try:
            # Ensure install_dir and data_dir are attributes of self
            install_dir_path = getattr(self, 'install_dir', './quantum_kaleidoscope') # Use default if not found
            data_dir_path = getattr(self, 'data_dir', './data') # Use default if not found
            os.makedirs(install_dir_path, exist_ok=True)
            os.makedirs(data_dir_path, exist_ok=True)
            logger.info(f"Ensured installation directories exist: {install_dir_path}, {data_dir_path}")
        except AttributeError as ae:
            logger.error(f"FATAL: Deployer object missing 'install_dir' or 'data_dir' attribute: {ae}")
            return False
        except Exception as e:
            logger.error(f"FATAL: Error creating required directories: {e}")
            return False
        # --- FIX END ---

        # Check Python version
        python_version = platform.python_version_tuple()
        if int(python_version[0]) < 3 or (int(python_version[0]) == 3 and int(python_version[1]) < 8): # Often need 3.8+
            logger.error(f"Python 3.8+ required, found {platform.python_version()}")
            prereqs_ok = False

        # Check Python venv module
        try:
            subprocess.run([sys.executable, "-m", "venv", "--help"], capture_output=True, check=True, text=True)
            logger.info("Python 'venv' module found.")
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("Python 'venv' module is required but not found or not functional.")
            prereqs_ok = False

        # Check available disk space
        try:
            free_space_gb = shutil.disk_usage(self.install_dir).free / (1024**3)
            required_space_gb = 5 # Estimate needed space (adjust as needed)
            if free_space_gb < required_space_gb:
                logger.warning(f"Low disk space: Only {free_space_gb:.2f} GB available in {self.install_dir}. Need ~{required_space_gb} GB.")
                # prereqs_ok = False # Decide if this is a fatal error
            else:
                logger.info(f"Disk space check passed: {free_space_gb:.2f} GB available.")
        except Exception as e:
            logger.error(f"Error checking disk space: {e}")
            prereqs_ok = False

        # Check if required ports are available
        ports_available = True
        for port in [self.main_port, self.visualizer_port]:
            if not self._is_port_available(port):
                logger.error(f"Port {port} is already in use.")
                ports_available = False
        if ports_available:
            logger.info(f"Required ports ({self.main_port}, {self.visualizer_port}) appear available.")
        else:
            prereqs_ok = False

        if prereqs_ok:
            logger.info("All prerequisites seem to be met.")
        else:
            logger.error("Prerequisite check failed.")
        return prereqs_ok

    def load_config(self) -> bool:
        """Load configuration from file if provided."""
        if not self.config_file:
            logger.info("No config file specified, using defaults.")
            # Ensure default paths exist
            os.makedirs(self.config["paths"]["log_dir"], exist_ok=True)
            os.makedirs(self.config["paths"]["model_cache"], exist_ok=True)
            return True

        logger.info(f"Loading configuration from {self.config_file}")
        if not os.path.exists(self.config_file):
            logger.warning(f"Config file {self.config_file} not found. Using defaults and attempting to save.")
            return self.save_config(self.config_file) # Try saving defaults

        try:
            with open(self.config_file, 'r') as f:
                loaded_config = json.load(f)

            # Deep merge loaded config into defaults
            def deep_merge(source, destination):
                for key, value in source.items():
                    if isinstance(value, dict):
                        node = destination.setdefault(key, {})
                        deep_merge(value, node)
                    else:
                        destination[key] = value
                return destination

            self.config = deep_merge(loaded_config, self.config)
            logger.info(f"Successfully loaded and merged configuration.")
            # Ensure paths from config exist
            os.makedirs(self.config["paths"]["log_dir"], exist_ok=True)
            os.makedirs(self.config["paths"]["model_cache"], exist_ok=True)
            return True

        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from config file: {self.config_file}")
            return False
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return False

    def save_config(self, path: Optional[str] = None) -> bool:
        """Save the current configuration."""
        save_path = path or self.config_file or os.path.join(self.data_dir, "config.json")
        logger.info(f"Saving configuration to {save_path}")
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info("Configuration saved successfully.")
            return True
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            return False

    def _add_to_path_if_needed(self):
        """Suggests adding ~/.local/bin to PATH in shell config files if not present."""
        local_bin = os.path.expanduser("~/.local/bin")
        current_path = os.environ.get('PATH', '')

        if local_bin not in current_path.split(os.pathsep):
            logger.warning(f"Directory '{local_bin}' is not found in your PATH environment variable.")
            export_cmd = f'export PATH="$HOME/.local/bin:$PATH"'
            suggestion = f"    Consider adding the following line to your shell configuration file (e.g., ~/.bashrc, ~/.zshrc, ~/.profile):\n    {export_cmd}"
            print("\n" + "="*30 + " PATH Recommendation " + "="*30)
            print(suggestion)
            print("="*83 + "\n")
            # Attempt to add it automatically (use with caution)
            # shell_configs = ["~/.bashrc", "~/.zshrc", "~/.profile"]
            # added = False
            # for config_file in shell_configs:
            #     full_path = os.path.expanduser(config_file)
            #     if os.path.exists(full_path):
            #         try:
            #             with open(full_path, 'r') as f:
            #                 content = f.read()
            #             if export_cmd not in content:
            #                 with open(full_path, 'a') as f:
            #                     f.write(f"\n# Added by Kaleidoscope Deployer\n{export_cmd}\n")
            #                 logger.info(f"Added PATH export to {full_path}. Please restart your shell or run 'source {full_path}'.")
            #                 added = True
            #         except Exception as e:
            #             logger.warning(f"Could not automatically update {full_path}: {e}")
            # if not added:
            #     logger.warning("Could not automatically update PATH. Please add it manually.")
        else:
            logger.info(f"'{local_bin}' is already in PATH.")

    def _create_requirements_file(self):
        """Creates the requirements.txt file in the install directory."""
        logger.info(f"Creating requirements file: {self.requirements_file}")
        try:
            with open(self.requirements_file, 'w') as f:
                for package in REQUIRED_PACKAGES:
                    f.write(f"{package}\n")
            logger.info(f"requirements.txt created successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to create requirements.txt: {e}")
            return False

    def install(self) -> bool:
        """Install the Quantum Kaleidoscope system with integrated venv setup."""
        logger.info(f"Starting installation in {self.install_dir}...")

        # 0. Prerequisite Check
        if not self.check_prerequisites():
            error_exit("Prerequisites not met. Installation aborted.")

        # 1. Create Directories (already done in prereqs, but ensure again)
        try:
            os.makedirs(self.install_dir, exist_ok=True)
            os.makedirs(self.data_dir, exist_ok=True)
            os.makedirs(os.path.join(self.install_dir, "static", "js"), exist_ok=True)
            os.makedirs(os.path.join(self.install_dir, "static", "css"), exist_ok=True)
            os.makedirs(os.path.join(self.install_dir, "templates"), exist_ok=True)
            os.makedirs(os.path.join(self.data_dir, "logs"), exist_ok=True)
        except Exception as e:
            error_exit(f"Failed to create installation directories: {e}")

        # 2. Create requirements.txt
        if not self._create_requirements_file():
            error_exit("Failed to create requirements.txt file.")

        # 3. Setup Virtual Environment
        logger.info(f"Setting up virtual environment in {self.venv_dir}...")
        if os.path.exists(self.venv_dir):
            logger.info("Virtual environment already exists.")
        else:
            if not self._run_command([sys.executable, "-m", "venv", self.venv_dir]):
                error_exit("Failed to create virtual environment.")
            logger.info("Virtual environment created successfully.")

        # Determine correct pip path within venv
        pip_executable = os.path.join(self.venv_dir, 'bin', 'pip')
        if platform.system() == 'Windows':
            pip_executable = os.path.join(self.venv_dir, 'Scripts', 'pip.exe')

        if not os.path.exists(pip_executable):
             error_exit(f"Could not find pip executable in virtual environment: {pip_executable}")

        # 4. Install Dependencies into Venv
        logger.info("Installing Python dependencies into virtual environment...")
        # Upgrade pip first
        if not self._run_command([pip_executable, "install", "--upgrade", "pip"]):
            logger.warning("Failed to upgrade pip in venv, continuing anyway...")
        # Install from requirements file
        if not self._run_command([pip_executable, "install", "-r", self.requirements_file]):
            error_exit("Failed to install dependencies from requirements.txt.")
        logger.info("Python dependencies installed successfully.")

        # 5. Add ~/.local/bin to PATH (Suggest or attempt)
        self._add_to_path_if_needed() # Checks and suggests

        # 6. Copy Core Application Files (using an integration script concept)
        logger.info("Copying application files using integration logic...")
        # Simulate running the integration script logic here within the deployer
        # We need to know the source directory of the original scripts
        # Assuming the source directory is where this deployment script lives, or one level up
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Try sibling directory 'quatumunravel' or current dir as source
        possible_source_dirs = [os.path.join(script_dir, "../quatumunravel"), script_dir]
        source_dir_found = None
        for d in possible_source_dirs:
            if os.path.isdir(d) and os.path.exists(os.path.join(d, "integrate_kaleidoscope.py")): # Check for a known file
                 source_dir_found = d
                 break
        if not source_dir_found:
             error_exit(f"Cannot find the source directory containing integration scripts (e.g., 'quatumunravel') relative to {script_dir}")

        # Define integration function based on integrate_kaleidoscope.py logic
        # This is a simplified inline version; a real scenario might import/call it
        def run_integration_logic(src_dir, target_dir):
            logger.info(f"Running integration logic: {src_dir} -> {target_dir}")
            # This should replicate the core file copying and patching logic
            # from integrate_kaleidoscope.py, adapting paths as needed.
            # Example (simplified - assumes integration script functions are available):
            try:
                 # We need to load/execute the integration functions somehow
                 # For simplicity, assume they are defined/imported here
                 # --- Start Simplified Integration ---
                 temp_config_file = os.path.join(src_dir, "integrate_kaleidoscope.py") # Use the integration script itself
                 if not os.path.exists(temp_config_file):
                      error_exit(f"Integration script not found at {temp_config_file}")

                 # Execute the integration (needs modification of integrate_kaleidoscope.py to be callable)
                 # This requires refactoring integrate_kaleidoscope.py to be importable
                 # OR using subprocess again, which adds complexity
                 logger.warning("Inline integration logic needs proper implementation or subprocess call.")
                 # Example subprocess call:
                 integration_cmd = [sys.executable, temp_config_file, src_dir, target_dir, "--force"]
                 if not self._run_command(integration_cmd):
                      error_exit("Integration script execution failed.")
                 # --- End Simplified Integration ---
                 logger.info("Integration logic completed.")
                 return True
            except Exception as ie:
                 logger.error(f"Integration logic failed: {ie}", exc_info=True)
                 return False

        if not run_integration_logic(source_dir_found, self.install_dir):
             error_exit("Core application file integration failed.")

        # 7. Create/Save Configuration
        if not self.load_config(): # Load config file if specified, or use/save defaults
             error_exit("Failed to load or save configuration.")
        # Ensure config is saved in the install_dir/data_dir
        self.save_config(os.path.join(self.data_dir, "config.json"))

        logger.info("Installation completed successfully!")
        logger.info(f"Installed system in: {self.install_dir}")
        logger.info(f"Data directory: {self.data_dir}")
        return True

    def start(self) -> bool:
        """Start the Quantum Kaleidoscope system using the launcher within the venv."""
        logger.info("Starting the Autonomous Quantum Kaleidoscope system...")

        if self._is_running():
            logger.error("Quantum Kaleidoscope system is already running (PID file exists).")
            return False

        # Construct command to run launcher *within the virtual environment*
        launcher_script = os.path.join(self.install_dir, "quantum_kaleidoscope_launcher.py")
        venv_python = os.path.join(self.venv_dir, 'bin', 'python')
        if platform.system() == 'Windows':
            venv_python = os.path.join(self.venv_dir, 'Scripts', 'python.exe')

        if not os.path.exists(launcher_script):
            error_exit(f"Launcher script not found at {launcher_script}")
        if not os.path.exists(venv_python):
            error_exit(f"Virtual environment Python not found at {venv_python}. Run install first.")

        cmd = [
            venv_python, # Use python from venv
            launcher_script,
            "--data-dir", self.data_dir, # Pass data dir explicitly
            "--main-port", str(self.main_port),
            "--visualizer-port", str(self.visualizer_port)
        ]

        # Add auto-gen flag if enabled in loaded config
        if self.config.get("system", {}).get("auto_discovery", False):
            cmd.append("--auto-gen")

        try:
            logger.info(f"Executing start command: {' '.join(cmd)}")
            # Start the process in the background (detached)
            # Using start_new_session=True on Linux/macOS, DETACHED_PROCESS on Windows
            creationflags = 0
            if platform.system() == 'Windows':
                creationflags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                 # start_new_session=True equivalent
                 pass # Handled by Popen default or specific preexec_fn if needed

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE, # Redirect for cleaner terminal
                stderr=subprocess.PIPE, # Redirect for cleaner terminal
                text=True,
                cwd=self.install_dir, # Run from install directory
                start_new_session=True # Ensures it runs independently
                # Windows equivalent for start_new_session is creationflags=...
            )

            # Write PID to file
            time.sleep(1) # Give process a moment to start and get PID
            if process.pid is None:
                 raise RuntimeError("Process failed to start or get PID.")

            os.makedirs(os.path.dirname(self.pid_file), exist_ok=True)
            with open(self.pid_file, 'w') as f:
                f.write(str(process.pid))

            logger.info(f"Autonomous Quantum Kaleidoscope started with PID {process.pid}")
            logger.info(f"  Installation: {self.install_dir}")
            logger.info(f"  Main API/UI: http://localhost:{self.main_port}")
            logger.info(f"  Visualizer: http://localhost:{self.visualizer_port}")
            logger.info(f"  To view logs, check files in: {self.config['paths']['log_dir']}")
            logger.info(f"  To stop, run: python {sys.argv[0]} stop --data-dir {self.data_dir}")

            return True

        except Exception as e:
            logger.error(f"Error starting system: {e}", exc_info=True)
            # Clean up pid file if process failed to start properly
            if os.path.exists(self.pid_file):
                 try: os.remove(self.pid_file)
                 except: pass
            return False

    def stop(self) -> bool:
        """Stop the Quantum Kaleidoscope system."""
        logger.info("Stopping the Autonomous Quantum Kaleidoscope system...")

        if not os.path.exists(self.pid_file):
            logger.warning("PID file not found. System may not be running or was not started by this script.")
            # Attempt to find processes by name (more complex, platform-dependent)
            return True # Return True as there's nothing known to stop

        try:
            with open(self.pid_file, 'r') as f:
                pid = int(f.read().strip())
        except (FileNotFoundError, ValueError) as e:
             logger.error(f"Error reading PID file {self.pid_file}: {e}. Cannot stop process by PID.")
             return False

        try:
            logger.info(f"Attempting to terminate process with PID {pid}...")
            process = psutil.Process(pid)
            # Try graceful termination first
            process.terminate()
            try:
                process.wait(timeout=10) # Wait up to 10 seconds
                logger.info(f"Process {pid} terminated gracefully.")
            except psutil.TimeoutExpired:
                logger.warning(f"Process {pid} did not terminate gracefully, forcing kill...")
                process.kill()
                process.wait(timeout=5)
                logger.info(f"Process {pid} killed.")

        except psutil.NoSuchProcess:
            logger.info(f"Process with PID {pid} not found. It may have already stopped.")
        except Exception as e:
            logger.error(f"Error stopping process {pid}: {e}. Manual intervention may be required.")
            return False # Indicate failure
        finally:
            # Clean up PID file regardless of success/failure finding process
            if os.path.exists(self.pid_file):
                try:
                    os.remove(self.pid_file)
                    logger.debug(f"Removed PID file: {self.pid_file}")
                except Exception as e:
                    logger.warning(f"Could not remove PID file {self.pid_file}: {e}")

        logger.info("Stop command issued successfully.")
        return True # Indicate command executed, even if process was already gone

    def status(self) -> Dict[str, Any]:
        """Check the status of the Quantum Kaleidoscope system."""
        is_running, pid = self._is_running()

        status = {
            "running": is_running,
            "pid": pid,
            "install_dir": self.install_dir,
            "data_dir": self.data_dir,
            "main_port": self.main_port,
            "visualizer_port": self.visualizer_port,
            "config_loaded": bool(self.config_file and os.path.exists(self.config_file))
        }

        if is_running:
            # Try to get more detailed status from the API
            try:
                url = f"http://localhost:{self.main_port}/api/status"
                # Use requests if available, fallback to urllib
                try:
                    import requests
                    response = requests.get(url, timeout=5)
                    response.raise_for_status() # Raise exception for bad status codes
                    api_status = response.json()
                except ImportError:
                     with urllib.request.urlopen(url, timeout=5) as response:
                          if response.status != 200:
                               raise ConnectionError(f"API returned status {response.status}")
                          api_status = json.loads(response.read().decode('utf-8'))

                status["system_status"] = api_status
                logger.debug("Fetched detailed status from API.")
            except Exception as e:
                logger.warning(f"Could not get detailed status from API (http://localhost:{self.main_port}/api/status): {e}")
                status["system_status"] = {"error": "API unreachable or failed"}

        return status

    def _is_running(self) -> Tuple[bool, Optional[int]]:
        """Check if the system process is running based on PID file."""
        if not os.path.exists(self.pid_file):
            return False, None

        try:
            with open(self.pid_file, 'r') as f:
                pid = int(f.read().strip())
        except (FileNotFoundError, ValueError):
            # PID file missing or corrupt
            return False, None

        # Check if process with this PID exists
        if psutil.pid_exists(pid):
             try:
                 p = psutil.Process(pid)
                 # Basic check if the process command seems related (optional, platform dependent)
                 # cmdline = p.cmdline()
                 # if "quantum_kaleidoscope_launcher.py" in " ".join(cmdline):
                 #     return True, pid
                 return True, pid # Assume it's the correct process if PID exists
             except (psutil.NoSuchProcess, psutil.AccessDenied):
                 # Process died or we don't have permission
                 pass
             except Exception as e:
                 logger.warning(f"Error checking process {pid}: {e}")


        # If process doesn't exist, clean up stale PID file
        logger.warning(f"Process with PID {pid} not found. Cleaning up stale PID file.")
        try:
            os.remove(self.pid_file)
        except OSError as e:
            logger.error(f"Error removing stale PID file {self.pid_file}: {e}")
        return False, None


# --- Main Execution Block ---
def main():
    """Main entry point for command-line interface."""
    parser = argparse.ArgumentParser(
        description="Quantum Kaleidoscope Deployment Tool (Integrated venv)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # --- Global Arguments ---
    parser.add_argument("--install-dir", type=str, default="./quantum_kaleidoscope_installed",
                        help="Target installation directory for the system.")
    parser.add_argument("--data-dir", type=str, default="./kaleidoscope_data",
                        help="Directory for persistent data, logs, and config.")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to custom JSON configuration file (overrides defaults).")
    parser.add_argument("--main-port", type=int, default=8000,
                        help="Port for the main system API and UI.")
    parser.add_argument("--vis-port", type=int, default=8080,
                        help="Port for the separate visualizer (used by launcher).")

    # --- Command Subparsers ---
    subparsers = parser.add_subparsers(dest="command", required=True,
                                       help="Action to perform")

    # Install command
    install_parser = subparsers.add_parser("install", help="Install the system including venv and dependencies.")
    # install_parser has no specific arguments, uses globals

    # Start command
    start_parser = subparsers.add_parser("start", help="Start the installed system.")
    # start_parser has no specific arguments, uses globals

    # Stop command
    stop_parser = subparsers.add_parser("stop", help="Stop the running system.")
    # stop_parser has no specific arguments, uses globals

    # Status command
    status_parser = subparsers.add_parser("status", help="Check the status of the running system.")
    # status_parser has no specific arguments, uses globals

    # --- Argument Parsing ---
    args = parser.parse_args()

    # --- Initialize Deployer ---
    # Use provided config path or default location within data_dir
    config_file_path = args.config or os.path.join(args.data_dir, "config.json")
    deployer = KaleidoscopeDeployer(
        install_dir=args.install_dir,
        data_dir=args.data_dir,
        config_file=config_file_path, # Pass config path for loading/saving
        ports=(args.main_port, args.vis_port)
    )

    # --- Execute Command ---
    exit_code = 0
    if args.command == "install":
        if not deployer.install():
            exit_code = 1
    elif args.command == "start":
        if not deployer.start():
            exit_code = 1
    elif args.command == "stop":
        if not deployer.stop():
            exit_code = 1
    elif args.command == "status":
        status = deployer.status()
        print("\n=== Quantum Kaleidoscope Status ===")
        for key, value in status.items():
            if key == "system_status" and isinstance(value, dict):
                 print("\n--- API Status ---")
                 for api_key, api_value in value.items():
                      print(f"  {api_key.replace('_', ' ').title()}: {api_value}")
                 print("--- End API Status ---")
            else:
                print(f"{key.replace('_', ' ').title()}: {value}")
        print("=================================")
    else:
        parser.print_help()
        exit_code = 1

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
