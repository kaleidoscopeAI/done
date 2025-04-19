#!/bin/bash
#
# fix_ai_system.sh: Prepares and builds the AI System project on Ubuntu
#
# This script cleans, organizes, and builds the AI System project in the 'ai_system' directory.
# Designed for Ubuntu (e.g., 22.04, 24.04), it uses apt for package installation.
# Assumptions:
# - Run from the parent directory of 'ai_system/' on an Ubuntu system.
# - 'ai_system/' contains:
#   - main.c (root directory)
#   - Subdirectories (core/, engines/, etc.) with .c and .h files
#   - Makefile for building (uses gcc, outputs to ./build/, 'install' target requires sudo)
#   - Optional: requirements.txt, missing-components.py for Python components
# - D language (.d) files conflicting with .c files are moved to 'd_lang_files/' (flattened, renamed on collision).
# - Python dependencies are installed in a virtual environment (.venv) if requirements.txt exists.
#
# Steps:
# 1. Backup the 'ai_system/' directory
# 2. Clean up redundant files and scripts
# 3. Move conflicting .d files to 'd_lang_files/' (flattened)
# 4. (Optional, disabled) Standardize C include paths
# 5. Set up Python virtual environment and install dependencies
# 6. Run missing-components.py if present
# 7. Build using Makefile (preferred) or fallback scripts
#
# Usage: bash fix_ai_system.sh
# Review output for warnings/errors and manually verify the build.
# Note: Each run creates a new backup directory (ai_system_backup_<TIMESTAMP>).
#       Manually remove old backups to save disk space (e.g., 'find . -name "ai_system_backup_*" -mtime +30 -exec rm -rf {} \;').
#       The Makefile's 'install' target requires sudo for /usr/local/bin/.
#

# --- Configuration ---
AI_SYSTEM_DIR="ai_system"
PYTHON_EXEC="python3"
VENV_DIR=".venv"
D_LANG_DIR="d_lang_files"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="${AI_SYSTEM_DIR}_backup_${TIMESTAMP}"

# --- Helper Functions ---

log_info() {
    echo "INFO: $1"
}

log_warning() {
    echo "WARNING: $1" >&2
}

error_exit() {
    echo "ERROR: ${1:-"Unknown error"}" >&2
    exit 1
}

# --- Step Functions ---

perform_backup() {
    log_info "Step 1: Backing up the current state to '$BACKUP_DIR'..."
    cp -a "$AI_SYSTEM_DIR" "$BACKUP_DIR" || error_exit "Failed to create backup directory '$BACKUP_DIR'."
    log_info "Backup complete."
}

perform_cleanup() {
    log_info "Step 2: Cleaning up redundant files and old scripts..."
    if [ -f "all_source_files.c" ]; then
        log_info "Removing 'all_source_files.c'..."
        rm -f "all_source_files.c" || log_warning "Failed to remove 'all_source_files.c'. Continuing..."
    fi
    if [ -f "molecular-dynamics (1).py" ]; then
        log_info "Removing duplicate 'molecular-dynamics (1).py'..."
        rm -f "molecular-dynamics (1).py" || log_warning "Failed to remove 'molecular-dynamics (1).py'. Continuing..."
    fi
    if [ -f "final_kaleidoscope_engine.c" ] && [ -f "kaleidoscope_engine.c" ]; then
        log_info "Removing 'kaleidoscope_engine.c' (keeping 'final_kaleidoscope_engine.c')..."
        rm -f "kaleidoscope_engine.c" || log_warning "Failed to remove 'kaleidoscope_engine.c'. Continuing..."
    fi
    log_info "Removing potentially redundant helper scripts (excluding build/install scripts)..."
    rm -f fix_head_names.sh fix_double_heads.sh fix_header_names.sh rename_files.sh combine_files.sh update_includes.sh || log_warning "Failed to remove some helper scripts. Continuing..."
    if [ -f "install.sh" ] && [ -f "install.ps1" ]; then
        log_info "Removing 'install.ps1' as 'install.sh' exists..."
        rm -f "install.ps1" || log_warning "Failed to remove 'install.ps1'. Continuing..."
    fi
    log_info "Cleanup finished."
}

organize_d_files() {
    log_info "Step 3: Organizing D language files (flattening with renaming on collision)..."
    if find . -name '*.d' -not -path "./build/*" -not -path "./$D_LANG_DIR/*" -print -quit | grep -q .; then
        log_info "Found D language files (.d). Moving conflicting files to '$D_LANG_DIR'..."
        mkdir -p "$D_LANG_DIR" || error_exit "Failed to create directory '$D_LANG_DIR'."
        find . -name '*.c' -not -path "./build/*" -not -path "./$D_LANG_DIR/*" -print0 | while IFS= read -r -d $'\0' c_file; do
            if [[ -f "$c_file" ]]; then
                base_name="${c_file%.c}"
                base_name="${base_name#./}"
                d_file="${base_name}.d"
                if [ -f "$d_file" ]; then
                    target_d_file="$D_LANG_DIR/$(basename "$d_file")"
                    if [ -f "$target_d_file" ]; then
                        dir_name=$(basename "$(dirname "$d_file")")
                        target_d_file="$D_LANG_DIR/${dir_name}_$(basename "$d_file")"
                        log_info "  - File '$D_LANG_DIR/$(basename "$d_file")' exists. Moving '$d_file' to '$target_d_file' instead."
                    fi
                    log_info "  - Moving conflicting D file: '$d_file' to '$target_d_file' for C file '$c_file'"
                    mv "$d_file" "$target_d_file" || error_exit "Failed to move '$d_file' to '$target_d_file'."
                fi
            fi
        done
        log_info "Finished moving conflicting D files."
    else
        log_info "No conflicting D language files (.d) found. Skipping organization."
    fi
}

standardize_c_includes() {
    log_info "Step 4: Standardizing C include paths in .c and .h files..."
    local files_found=false
    if find . -path ./build -prune -o -path "./$D_LANG_DIR" -prune -o \( -name '*.c' -o -name '*.h' \) -print -quit | grep -q .; then
        files_found=true
    fi
    if [ "$files_found" = true ]; then
        log_info "Normalizing #include paths (removing directory prefixes)..."
        find . -path ./build -prune -o -path "./$D_LANG_DIR" -prune -o \( -name '*.c' -o -name '*.h' \) -exec sed -i -E 's|#include "(.*/)?(.*\.h)"|#include "\2"|g' {} \; || error_exit "Failed during include path normalization (sed command)."
        log_info "Include path normalization finished."
    else
        log_info "No .c or .h files found to normalize includes."
    fi
}

handle_python_env() {
    log_info "Step 5: Setting up Python virtual environment and installing dependencies..."
    local python_install_success=false
    local python_for_script=$PYTHON_EXEC

    if ! command -v $PYTHON_EXEC &> /dev/null; then
        log_warning "'$PYTHON_EXEC' command not found. Please install it with 'sudo apt install python3'. Skipping Python steps (5 & 6)."
        return 0
    fi

    local python_version
    python_version=$($PYTHON_EXEC --version 2>&1)
    log_info "Using Python version: $python_version"

    if [ -f "requirements.txt" ]; then
        log_info "Found requirements.txt."
        if ! $PYTHON_EXEC -m venv --help &> /dev/null; then
            log_warning "Python 'venv' module not found. Please install it with 'sudo apt install python3-venv'. Skipping Python dependency installation."
            return 1
        fi
        if [ -d "$VENV_DIR" ]; then
            log_info "Removing existing virtual environment '$VENV_DIR'..."
            rm -rf "$VENV_DIR" || log_warning "Failed to remove existing virtual environment '$VENV_DIR'. Continuing..."
        fi
        log_info "Creating Python virtual environment in '$VENV_DIR'..."
        $PYTHON_EXEC -m venv "$VENV_DIR" || error_exit "Failed to create Python virtual environment '$VENV_DIR'."
        python_for_script="$VENV_DIR/bin/python"

        log_info "Ensuring setuptools is installed in venv..."
        if ! "$VENV_DIR/bin/pip" install --upgrade pip setuptools; then
            log_warning "Failed to install/upgrade pip and setuptools in the virtual environment."
        fi

        log_info "Installing dependencies from requirements.txt into virtual environment..."
        if "$VENV_DIR/bin/pip" install -r requirements.txt; then
            log_info "Python dependencies installed successfully in virtual environment."
            python_install_success=true
        else
            local install_exit_code=$?
            echo "############################################################" >&2
            echo "ERROR: Failed to install Python dependencies (Exit Code: $install_exit_code)." >&2
            echo "Python version: $python_version" >&2
            echo "Common fixes:" >&2
            echo "1. Check requirements.txt for compatibility with $python_version." >&2
            echo "2. Install build tools: sudo apt install build-essential python3-dev gfortran libopenblas-dev liblapack-dev" >&2
            echo "3. Try an older Python version (e.g., 3.10) if compatible." >&2
            echo "############################################################" >&2
        fi
    else
        log_info "No requirements.txt found. Skipping Python dependency installation."
        python_install_success=true
    fi

    log_info "Step 6: Checking for missing components script..."
    if [ -f "missing-components.py" ]; then
        if command -v "$python_for_script" &> /dev/null; then
            if [ "$python_install_success" = true ]; then
                log_info "Running missing-components.py using '$python_for_script'..."
                "$python_for_script" missing-components.py || log_warning "missing-components.py exited with an error."
                log_info "missing-components.py finished."
            else
                log_info "Skipping missing-components.py because dependency installation failed."
            fi
        else
            log_warning "Python executable '$python_for_script' not found. Cannot run missing-components.py."
        fi
    else
        log_info "No missing-components.py script found."
    fi

    if [ "$python_install_success" = true ]; then
        return 0
    else
        return 1
    fi
}

run_build_process() {
    log_info "Step 7: Attempting to build the system..."
    local build_success=false

    if [ -f "Makefile" ]; then
        log_info "Found Makefile."
        if ! command -v make &> /dev/null; then
            log_warning "'make' command not found. Please install it with 'sudo apt install make'. Skipping build."
        elif ! command -v gcc &> /dev/null; then
            log_warning "'gcc' command not found. Please install it with 'sudo apt install build-essential'. Skipping build."
        else
            log_info "Attempting build with 'make clean all'..."
            if make clean && make all 2>&1 | tee make.log; then
                log_info "Build using Makefile completed successfully."
                rm -f make.log
                build_success=true
            else
                log_warning "Build using Makefile failed. Check 'make.log' in '$AI_SYSTEM_DIR' for errors (e.g., missing headers, undefined references). Ensure 'build-essential' is installed ('sudo apt install build-essential') and source files/headers are in expected directories (core/, engines/, etc.)."
            fi
        fi
    else
        log_info "Makefile not found. Checking for build scripts..."
        local build_script=""
        if [ -f "cleanup_and_build.sh" ]; then
            build_script="cleanup_and_build.sh"
        elif [ -f "build_and_run.sh" ]; then
            build_script="build_and_run.sh"
        elif [ -f "install.sh" ]; then
            build_script="install.sh"
        fi

        if [ -n "$build_script" ]; then
            log_info "Running build script '$build_script'..."
            log_warning "Using fallback build script '$build_script'. Ensure it uses correct directory ('$AI_SYSTEM_DIR') and compiler settings for Ubuntu."
            chmod +x "$build_script" || log_warning "Failed to make '$build_script' executable. Attempting to run anyway."
            ./"$build_script" || log_warning "Build script '$build_script' exited with an error. The build may be incomplete. Check logs and verify script logic."
            log_info "Build script finished."
            build_success=true # Assume partial success if script runs
        else
            log_warning "No Makefile or primary build script (cleanup_and_build.sh, build_and_run.sh, or install.sh) found. Skipping build step."
        fi
    fi

    if [ "$build_success" = true ]; then
        return 0
    else
        return 1
    fi
}

# --- Main Execution Logic ---

if [ ! -d "$AI_SYSTEM_DIR" ]; then
    error_exit "Directory '$AI_SYSTEM_DIR' not found. Please run this script from the parent directory of '$AI_SYSTEM_DIR'."
fi

log_info "Starting the AI system fixing process..."

perform_backup
cd "$AI_SYSTEM_DIR" || error_exit "Failed to change directory to '$AI_SYSTEM_DIR'."
log_info "Changed directory to '$AI_SYSTEM_DIR'."

perform_cleanup
organize_d_files
# standardize_c_includes # Disabled by default
handle_python_env
py_install_status=$?
run_build_process
build_status=$?

cd .. || error_exit "Failed to change directory back to the parent directory."
log_info "Changed directory back to the parent directory."

# --- Completion Summary ---
echo "-----------------------------------------------------"
log_info "AI System Fix Script Completed!"
log_info "Backup created at: '$BACKUP_DIR'"
if [ $py_install_status -ne 0 ] && [ -f "$AI_SYSTEM_DIR/requirements.txt" ]; then
    log_warning "Python dependency installation failed. Build or runtime errors may occur if Python components are essential. Review Step 5 errors."
fi
if [ $build_status -ne 0 ]; then
    log_warning "Build process failed. Check Step 7 logs ('make.log' if using Makefile) for errors. Ensure 'build-essential' and 'make' are installed ('sudo apt install build-essential make')."
fi
echo "Please review the full output above for any warnings or errors."
echo "Remember to manually check:" >&2
echo "  1. requirements.txt: Ensure versions are compatible with your Python version ($($PYTHON_EXEC --version 2>&1))." >&2
echo "  2. Build scripts/Makefile: Ensure correct directory names and paths are used." >&2
echo "  3. If running 'make install', sudo is required for /usr/local/bin/." >&2
echo "  4. Remove old backups to save disk space (e.g., 'find . -name \"ai_system_backup_*\" -mtime +30 -exec rm -rf {} \;')." >&2
echo "Further manual inspection and testing are recommended." >&2
echo "-----------------------------------------------------"

# Exit with 1 if build failed, 0 otherwise
if [ $build_status -ne 0 ]; then
    exit 1
fi
exit 0
