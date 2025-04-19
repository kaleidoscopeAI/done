# External Tools Setup Guide

Kaleidoscope AI can work with various external tools to enhance its capabilities. This guide provides instructions for installing these tools on different platforms.

## Ollama (Required for LLM Features)

Ollama is required for running local LLMs for code analysis and generation.

### Installation

1. Visit [Ollama's website](https://ollama.ai) and download the appropriate version for your OS
2. Install and start the Ollama service
3. Pull the CodeLlama model:
   ```bash
   ollama pull codellama
   ```

## Radare2 (Recommended for Binary Analysis)

Radare2 is used for binary analysis and disassembly.

### Linux Installation

```bash
git clone https://github.com/radareorg/radare2
cd radare2
sys/install.sh
```

### macOS Installation

```bash
brew install radare2
```

### Windows Installation

Download the installer from [https://github.com/radareorg/radare2/releases](https://github.com/radareorg/radare2/releases)

## JS-Beautify (Recommended for JavaScript Analysis)

JS-Beautify is used to format and analyze JavaScript code.

### Installation (All Platforms)

```bash
npm install -g js-beautify
```

## RetDec (Optional for Advanced Decompilation)

RetDec is a retargetable machine-code decompiler.

### Installation

Follow the instructions at [https://github.com/avast/retdec](https://github.com/avast/retdec)

## Ghidra (Optional for Advanced Binary Analysis)

Ghidra is a software reverse engineering suite.

### Installation

1. Download from [https://ghidra-sre.org/](https://ghidra-sre.org/)
2. Extract the archive to a location of your choice
3. Add the `ghidra_server` path to your system PATH

## Configuration

After installing the tools, you can configure their paths in `config.json`:

```json
{
  "tools": {
    "radare2_path": "r2",
    "ghidra_path": "/path/to/ghidra_server",
    "retdec_path": "/path/to/retdec-decompiler",
    "js_beautify_path": "js-beautify"
  }
}
```

## Verification

To verify that the tools are properly installed and configured, run:

```bash
python kaleidoscope.py --check-tools
```

This will check for the availability of each tool and report any issues.
