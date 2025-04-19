{
  "llm_endpoint": "http://localhost:11434/api",
  "ollama_settings": {
    "default_model": "codellama",
    "available_models": ["codellama", "llama2", "mistral", "phi"],
    "temperature": 0.2,
    "max_tokens": 4096,
    "context_window": 8192
  },
  "max_workers": 4,
  "logging": {
    "level": "INFO",
    "file": "kaleidoscope.log",
    "console": true
  },
  "tools": {
    "radare2_path": "r2",
    "ghidra_path": "ghidra_server",
    "retdec_path": "retdec-decompiler",
    "js_beautify_path": "js-beautify"
  },
  "processing": {
    "chunk_size": 4096,
    "max_file_size": 100000000
  },
  "web_interface": {
    "host": "127.0.0.1",
    "port": 5000,
    "enable_upload": true,
    "max_upload_size": 50000000
  }
}
