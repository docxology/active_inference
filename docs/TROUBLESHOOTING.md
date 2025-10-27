# Troubleshooting Guide - Active Inference Knowledge Environment

**Comprehensive troubleshooting guide for common issues, debugging strategies, and solutions.**

## Overview

This guide provides solutions to common issues encountered when working with the Active Inference Knowledge Environment. It covers setup problems, runtime errors, integration issues, and optimization challenges.

## Table of Contents

1. [Installation and Setup Issues](#installation-and-setup-issues)
2. [Runtime Errors](#runtime-errors)
3. [Integration Problems](#integration-problems)
4. [Performance Issues](#performance-issues)
5. [Testing and Validation Issues](#testing-and-validation-issues)
6. [Documentation and Development Issues](#documentation-and-development-issues)

---

## Installation and Setup Issues

### Issue: Python Version Incompatibility

**Symptoms**: Package installation fails or runtime errors occur

```bash
ERROR: This package requires Python >=3.9, but you have Python 3.8
```

**Solutions**:
1. **Check Python Version**:
   ```bash
   python --version
   python3 --version
   ```

2. **Install Correct Python Version**:
   ```bash
   # Using pyenv (recommended)
   pyenv install 3.11
   pyenv local 3.11
   
   # Or use system package manager
   sudo apt install python3.11 python3.11-venv
   ```

3. **Create Virtual Environment with Correct Version**:
   ```bash
   python3.11 -m venv venv
   source venv/bin/activate
   ```

---

### Issue: Dependency Installation Failures

**Symptoms**: `pip install` or `uv sync` fails with dependency conflicts

```bash
ERROR: Could not resolve dependencies
```

**Solutions**:
1. **Use uv for Dependency Management** (recommended):
   ```bash
   # Install uv
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # Sync dependencies
   uv sync
   ```

2. **Clean Shared Environment**:
   ```bash
   rm -rf uv.lock
   uv sync
   ```

3. **Check for Conflicting Packages**:
   ```bash
   # List installed packages
   uv pip list
   
   # Try upgrading specific problematic packages
   uv pip install --upgrade <package_name>
   ```

4. **Reset Environment**:
   ```bash
   rm -rf venv
   python3 -m venv venv
   source venv/bin/activate
   uv sync
   ```

---

### Issue: Virtual Environment Not Activating

**Symptoms**: Commands fail because virtual environment is not active

```bash
bash: command not found
```

**Solutions**:
1. **Activate Virtual Environment**:
   ```bash
   # macOS/Linux
   source venv/bin/activate
   
   # Windows
   venv\Scripts\activate
   ```

2. **Check Activation**:
   ```bash
   which python  # Should show venv path
   echo $VIRTUAL_ENV  # Should show venv directory
   ```

3. **Auto-activate with direnv** (recommended):
   ```bash
   # Install direnv
   brew install direnv  # macOS
   sudo apt install direnv  # Linux
   
   # Add to shell config (.bashrc, .zshrc, etc.)
   eval "$(direnv hook bash)"
   
   # Create .envrc in project root
   echo "source venv/bin/activate" > .envrc
   direnv allow
   ```

---

## Runtime Errors

### Issue: Module Import Errors

**Symptoms**: Python fails to import modules

```bash
ModuleNotFoundError: No module named 'active_inference'
```

**Solutions**:
1. **Check Installation**:
   ```bash
   uv pip list | grep active_inference
   ```

2. **Reinstall in Development Mode**:
   ```bash
   uv pip install -e .
   ```

3. **Check Python Path**:
   ```python
   import sys
   print(sys.path)
   ```

4. **Add Project to PYTHONPATH**:
   ```bash
   export PYTHONPATH="${PYTHONPATH}:/path/to/active_inference"
   # Or in your shell config
   echo 'export PYTHONPATH="${PYTHONPATH}:$(pwd)"' >> ~/.bashrc
   ```

---

### Issue: LLM Service Connection Errors

**Symptoms**: LLM operations fail with connection errors

```bash
ConnectionError: Failed to connect to Ollama service
```

**Solutions**:
1. **Check Ollama Service Status**:
   ```bash
   # Check if Ollama is running
   curl http://localhost:11434/api/tags
   
   # Start Ollama if not running
   ollama serve
   ```

2. **Verify Service URL**:
   ```python
   # Check configuration
   from active_inference.llm import OllamaClient
   client = OllamaClient({"base_url": "http://localhost:11434"})
   
   # Test connection
   await client.check_connection()
   ```

3. **Check Firewall and Network**:
   ```bash
   # Test connectivity
   curl -v http://localhost:11434
   
   # Check firewall
   sudo ufw status  # Linux
   ```

4. **Use Alternative Configuration**:
   ```python
   # Configure with explicit settings
   config = {
       "base_url": "http://localhost:11434",
       "timeout": 30,
       "max_retries": 3
   }
   ```

---

### Issue: Knowledge Repository Loading Errors

**Symptoms**: Knowledge content fails to load or validate

```bash
ValidationError: Invalid knowledge node structure
```

**Solutions**:
1. **Validate Knowledge Files**:
   ```bash
   # Use validation tool
   python tools/json_validation.py
   ```

2. **Check JSON Syntax**:
   ```bash
   # Validate JSON files
   python -m json.tool knowledge/foundations/*.json > /dev/null
   ```

3. **Verify Schema Compliance**:
   ```bash
   # Run knowledge validation
   python tools/knowledge_validation.py
   ```

4. **Fix Common Schema Issues**:
   ```json
   {
     "id": "correct_id",
     "title": "Title",
     "content_type": "foundation",
     "difficulty": "beginner",
     "description": "Description",
     "prerequisites": [],
     "tags": ["tag1", "tag2"],
     "learning_objectives": ["objective1"],
     "content": {
       "overview": "Overview text"
     },
     "metadata": {
       "estimated_reading_time": 15,
       "author": "Author",
       "last_updated": "2024-12-01",
       "version": "1.0"
     }
   }
   ```

---

## Integration Problems

### Issue: Platform Server Won't Start

**Symptoms**: Platform server fails to start or crashes

```bash
OSError: [Errno 48] Address already in use
```

**Solutions**:
1. **Check Port Availability**:
   ```bash
   # Find process using port
   lsof -i :8080
   
   # Kill process if needed
   kill -9 <PID>
   ```

2. **Use Different Port**:
   ```bash
   # Configure different port
   export PLATFORM_PORT=8081
   python platform/serve.py
   ```

3. **Check Configuration**:
   ```bash
   # Verify server configuration
   python -c "from platform.serve import config; print(config)"
   ```

4. **Check Dependencies**:
   ```bash
   # Ensure all dependencies are installed
   uv sync --extra platform
   ```

---

### Issue: Cross-Component Integration Failures

**Symptoms**: Components fail to communicate

```bash
IntegrationError: Failed to connect to knowledge repository
```

**Solutions**:
1. **Verify Component Initialization**:
   ```python
   from active_inference.platform import Platform
   
   platform = Platform(config)
   assert platform.knowledge_repository is not None
   assert platform.search_engine is not None
   ```

2. **Check Service Health**:
   ```python
   # Check component health
   health = platform.health_check()
   print(health)
   ```

3. **Enable Debug Logging**:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

4. **Test Individual Components**:
   ```python
   # Test components independently
   from active_inference.knowledge import KnowledgeRepository
   
   repo = KnowledgeRepository(config)
   results = repo.search("entropy")
   print(f"Found {len(results)} results")
   ```

---

## Performance Issues

### Issue: Slow Knowledge Search

**Symptoms**: Search operations take too long

**Solutions**:
1. **Enable Indexing**:
   ```python
   from active_inference.knowledge import KnowledgeRepository
   
   config = {
       "root_path": "./knowledge",
       "auto_index": True,
       "index_optimization": True
   }
   repo = KnowledgeRepository(config)
   ```

2. **Use Appropriate Limits**:
   ```python
   # Limit search results
   results = repo.search("query", limit=10)
   ```

3. **Optimize Search Query**:
   ```python
   # Use specific filters
   results = repo.search(
       query="entropy",
       content_types=["foundation"],
       difficulty=["beginner", "intermediate"],
       limit=20
   )
   ```

4. **Enable Caching**:
   ```python
   config = {
       "cache_enabled": True,
       "cache_size": 1000,
       "cache_ttl": 3600
   }
   ```

---

### Issue: Memory Leaks or High Memory Usage

**Symptoms**: Application uses too much memory

**Solutions**:
1. **Enable Memory Profiling**:
   ```bash
   # Install memory profiler
   uv pip install memory-profiler
   
   # Profile specific function
   python -m memory_profiler script.py
   ```

2. **Check for Circular References**:
   ```python
   import gc
   
   # Force garbage collection
   gc.collect()
   
   # Check object counts
   print(gc.get_count())
   ```

3. **Optimize Data Structures**:
   ```python
   # Use generators instead of lists
   def large_iterable():
       for item in source:
           yield process(item)
   
   # Clear large caches
   cache.clear()
   ```

4. **Monitor Memory Usage**:
   ```python
   import psutil
   import os
   
   process = psutil.Process(os.getpid())
   memory_info = process.memory_info()
   print(f"Memory usage: {memory_info.rss / 1024 / 1024} MB")
   ```

---

## Testing and Validation Issues

### Issue: Tests Fail After Code Changes

**Symptoms**: Previously passing tests now fail

**Solutions**:
1. **Run Tests in Isolation**:
   ```bash
   # Run specific test file
   uv run pytest tests/unit/test_specific.py -v
   ```

2. **Clear Test Cache**:
   ```bash
   # Clear pytest cache
   rm -rf .pytest_cache
   
   # Run tests with cache disabled
   uv run pytest --cache-clear
   ```

3. **Check Test Dependencies**:
   ```bash
   # Ensure test dependencies are up to date
   uv sync --extra test
   ```

4. **Run Tests with Verbose Output**:
   ```bash
   # Detailed test output
   uv run pytest -vv -s
   ```

---

### Issue: Coverage Reports Show Low Coverage

**Symptoms**: Coverage below target thresholds

**Solutions**:
1. **Generate Detailed Coverage Report**:
   ```bash
   uv run pytest --cov=src/active_inference --cov-report=html --cov-report=term-missing
   ```

2. **Identify Coverage Gaps**:
   ```bash
   # Show uncovered lines
   uv run pytest --cov=src/active_inference --cov-report=term-missing
   ```

3. **Focus on Critical Paths**:
   ```bash
   # Check specific module coverage
   uv run pytest --cov=src/active_inference/knowledge tests/unit/
   ```

4. **Update Coverage Configuration**:
   ```python
   # In pytest.ini or coverage_config.py
   [coverage:run]
   omit = [
       "*/tests/*",
       "*/__pycache__/*",
       "*/venv/*"
   ]
   ```

---

## Documentation and Development Issues

### Issue: Documentation Build Failures

**Symptoms**: Sphinx documentation won't build

```bash
Sphinx error: toctree contains reference to nonexisting document
```

**Solutions**:
1. **Check Sphinx Configuration**:
   ```bash
   # Validate conf.py
   sphinx-build -b linkcheck docs/ docs/_build/
   ```

2. **Check for Broken References**:
   ```bash
   # Find broken links
   sphinx-build -b linkcheck docs/ docs/_build/ | grep broken
   ```

3. **Rebuild Documentation**:
   ```bash
   # Clean build directory
   rm -rf docs/_build/
   
   # Rebuild documentation
   make docs
   ```

4. **Check Dependencies**:
   ```bash
   # Ensure documentation dependencies are installed
   uv sync --extra docs
   ```

---

### Issue: Code Quality Checks Fail

**Symptoms**: Linting or formatting checks fail

**Solutions**:
1. **Run Auto-formatting**:
   ```bash
   # Format code
   uv run black src/
   uv run isort src/
   ```

2. **Fix Linting Issues**:
   ```bash
   # Show linting errors
   uv run flake8 src/
   
   # Auto-fix what's possible
   uv run autopep8 --in-place --recursive src/
   ```

3. **Type Checking**:
   ```bash
   # Run type checker
   uv run mypy src/active_inference
   ```

4. **Update Configuration**:
   ```bash
   # Check configuration files
   cat pyproject.toml | grep -A 5 black
   cat pyproject.toml | grep -A 5 flake8
   ```

---

## Advanced Debugging

### Enable Debug Logging

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug.log'),
        logging.StreamHandler()
    ]
)

# Set specific module logging
logging.getLogger('active_inference').setLevel(logging.DEBUG)
```

### Use Debugger

```bash
# Run with Python debugger
uv run python -m pdb script.py

# Or use pytest debugger
uv run pytest --pdb tests/
```

### Profiling

```python
# Profile execution time
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Your code here
do_work()

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)  # Top 20 functions
```

---

## Getting Help

### Community Resources

- **GitHub Issues**: [Report issues and bugs](https://github.com/docxology/active_inference/issues)
- **Discussions**: [Ask questions](https://github.com/docxology/active_inference/discussions)
- **Documentation**: [Comprehensive docs](https://active-inference.readthedocs.io)

### Diagnostic Information

When reporting issues, include:

```bash
# System information
python --version
pip list
uname -a  # Linux/Mac
systeminfo  # Windows

# Environment information
uv --version
echo $VIRTUAL_ENV
python -c "import sys; print(sys.path)"
```

---

**Last Updated**: December 2024  
**Version**: 1.0.0

*"Active Inference for, with, South AI"* - Resolving issues through comprehensive troubleshooting and community support.
