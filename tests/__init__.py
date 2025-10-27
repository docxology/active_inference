"""
Active Inference Knowledge Environment - Test Suite

Comprehensive testing framework for the Active Inference Knowledge Environment.
Provides shared fixtures, utilities, and testing infrastructure for all components.
"""

import pytest
import tempfile
from pathlib import Path
from typing import Dict, Any, Generator

# Test configuration and shared fixtures
TEST_CONFIG = {
    "test_mode": True,
    "debug": True,
    "cache_enabled": False,
    "auto_index": False,
}
