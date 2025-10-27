#!/usr/bin/env python3
"""Test script to debug validation issues"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import json
from knowledge_validation import KnowledgeSchemaValidator

def test_single_file():
    """Test validation on a single file"""
    validator = KnowledgeSchemaValidator()

    # Test the entropy file
    result = validator.validate_knowledge_json('knowledge/foundations/info_theory_entropy.json')

    print(f"Is valid: {result.is_valid}")
    print(f"Score: {result.score}")
    print(f"Errors: {result.errors}")
    print(f"Warnings: {result.warnings}")

    # Also check metadata
    with open('knowledge/foundations/info_theory_entropy.json', 'r') as f:
        data = json.load(f)

    print(f"Metadata keys: {list(data.get('metadata', {}).keys())}")

if __name__ == "__main__":
    test_single_file()
