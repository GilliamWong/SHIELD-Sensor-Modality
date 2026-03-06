"""Pytest configuration for Project SHIELD tests.

Adds src/ to sys.path so imports like
'from physics_based_classification.xxx import yyy' work.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
