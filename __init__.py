
"""
Bridge Module for Kaleidoscope AI

This module provides bridges between different components and languages
in the Kaleidoscope AI system.
"""

from .c_bridge import bridge, CPythonBridge, TaskStruct, NodeStruct

__all__ = ['bridge', 'CPythonBridge', 'TaskStruct', 'NodeStruct']