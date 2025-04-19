"""
Bridge module for the Kaleidoscope AI system.
This module provides interfaces to the C components of the system.
"""

from .c_bridge import BridgeInterface, BridgeError

__all__ = ['BridgeInterface', 'BridgeError']