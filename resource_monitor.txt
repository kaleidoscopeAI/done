#!/usr/bin/env python3
import os
import sys
import logging
import threading
import time
from typing import Dict, Optional

# Add project root to path if running this file directly
if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    sys.path.insert(0, project_root)
    from src.utils.logging_config import configure_logging, get_logger
    configure_logging()
else:
    from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# Try to import psutil, but have fallback if not available
try:
    import psutil
    PSUTIL_AVAILABLE = True
    logger.info("psutil library found. Full resource monitoring available.")
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil library not found. Resource monitoring will be limited. pip install psutil for full functionality.")

class ResourceMonitor:
    """
    Monitors system resources and tracks estimated task allocations.
    Provides CPU and memory availability information for scheduling decisions.
    """

    def __init__(self, 
                 max_cpu_percent: float = 80.0, 
                 max_memory_percent: float = 80.0, 
                 monitor_interval_sec: int = 5,
                 enable_monitoring: bool = True):
        """
        Initialize the resource monitor.
        
        Args:
            max_cpu_percent: Maximum allowed CPU usage percentage (0-100)
            max_memory_percent: Maximum allowed memory usage percentage (0-100)
            monitor_interval_sec: How often to update resource measurements (seconds)
            enable_monitoring: Whether to enable monitoring thread
        """
        self._monitoring_enabled = PSUTIL_AVAILABLE and enable_monitoring
        
        # Set resource limits (ensure within valid range)
        self.max_cpu_percent = max(10.0, min(100.0, max_cpu_percent)) 
        self.max_memory_percent = max(10.0, min(100.0, max_memory_percent))
        self.monitor_interval_sec = monitor_interval_sec
        
        # Thread safety
        self.resource_lock = threading.RLock()
        
        # Track estimated resources allocated by tasks
        self.allocated_cpu_percent: float = 0.0
        self.allocated_memory_percent: float = 0.0

        # System state variables updated by monitor thread
        self._current_system_cpu_percent: float = 0.0
        self._current_system_memory_percent: float = 0.0
        
        # Resource usage history (for trend analysis)
        self.history_length = 60  # Keeps last 60 readings (5min at 5sec intervals)
        self.cpu_history = []
        self.memory_history = []
        
        # Create and start monitoring thread if enabled
        self.stop_event = threading.Event()
        self.monitor_thread = None
        
        if self._monitoring_enabled:
            # Get initial readings
            self._update_system_usage()
            
            # Start monitoring thread
            self.monitor_thread = threading.Thread(
                target=self._resource_monitor_loop, 
                daemon=True,
                name="ResourceMonitorThread"
            )
            self.monitor_thread.start()
            logger.info(f"ResourceMonitor started. Max CPU: {self.max_cpu_percent}%, Max Memory: {self.max_memory_percent}%")
        else:
            logger.info("ResourceMonitor running in estimation-only mode (no system monitoring).")

    def is_enabled(self) -> bool:
        """Returns whether full monitoring is enabled."""
        return self._monitoring_enabled

    def get_current_system_usage(self) -> Dict[str, float]:
        """
        Returns the latest measured system resource usage.
        
        Returns:
            Dict with cpu_percent and memory_percent
        """
        if not self.is_enabled():
            # Without monitoring, try to get a one-time reading
            if PSUTIL_AVAILABLE:
                try:
                    # Non-blocking reading (might be less accurate)
                    cpu = psutil.cpu_percent(interval=None)
                    memory = psutil.virtual_memory().percent
                    return {"cpu_percent": cpu, "memory_percent": memory}
                except Exception as e:
                    logger.error(f"Error getting one-time system usage: {e}")
                    
            # Return zeros if no monitoring or error
            return {"cpu_percent": 0.0, "memory_percent": 0.0}
            
        # Return last measured values from monitoring thread
        with self.resource_lock:
            return {
                "cpu_percent": self._current_system_cpu_percent,
                "memory_percent": self._current_system_memory_percent
            }

    def get_available_resources(self) -> Dict[str, float]:
        """
        Estimates available resources considering current usage and allocations.
        
        Returns:
            Dict with available cpu_percent and memory_percent
        """
        if not self.is_enabled() and not PSUTIL_AVAILABLE:
            # Without psutil, assume all resources are available (limited functionality)
            return {"cpu_percent": 100.0, "memory_percent": 100.0}

        # Get current system metrics (either from monitoring thread or one-time reading)
        system_usage = self.get_current_system_usage()
        system_cpu = system_usage["cpu_percent"]
        system_mem = system_usage["memory_percent"]

        with self.resource_lock:
            # Available = Max Allowed - Current System Usage - Currently Allocated Estimate
            available_cpu = max(0.0, self.max_cpu_percent - system_cpu - self.allocated_cpu_percent)
            available_memory = max(0.0, self.max_memory_percent - system_mem - self.allocated_memory_percent)

        return {
            "cpu_percent": available_cpu,
            "memory_percent": available_memory
        }

    def can_allocate(self, estimated_resources: Dict[str, float]) -> bool:
        """
        Checks if estimated resources can be allocated.
        
        Args:
            estimated_resources: Dict with cpu_percent and memory_percent requirements
            
        Returns:
            True if resources can be allocated, False otherwise
        """
        if not self.is_enabled() and not PSUTIL_AVAILABLE:
            # Without monitoring, assume allocation is always possible
            return True

        available = self.get_available_resources()
        cpu_required = estimated_resources.get("cpu_percent", 0.0)
        memory_required = estimated_resources.get("memory_percent", 0.0)

        return (cpu_required <= available["cpu_percent"] and 
                memory_required <= available["memory_percent"])

    def allocate_resources(self, estimated_resources: Dict[str, float]) -> bool:
        """
        Attempts to allocate estimated resources for a task.
        
        Args:
            estimated_resources: Dict with cpu_percent and memory_percent to allocate
            
        Returns:
            True if allocation successful, False if not enough resources
        """
        if not self.is_enabled() and not PSUTIL_AVAILABLE:
            # Without monitoring, allocation always succeeds
            return True

        with self.resource_lock:
            # Double-check allocation availability within lock
            if not self.can_allocate(estimated_resources):
                return False

            cpu_required = estimated_resources.get("cpu_percent", 0.0)
            memory_required = estimated_resources.get("memory_percent", 0.0)

            self.allocated_cpu_percent += cpu_required
            self.allocated_memory_percent += memory_required
            
            logger.debug(f"Allocated: CPU +{cpu_required:.1f}% (Total: {self.allocated_cpu_percent:.1f}%), "
                         f"Memory +{memory_required:.1f}% (Total: {self.allocated_memory_percent:.1f}%)")
            return True

    def release_resources(self, estimated_resources: Dict[str, float]):
        """
        Releases previously allocated estimated resources.
        
        Args:
            estimated_resources: Dict with cpu_percent and memory_percent to release
        """
        if not self.is_enabled() and not PSUTIL_AVAILABLE:
            # Without monitoring, no resource tracking
            return

        with self.resource_lock:
            cpu_allocated = estimated_resources.get("cpu_percent", 0.0)
            memory_allocated = estimated_resources.get("memory_percent", 0.0)

            # Safeguard against negative values
            self.allocated_cpu_percent = max(0.0, self.allocated_cpu_percent - cpu_allocated)
            self.allocated_memory_percent = max(0.0, self.allocated_memory_percent - memory_allocated)
            
            logger.debug(f"Released: CPU -{cpu_allocated:.1f}% (Total: {self.allocated_cpu_percent:.1f}%), "
                         f"Memory -{memory_allocated:.1f}% (Total: {self.allocated_memory_percent:.1f}%)")

    def _update_system_usage(self):
        """Updates current system resource usage measurements."""
        if not PSUTIL_AVAILABLE:
            return
            
        try:
            # Get CPU usage (non-blocking)
            cpu_percent = psutil.cpu_percent(interval=None)
            
            # Measure again after short interval for better average (optional)
            time.sleep(0.1)
            second_reading = psutil.cpu_percent(interval=None)
            cpu_percent = (cpu_percent + second_reading) / 2
            
            # Get memory usage
            memory_info = psutil.virtual_memory()
            memory_percent = memory_info.percent
            
            # Update the system metrics with thread safety
            with self.resource_lock:
                self._current_system_cpu_percent = cpu_percent
                self._current_system_memory_percent = memory_percent
                
                # Update history (keeping last N readings)
                self.cpu_history.append(cpu_percent)
                self.memory_history.append(memory_percent)
                if len(self.cpu_history) > self.history_length:
                    self.cpu_history.pop(0)
                if len(self.memory_history) > self.history_length:
                    self.memory_history.pop(0)
                    
        except Exception as e:
            logger.error(f"Error updating system resource usage: {e}", exc_info=True)

    def _resource_monitor_loop(self):
        """Background thread to periodically measure system resources."""
        if not PSUTIL_AVAILABLE:
            logger.error("Resource monitoring thread started but psutil is not available!")
            return

        logger.info("Resource monitor thread started.")
        while not self.stop_event.is_set():
            try:
                self._update_system_usage()
                
                # Check for high resource usage warnings
                with self.resource_lock:
                    cpu = self._current_system_cpu_percent
                    memory = self._current_system_memory_percent
                
                # Warn if within 10% of limit
                if cpu > self.max_cpu_percent - 10:
                    logger.warning(f"System CPU usage high: {cpu:.1f}% (Limit: {self.max_cpu_percent}%)")
                if memory > self.max_memory_percent - 10:
                    logger.warning(f"System Memory usage high: {memory:.1f}% (Limit: {self.max_memory_percent}%)")

            except Exception as e:
                logger.error(f"Error in resource monitor loop: {e}", exc_info=True)
                # Avoid busy-looping on continuous errors
                time.sleep(self.monitor_interval_sec * 2)
                continue  # Skip the regular sleep

            # Wait for the specified interval before next check
            self.stop_event.wait(timeout=self.monitor_interval_sec)

        logger.info("Resource monitor thread stopped.")

    def get_resource_trends(self) -> Dict[str, Any]:
        """
        Get resource usage trends from history data.
        
        Returns:
            Dict with trend information
        """
        if not self.is_enabled() or not self.cpu_history:
            return {
                "trend_available": False,
                "message": "No trend data available - monitoring not enabled or no data collected yet"
            }
            
        with self.resource_lock:
            cpu_history = self.cpu_history.copy()
            memory_history = self.memory_history.copy()
            
        # Calculate simple trends
        cpu_avg = sum(cpu_history) / len(cpu_history)
        memory_avg = sum(memory_history) / len(memory_history)
        
        # Calculate trend direction (using last 10 readings if available)
        trend_window = min(10, len(cpu_history))
        recent_cpu = cpu_history[-trend_window:]
        recent_memory = memory_history[-trend_window:]
        
        cpu_direction = "stable"
        if trend_window >= 3:  # Need at least 3 points for trend
            if recent_cpu[-1] > recent_cpu[0] + 5:  # 5% increase threshold
                cpu_direction = "increasing"
            elif recent_cpu[-1] < recent_cpu[0] - 5:  # 5% decrease threshold
                cpu_direction = "decreasing"
                
        memory_direction = "stable"
        if trend_window >= 3:
            if recent_memory[-1] > recent_memory[0] + 5:
                memory_direction = "increasing"
            elif recent_memory[-1] < recent_memory[0] - 5:
                memory_direction = "decreasing"
        
        return {
            "trend_available": True,
            "cpu": {
                "current": cpu_history[-1],
                "avg": cpu_avg,
                "min": min(cpu_history),
                "max": max(cpu_history),
                "direction": cpu_direction
            },
            "memory": {
                "current": memory_history[-1],