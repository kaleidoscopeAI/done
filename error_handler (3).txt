#!/usr/bin/env python3
import os
import sys
import time
import traceback
import logging
import json
import hashlib
import asyncio
import random
import threading
from enum import Enum, auto
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Union, Tuple, Callable, Type, Set
from contextlib import contextmanager

# Add project root to path if running this file directly
if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    sys.path.insert(0, project_root)
    from src.utils.logging_config import configure_logging, get_logger
    configure_logging()
else:
    from src.utils.logging_config import get_logger

# Import definitions from the same directory
from .definitions import ErrorSeverity, ErrorCategory, ErrorContext, EnhancedError

logger = get_logger(__name__)

# --- ErrorHandlerRegistry ---
class ErrorHandlerRegistry:
    """Registry of error handlers for different categories and severities"""
    def __init__(self):
        self.handlers: Dict[Tuple[Optional[ErrorCategory], Optional[ErrorSeverity]], List[Callable]] = {}
        self.default_handlers: List[Callable[[EnhancedError], None]] = []

    def register_handler(
        self,
        handler: Callable[[EnhancedError], None],
        category: Optional[ErrorCategory] = None,
        severity: Optional[ErrorSeverity] = None,
        is_default: bool = False
    ):
        """Register an error handler."""
        if is_default:
            if handler not in self.default_handlers:
                self.default_handlers.append(handler)
                logger.debug(f"Registered default error handler: {handler.__name__}")
            return

        key = (category, severity) # Can register for category only, severity only, or both
        if key not in self.handlers:
            self.handlers[key] = []
        if handler not in self.handlers[key]:
             self.handlers[key].append(handler)
             logger.debug(f"Registered handler '{handler.__name__}' for Key={key}")

    def get_handlers(self, error: EnhancedError) -> List[Callable]:
        """Get all applicable handlers for an error, from specific to general."""
        applicable_handlers = []
        keys_to_check = [
            (error.category, error.severity),  # Most specific
            (error.category, None),            # Category specific
            (None, error.severity),            # Severity specific
        ]

        # Add handlers in order of specificity
        for key in keys_to_check:
            if key in self.handlers:
                 # Add handlers only if not already added
                 for handler in self.handlers[key]:
                     if handler not in applicable_handlers:
                          applicable_handlers.append(handler)

        # Add default handlers at the end
        for handler in self.default_handlers:
             if handler not in applicable_handlers:
                 applicable_handlers.append(handler)

        return applicable_handlers

# --- ErrorManager (Singleton) ---
class ErrorManager:
    """Central error management system (Singleton)."""
    _instance = None
    _initialized = False # Class variable to track initialization

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ErrorManager, cls).__new__(cls)
        return cls._instance

    def __init__(self, error_log_path: str = None, logger_instance=None):
        """Initialize the error manager."""
        if ErrorManager._initialized: # Prevent re-initialization
            return

        # Default error log path in workdir
        if error_log_path is None:
            workdir = os.environ.get("WORK_DIR", "unravel_ai_workdir")
            os.makedirs(os.path.join(workdir, "logs"), exist_ok=True)
            error_log_path = os.path.join(workdir, "logs", "errors.json")
            
        self.error_log_path = error_log_path
        self.logger = logger_instance or logger
        self.registry = ErrorHandlerRegistry()
        self.recent_errors: List[EnhancedError] = []
        self.max_recent_errors = 100
        self.error_counts: Dict[ErrorCategory, int] = {category: 0 for category in ErrorCategory}
        self.recovery_strategies: Dict[ErrorCategory, List[Callable]] = {}
        ErrorManager._initialized = True # Mark as initialized

        # Thread safety
        self._lock = threading.RLock()

        # Load persistent errors? (Optional)
        # self._load_persistent_errors()

        # Register built-in default handlers
        self._register_builtin_handlers()
        self.logger.info("ErrorManager initialized.")

    def _register_builtin_handlers(self):
        """Register built-in default error handlers."""
        self.registry.register_handler(self._log_error_handler, is_default=True)
        # Save critical/fatal errors by default
        self.registry.register_handler(self._save_error_handler, severity=ErrorSeverity.CRITICAL)
        self.registry.register_handler(self._save_error_handler, severity=ErrorSeverity.FATAL)

    def register_recovery_strategy(self, strategy: Callable[[EnhancedError], bool], category: ErrorCategory):
        """Register a recovery strategy for an error category."""
        with self._lock:
            if category not in self.recovery_strategies:
                self.recovery_strategies[category] = []
            if strategy not in self.recovery_strategies[category]:
                 self.recovery_strategies[category].append(strategy)
                 self.logger.debug(f"Registered recovery strategy '{strategy.__name__}' for category {category.name}")

    def handle_error(self, error: EnhancedError) -> bool:
        """Process an EnhancedError object."""
        with self._lock:
            # Add to recent errors queue
            self.recent_errors.append(error)
            if len(self.recent_errors) > self.max_recent_errors:
                self.recent_errors.pop(0)

            # Update error counts
            self.error_counts[error.category] = self.error_counts.get(error.category, 0) + 1

        # Log the error using its own log method (which uses severity)
        error.log(self.logger)

        # Get and run handlers
        handlers = self.registry.get_handlers(error)
        handled_by_specific = False
        if not handlers:
             self.logger.warning(f"No handlers registered for error: {error.error_id} ({error.category.name}, {error.severity.name})")

        for handler in handlers:
            try:
                handler(error)
                handled_by_specific = True # Consider handled if any handler runs
            except Exception as handler_e:
                self.logger.error(f"Error within error handler '{handler.__name__}': {handler_e}", exc_info=True)

        # Try recovery strategies only if not handled adequately by specific handlers? (Design choice)
        # Or always try recovery? Let's always try recovery if available.
        recovery_successful = False
        with self._lock:
            if error.category in self.recovery_strategies:
                self.logger.info(f"Attempting recovery strategies for error {error.error_id} (Category: {error.category.name})")
                for strategy in self.recovery_strategies[error.category]:
                    try:
                        if strategy(error): # Strategy returns True if successful
                            self.logger.info(f"Recovery strategy '{strategy.__name__}' succeeded for error {error.error_id}")
                            recovery_successful = True
                            break # Stop after first successful recovery
                    except Exception as recovery_e:
                        self.logger.error(f"Error during recovery strategy '{strategy.__name__}': {recovery_e}", exc_info=True)

        return recovery_successful # Return if recovery was successful

    # --- Built-in Handlers ---
    def _log_error_handler(self, error: EnhancedError):
        """Default handler to ensure error is logged (already done by error.log)."""
        pass # Logging is handled by error.log() called in handle_error

    def _save_error_handler(self, error: EnhancedError):
        """Default handler to save severe errors to a JSON log file."""
        self.logger.debug(f"Executing save error handler for {error.error_id}")
        errors = []
        try:
            if os.path.exists(self.error_log_path):
                try:
                    with open(self.error_log_path, 'r') as f:
                        content = f.read()
                        if content: # Avoid error on empty file
                             errors = json.loads(content)
                        if not isinstance(errors, list): # Handle corrupted file
                             self.logger.warning(f"Error log file '{self.error_log_path}' corrupted. Starting fresh.")
                             errors = []
                except (json.JSONDecodeError, IOError) as e:
                    self.logger.warning(f"Error reading error log file '{self.error_log_path}': {e}. Starting fresh.")
                    errors = []

            errors.append(error.to_dict())

            # Optional: Limit log file size (e.g., keep last N errors)
            max_log_entries = 500
            if len(errors) > max_log_entries:
                 errors = errors[-max_log_entries:]

            # Ensure directory exists
            os.makedirs(os.path.dirname(self.error_log_path), exist_ok=True)
            
            with open(self.error_log_path, 'w') as f:
                json.dump(errors, f, indent=2)
            self.logger.debug(f"Saved error {error.error_id} to {self.error_log_path}")

        except Exception as e:
            # Log error about saving the error itself
            self.logger.error(f"CRITICAL: Failed to save error {error.error_id} to log file '{self.error_log_path}': {e}", exc_info=True)

    # --- Error Creation and Handling Helpers ---
    def create_error(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        exception: Optional[Exception] = None,
        operation: Optional[str] = None,
        component: Optional[str] = None,
        task_id: Optional[str] = None,
        **context_kwargs
    ) -> EnhancedError:
        """Creates an EnhancedError object."""
        context = ErrorContext(
            operation=operation or "N/A",
            component=component,
            task_id=task_id,
            **context_kwargs
        )
        error = EnhancedError(
            message=message,
            category=category,
            severity=severity,
            exception=exception,
            context=context
        )
        return error

    def handle_exception(
        self,
        exception: Exception,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        operation: Optional[str] = None,
        component: Optional[str] = None,
        task_id: Optional[str] = None,
        reraise: bool = False, # Option to re-raise after handling
        **context_kwargs
    ) -> Optional[EnhancedError]:
        """Creates an EnhancedError from an exception and handles it."""
        error = self.create_error(
            message=str(exception) or f"Exception of type {type(exception).__name__} occurred.",
            category=category,
            severity=severity,
            exception=exception,
            operation=operation,
            component=component,
            task_id=task_id,
            **context_kwargs
        )
        self.handle_error(error)

        if reraise:
            raise exception
        return error

    @contextmanager
    def error_context(
        self,
        operation: str,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        component: Optional[str] = None,
        task_id: Optional[str] = None,
        reraise: bool = True, # Default to re-raising after handling
        **context_kwargs
    ):
        """Context manager for handling exceptions within a block."""
        try:
            yield
        except Exception as e:
            self.handle_exception(
                exception=e,
                category=category,
                severity=severity,
                operation=operation,
                component=component,
                task_id=task_id,
                reraise=reraise, # Control re-raising
                **context_kwargs
            )


# --- RetryManager ---
class RetryManager:
    """Manages retrying operations with various strategies."""
    def __init__(self, logger_instance=None):
        self.logger = logger_instance or logger
        self.default_max_retries = 3
        self.default_initial_delay = 1.0
        self.default_max_delay = 60.0
        self.default_backoff_factor = 2.0

    async def retry_async(
        self,
        operation: Callable,
        *args,
        max_retries: int = None,
        initial_delay: float = None,
        max_delay: float = None,
        backoff_factor: float = None,
        retry_exceptions: Tuple[Type[Exception], ...] = (Exception,),
        operation_name: str = None, # Optional name for logging
        **kwargs
    ):
        """Retries an async operation with exponential backoff."""
        max_retries = max_retries if max_retries is not None else self.default_max_retries
        initial_delay = initial_delay if initial_delay is not None else self.default_initial_delay
        max_delay = max_delay if max_delay is not None else self.default_max_delay
        backoff_factor = backoff_factor if backoff_factor is not None else self.default_backoff_factor
        op_name = operation_name or getattr(operation, '__name__', 'Unnamed Operation')

        last_exception = None
        for attempt in range(max_retries + 1):
            try:
                return await operation(*args, **kwargs)
            except retry_exceptions as e:
                last_exception = e
                if attempt >= max_retries:
                    self.logger.error(f"All {max_retries+1} attempts failed for async operation '{op_name}'. Last error: {e}", exc_info=True)
                    raise # Re-raise the last exception
                else:
                    delay = min(initial_delay * (backoff_factor ** attempt), max_delay)
                    # Add jitter to delay
                    jitter = delay * 0.1
                    actual_delay = delay + random.uniform(-jitter, jitter)
                    self.logger.warning(f"Async operation '{op_name}' failed (Attempt {attempt+1}/{max_retries+1}). Retrying in {actual_delay:.2f}s. Error: {e}")
                    await asyncio.sleep(actual_delay)
            except Exception as non_retry_e:
                 # If it's not an exception we should retry, re-raise immediately
                 self.logger.error(f"Non-retryable error during async operation '{op_name}': {non_retry_e}", exc_info=True)
                 raise non_retry_e

    def retry(
        self,
        operation: Callable,
        *args,
        max_retries: int = None,
        initial_delay: float = None,
        max_delay: float = None,
        backoff_factor: float = None,
        retry_exceptions: Tuple[Type[Exception], ...] = (Exception,),
        operation_name: str = None,
        **kwargs
    ):
        """Retries a synchronous operation with exponential backoff."""
        # Similar logic as retry_async, but using time.sleep
        max_retries = max_retries if max_retries is not None else self.default_max_retries
        initial_delay = initial_delay if initial_delay is not None else self.default_initial_delay
        max_delay = max_delay if max_delay is not None else self.default_max_delay
        backoff_factor = backoff_factor if backoff_factor is not None else self.default_backoff_factor
        op_name = operation_name or getattr(operation, '__name__', 'Unnamed Operation')

        last_exception = None
        for attempt in range(max_retries + 1):
            try:
                return operation(*args, **kwargs)
            except retry_exceptions as e:
                last_exception = e
                if attempt >= max_retries:
                    self.logger.error(f"All {max_retries+1} attempts failed for operation '{op_name}'. Last error: {e}", exc_info=True)
                    raise
                else:
                    delay = min(initial_delay * (backoff_factor ** attempt), max_delay)
                    jitter = delay * 0.1
                    actual_delay = delay + random.uniform(-jitter, jitter)
                    self.logger.warning(f"Operation '{op_name}' failed (Attempt {attempt+1}/{max_retries+1}). Retrying in {actual_delay:.2f}s. Error: {e}")
                    time.sleep(actual_delay)
            except Exception as non_retry_e:
                 self.logger.error(f"Non-retryable error during operation '{op_name}': {non_retry_e}", exc_info=True)
                 raise non_retry_e


# --- GracefulDegradation ---
class GracefulDegradation:
    """Implements fallback strategies for graceful degradation."""
    def __init__(self, error_manager: ErrorManager = None, logger_instance=None):
        self.error_manager = error_manager or ErrorManager()
        self.logger = logger_instance or logger
        self.fallback_strategies: Dict[str, Callable] = {} # Key: "component.function"
        self.degradation_state: Dict[str, bool] = {} # Track which components are degraded

        # Register built-in strategies (example placeholders)
        # self._register_builtin_strategies() # Call this if you define built-ins

    def register_fallback(self, component: str, function: str, fallback: Callable):
        """Registers a fallback strategy for a component.function."""
        key = f"{component}.{function}"
        self.fallback_strategies[key] = fallback
        self.logger.debug(f"Registered fallback for {key}: {fallback.__name__}")

    def get_fallback(self, component: str, function: str) -> Optional[Callable]:
        """Gets the fallback strategy."""
        key = f"{component}.{function}"
        return self.fallback_strategies.get(key)

    @contextmanager
    def degradable_operation(self, component: str, function: str, *args, **kwargs):
        """Context manager for operations that can degrade gracefully."""
        key = f"{component}.{function}"
        try:
            # The 'yield' passes control back to the 'with' block
            # If the block completes without error, we're done.
            yield # Result of the block (if any) is implicitly returned by 'with' statement if needed
        except Exception as e:
            fallback = self.get_fallback(component, function)
            if fallback:
                self.logger.warning(f"Operation {key} failed with error: {e}. Attempting fallback {fallback.__name__}.", exc_info=True)
                self.degradation_state[key] = True # Mark as degraded
                try:
                    # Execute the fallback function.
                    # We need to return its result if the 'with' block expects one.
                    # The context manager itself doesn't explicitly return,
                    # but the fallback execution happens here.
                    fallback_result = fallback(original_exception=e, *args, **kwargs)
                    # If the fallback succeeds, we effectively swallow the original exception here.
                    # The result of the 'with' block would be None unless the fallback returns something
                    # and the caller assigns the 'with' statement, which is unusual.
                    # If the fallback *itself* raises an error, it propagates out.
                    self.logger.info(f"Fallback for {key} completed.")
                    # How to return the fallback result? The context manager protocol doesn't directly support this easily.
                    # Usually, the code inside the 'with' block would handle the result or expect None on fallback.
                    # Or, the fallback could modify state that the code after 'with' checks.
                    # For simplicity, we log and swallow the original exception if fallback runs without error.
                    # If the fallback should *provide* a return value *instead* of the original block,
                    # the calling code needs modification, or the fallback needs to store the result somewhere accessible.

                    # Let's assume the fallback logs or modifies state, and swallow the original error here if fallback succeeds.
                    # If the fallback itself fails, THAT exception will propagate.
                except Exception as fallback_e:
                     self.logger.error(f"Fallback function {fallback.__name__} for {key} also failed: {fallback_e}", exc_info=True)
                     raise e # Re-raise the *original* exception if fallback fails
            else:
                # No fallback available, handle the original error
                self.logger.error(f"Operation {key} failed and no fallback registered. Error: {e}", exc_info=True)
                # Handle the exception using ErrorManager before re-raising
                self.error_manager.handle_exception(
                     exception=e,
                     category=ErrorCategory.UNKNOWN, # Or try to categorize better
                     operation=f"{component}.{function}"
                )
                raise e # Re-raise the original exception


# --- ErrorMonitor ---
class ErrorMonitor:
    """Monitors errors, calculates trends, and triggers alerts."""
    def __init__(self, error_manager: ErrorManager = None, logger_instance=None):
        self.error_manager = error_manager or ErrorManager()
        self.logger = logger_instance or logger
        self.alert_thresholds: Dict[ErrorCategory, Dict[str, int]] = {} # {category: {'count': N, 'window': Sec}}
        self.alert_callbacks: Dict[ErrorCategory, List[Callable]] = {}
        self.error_trends: Dict[ErrorCategory, List[float]] = {} # {category: [timestamp1, timestamp2,...]}
        self.last_alert_time: Dict[ErrorCategory, float] = {}
        self.alert_cooldown_seconds: int = 300 # Default 5 minutes

        # Automatically subscribe to errors handled by the ErrorManager instance?
        # This requires ErrorManager to provide a way to subscribe, e.g., a callback list.
        # For now, assume add_error is called externally when an error occurs.

    def set_alert_threshold(self, category: ErrorCategory, count_threshold: int, time_window_seconds: int = 3600):
        """Sets an alert threshold for an error category."""
        if not isinstance(category, ErrorCategory):
             self.logger.error(f"Invalid category type for alert threshold: {type(category)}")
             return
        if count_threshold <= 0 or time_window_seconds <= 0:
             self.logger.error(f"Alert threshold count and window must be positive.")
             return
        self.alert_thresholds[category] = {'count': count_threshold, 'window': time_window_seconds}
        self.logger.info(f"Alert threshold set for {category.name}: {count_threshold} errors in {time_window_seconds}s")

    def register_alert_callback(self, category: ErrorCategory, callback: Callable[[ErrorCategory, int], None]):
        """Registers a callback function to be triggered when an alert threshold is met."""
        if not isinstance(category, ErrorCategory):
             self.logger.error(f"Invalid category type for alert callback: {type(category)}")
             return
        if category not in self.alert_callbacks:
            self.alert_callbacks[category] = []
        if callback not in self.alert_callbacks[category]:
            self.alert_callbacks[category].append(callback)
            self.logger.debug(f"Registered alert callback '{callback.__name__}' for category {category.name}")

    def add_error(self, error: EnhancedError):
        """Records an error and checks if alert thresholds are met."""
        if not isinstance(error, EnhancedError):
             self.logger.warning(f"Attempted to add non-EnhancedError to monitor: {type(error)}")
             return

        category = error.category
        timestamp = error.timestamp

        # Initialize trend list if needed
        if category not in self.error_trends:
            self.error_trends[category] = []

        # Add timestamp and prune old entries outside the largest window for this category
        # (Optimize by finding max window across all thresholds for this category if multiple exist)
        max_window = max(t['window'] for c, t in self.alert_thresholds.items() if c == category) if category in self.alert_thresholds else 3600 # Default window prune
        now = time.time()
        window_start = now - max_window
        self.error_trends[category] = [t for t in self.error_trends[category] if t > window_start]
        self.error_trends[category].append(timestamp)

        # Check alert threshold for this specific category
        self._check_alert_threshold(category)

    def _check_alert_threshold(self, category: ErrorCategory):
        """Checks if the alert threshold for a specific category is reached."""
        if category not in self.alert_thresholds:
            return

        threshold_info = self.alert_thresholds[category]
        count_threshold = threshold_info['count']
        time_window = threshold_info['window']
        now = time.time()

        # Count errors within the specific time window for this threshold
        window_start = now - time_window
        errors_in_window = [t for t in self.error_trends.get(category, []) if t > window_start]
        current_count = len(errors_in_window)

        if current_count >= count_threshold:
            # Check cooldown
            last_alert = self.last_alert_time.get(category, 0)
            if now - last_alert > self.alert_cooldown_seconds:
                self._trigger_alert(category, current_count)
                self.last_alert_time[category] = now # Update last alert time
            else:
                 self.logger.debug(f"Alert threshold for {category.name} reached ({current_count}), but within cooldown period.")
        else:
             # Reset last alert time if count drops below threshold? Optional.
             pass

    def _trigger_alert(self, category: ErrorCategory, error_count: int):
        """Triggers registered alert callbacks for a category."""
        self.logger.warning(f"ALERT Triggered for {category.name}: {error_count} errors reached threshold.")
        if category in self.alert_callbacks:
            for callback in self.alert_callbacks[category]:
                try:
                    # Consider running callbacks in separate threads/tasks if they might block
                    callback(category, error_count)
                except Exception as cb_e:
                    self.logger.error(f"Error executing alert callback '{callback.__name__}': {cb_e}", exc_info=True)

    def get_error_statistics(self) -> Dict[str, Any]:
        """Gets current error counts and trend statistics."""
        stats = {
            'counts_by_category': {cat.name: count for cat, count in self.error_manager.error_counts.items()},
            'recent_error_count': len(self.error_manager.recent_errors),
            'trends': {} # Calculate trends on demand
        }
        now = time.time()
        for category, timestamps in self.error_trends.items():
            stats['trends'][category.name] = {
                'last_minute': sum(1 for t in timestamps if t > now - 60),
                'last_10_minutes': sum(1 for t in timestamps if t > now - 600),
                'last_hour': sum(1 for t in timestamps if t > now - 3600),
                'total_tracked': len(timestamps)
            }
        return stats


# --- Example Usage ---
def example_main():
    """Example demonstrating the error handling system components."""
    # Create the main error handling components
    error_manager = ErrorManager()
    retry_manager = RetryManager()
    degradation = GracefulDegradation(error_manager=error_manager)
    monitor = ErrorMonitor(error_manager=error_manager)

    # --- Setup Monitoring ---
    monitor.set_alert_threshold(ErrorCategory.NETWORK, count_threshold=3, time_window_seconds=60) # 3 network errors in 60s
    monitor.set_alert_threshold(ErrorCategory.LLM, count_threshold=2, time_window_seconds=120) # 2 LLM errors in 120s

    def simple_alert_printer(category, count):
        print(f"\n *** ALERT CALLBACK: High rate of {category.name} errors! ({count} occurred recently) ***\n")

    monitor.register_alert_callback(ErrorCategory.NETWORK, simple_alert_printer)
    monitor.register_alert_callback(ErrorCategory.LLM, simple_alert_printer)

    # --- Example 1: Using the Context Manager ---
    print("\n--- Example 1: Context Manager ---")
    try:
        with error_manager.error_context("Reading User File", category=ErrorCategory.PARSING, component="FileReader", file_path="/path/to/nonexistent.txt", reraise=False):
             # Simulate an error
             print("Attempting operation within context manager...")
             raise FileNotFoundError("Simulated: User file is missing")
        print("Operation continued after context manager handled error (reraise=False).")
    except Exception as e:
         # This shouldn't be reached if reraise=