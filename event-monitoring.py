import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import json
from pathlib import Path
from dataclasses import dataclass, field
import threading
from queue import Queue
import traceback

@dataclass
class SystemEvent:
    """Represents a system event."""
    event_type: str
    component: str
    message: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    severity: str = "INFO"
    metadata: Dict[str, Any] = field(default_factory=dict)
    trace_id: Optional[str] = None

class EventMonitor:
    """Monitors and logs system events and behavior."""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.event_queue = Queue()
        self.is_running = True
        self.event_handlers = {}
        self.event_history = []
        
        # Initialize logging
        self._setup_logging()
        
        # Start event processing thread
        self.processing_thread = threading.Thread(target=self._process_events)
        self.processing_thread.daemon = True
        self.processing_thread.start()

    def _setup_logging(self):
        """Sets up logging configuration."""
        self.logger = logging.getLogger("KaleidoscopeAI")
        self.logger.setLevel(logging.DEBUG)
        
        # File handler for all logs
        fh = logging.FileHandler(self.log_dir / "system.log")
        fh.setLevel(logging.DEBUG)
        
        # Console handler for important logs
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Create formatters and add to handlers
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_formatter = logging.Formatter(
            '%(levelname)s: %(message)s'
        )
        
        fh.setFormatter(file_formatter)
        ch.setFormatter(console_formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def log_event(self, event: SystemEvent):
        """Logs a system event."""
        self.event_queue.put(event)
        
        # Log to system logger
        log_level = getattr(logging, event.severity.upper(), logging.INFO)
        self.logger.log(
            log_level,
            f"{event.component}: {event.message}"
        )

    def register_handler(self, event_type: str, handler):
        """Registers a handler for a specific event type."""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)

    def _process_events(self):
        """Processes events from the queue."""
        while self.is_running:
            try:
                event = self.event_queue.get(timeout=1.0)
                self._handle_event(event)
            except Queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error processing event: {str(e)}")

    def _handle_event(self, event: SystemEvent):
        """Handles a single event."""
        try:
            # Store in history
            self.event_history.append(event)
            
            # Call registered handlers
            handlers = self.event_handlers.get(event.event_type, [])
            for handler in handlers:
                try:
                    handler(event)
                except Exception as e:
                    self.logger.error(
                        f"Error in event handler for {event.event_type}: {str(e)}"
                    )
            
            # Write to event log file
            self._write_event_log(event)
            
        except Exception as e:
            self.logger.error(f"Error handling event: {str(e)}")

    def _write_event_log(self, event: SystemEvent):
        """Writes event to log file."""
        log_file = self.log_dir / f"{event.component.lower()}_events.log"
        
        with open(log_file, 'a') as f:
            json.dump(
                {
                    'timestamp': event.timestamp,
                    'type': event.event_type,
                    'component': event.component,
                    'message': event.message,
                    'severity': event.severity,
                    'metadata': event.metadata,
                    'trace_id': event.trace_id
                },
                f
            )
            f.write('\n')

    def get_events(
        self,
        component: Optional[str] = None,
        severity: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None
    ) -> List[SystemEvent]:
        """Retrieves filtered events from history."""
        filtered_events = self.event_history

        if component:
            filtered_events = [
                e for e in filtered_events
                if e.component == component
            ]

        if severity:
            filtered_events = [
                e for e in filtered_events
                if e.severity == severity
            ]

        if start_time:
            filtered_events = [
                e for e in filtered_events
                if e.timestamp >= start_time
            ]

        if end_time:
            filtered_events = [
                e for e in filtered_events
                if e.timestamp <= end_time
            ]

        return filtered_events

    def get_event_summary(self) -> Dict[str, Any]:
        """Generates summary of system events."""
        summary = {
            'total_events': len(self.event_history),
            'components': {},
            'severity_distribution': {},
            'event_types': {}
        }

        for event in self.event_history:
            # Count by component
            if event.component not in summary['components']:
                summary['components'][event.component] = 0
            summary['components'][event.component] += 1

            # Count by severity
            if event.severity not in summary['severity_distribution']:
                summary['severity_distribution'][event.severity] = 0
            summary['severity_distribution'][event.severity] += 1

            # Count by event type
            if event.event_type not in summary['event_types']:
                summary['event_types'][event.event_type] = 0
            summary['event_types'][event.event_type] += 1

        return summary

    def get_error_trace(self, trace_id: str) -> List[SystemEvent]:
        """Retrieves all events associated with an error trace."""
        return [
            event for event in self.event_history
            if event.trace_id == trace_id
        ]

    def clear_old_events(self, max_age_days: int = 30):
        """Clears events older than specified age."""
        cutoff_time = (
            datetime.now() - datetime.timedelta(days=max_age_days)
        ).isoformat()
        
        self.event_history = [
            event for event in self.event_history
            if event.timestamp >= cutoff_time
        ]

    def start_monitoring(self):
        """Starts the event monitoring system."""
        self.is_running = True
        if not self.processing_thread.is_alive():
            self.processing_thread = threading.Thread(target=self._process_events)
            self.processing_thread.daemon = True
            self.processing_thread.start()

    def stop_monitoring(self):
        """Stops the event monitoring system."""
        self.is_running = False
        self.processing_thread.join()

class EventTracer:
    """Traces execution flow through system components."""
    
    def __init__(self, monitor: EventMonitor):
        self.monitor = monitor
        self.current_trace = None

    def start_trace(self) -> str:
        """Starts a new execution trace."""
        self.current_trace = datetime.now().isoformat()
        return self.current_trace

    def trace_event(self, event_type: str, component: str, message: str, metadata: Dict[str, Any] = None):
        """Logs an event as part of the current trace."""
        if not self.current_trace:
            self.current_trace = self.start_trace()

        event = SystemEvent(
            event_type=event_type,
            component=component,
            message=message,
            metadata=metadata or {},
            trace_id=self.current_trace
        )
        self.monitor.log_event(event)

    def end_trace(self):
        """Ends the current execution trace."""
        self.current_trace = None

class DebugMonitor:
    """Provides debugging capabilities for system components."""
    
    def __init__(self, monitor: EventMonitor):
        self.monitor = monitor
        self.debug_handlers = {}
        self.breakpoints = set()

    def set_breakpoint(self, component: str, condition: callable):
        """Sets a breakpoint for a component."""
        self.breakpoints.add((component, condition))

    def remove_breakpoint(self, component: str):
        """Removes breakpoints for a component."""
        self.breakpoints = {
            (c, cond) for c, cond in self.breakpoints
            if c != component
        }

    def register_debug_handler(self, component: str, handler):
        """Registers a debug handler for a component."""
        self.debug_handlers[component] = handler

    def handle_breakpoint(self, component: str, context: Dict[str, Any]):
        """Handles a breakpoint being hit."""
        for comp, condition in self.breakpoints:
            if comp == component and condition(context):
                handler = self.debug_handlers.get(component)
                if handler:
                    handler(context)
                    
                # Log breakpoint hit
                self.monitor.log_event(
                    SystemEvent(
                        event_type="BREAKPOINT",
                        component=component,
                        message=f"Breakpoint hit: {component}",
                        severity="DEBUG",
                        metadata=context
                    )
                )

    def get_component_state(self, component: str) -> Dict[str, Any]:
        """Gets current state of a component for debugging."""
        handler = self.debug_handlers.get(component)
        if handler:
            return handler()
        return {}
