"""
backend/heartbeat.py — Heartbeat Monitor for Long-running Inference
====================================================================

Monitors OpenVINO inference and sends heartbeat signals to prevent 
Reflex server disconnects during NPU load.
"""

import logging
import threading
import time
import asyncio
from typing import Optional, Callable
from dataclasses import dataclass

log = logging.getLogger("heartbeat")

@dataclass
class HeartbeatConfig:
    interval: float = 1.0  # Send heartbeat every 1 second
    timeout: float = 2.0  # Consider inference slow after 2 seconds
    max_heartbeats: int = 300  # Maximum heartbeats (5 minutes)

class HeartbeatMonitor:
    def __init__(self, config: HeartbeatConfig = None):
        self.config = config or HeartbeatConfig()
        self._active = False
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._start_time: Optional[float] = None
        self._heartbeat_count = 0
        self._callbacks: list[Callable] = []
        self._lock = threading.Lock()
    
    def add_callback(self, callback: Callable[[dict], None]):
        """Add callback function to receive heartbeat events."""
        with self._lock:
            self._callbacks.append(callback)
    
    def start_monitoring(self):
        """Start heartbeat monitoring in background thread."""
        if self._active:
            return
        
        with self._lock:
            self._active = True
            self._start_time = time.time()
            self._heartbeat_count = 0
            self._heartbeat_thread = threading.Thread(
                target=self._heartbeat_loop,
                name="heartbeat-monitor",
                daemon=True
            )
            self._heartbeat_thread.start()
            log.info("🫀 Heartbeat monitoring started")
    
    def stop_monitoring(self):
        """Stop heartbeat monitoring."""
        with self._lock:
            if not self._active:
                return
            
            self._active = False
            if self._heartbeat_thread:
                self._heartbeat_thread.join(timeout=2.0)
        
        log.info("💔 Heartbeat monitoring stopped")
    
    def _heartbeat_loop(self):
        """Background thread that sends periodic heartbeats."""
        while self._active:
            try:
                current_time = time.time()
                elapsed = current_time - self._start_time if self._start_time else 0
                self._heartbeat_count += 1
                
                # Create heartbeat payload
                heartbeat_data = {
                    "timestamp": current_time,
                    "elapsed_seconds": elapsed,
                    "heartbeat_count": self._heartbeat_count,
                    "is_slow_inference": elapsed > self.config.timeout,
                    "status": "active" if elapsed < self.config.timeout * 60 else "warning"
                }
                
                # Send to all callbacks
                for callback in self._callbacks:
                    try:
                        callback(heartbeat_data)
                    except Exception as e:
                        log.error(f"Heartbeat callback failed: {e}")
                
                # Check limits
                if self._heartbeat_count >= self.config.max_heartbeats:
                    log.warning(f"Heartbeat limit reached ({self.config.max_heartbeats})")
                    break
                
                time.sleep(self.config.interval)
                
            except Exception as e:
                log.error(f"Heartbeat loop error: {e}")
                time.sleep(self.config.interval)
    
    def get_status(self) -> dict:
        """Get current heartbeat status."""
        with self._lock:
            current_time = time.time()
            elapsed = current_time - self._start_time if self._start_time else 0
            
            return {
                "active": self._active,
                "elapsed_seconds": elapsed,
                "heartbeat_count": self._heartbeat_count,
                "is_slow_inference": elapsed > self.config.timeout,
                "status": "active" if elapsed < self.config.timeout * 60 else "warning"
            }

# Global heartbeat instance
_global_heartbeat = HeartbeatMonitor()

def start_heartbeat_monitoring():
    """Start global heartbeat monitoring."""
    _global_heartbeat.start_monitoring()

def stop_heartbeat_monitoring():
    """Stop global heartbeat monitoring."""
    _global_heartbeat.stop_monitoring()

def add_heartbeat_callback(callback: Callable[[dict], None]):
    """Add callback to global heartbeat monitor."""
    _global_heartbeat.add_callback(callback)

def get_heartbeat_status() -> dict:
    """Get global heartbeat status."""
    return _global_heartbeat.get_status()

# Decorator for automatic heartbeat monitoring
def with_heartbeat_monitoring(func):
    """Decorator to automatically monitor function execution time."""
    def wrapper(*args, **kwargs):
        start_heartbeat_monitoring()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            stop_heartbeat_monitoring()
    return wrapper
