"""
Alert System Module

Handles audio and visual alerts for drowsiness detection.
"""

import logging
import time
from pathlib import Path
from threading import Thread, Event
from typing import Optional

logger = logging.getLogger(__name__)

# Try to import audio libraries
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    logger.warning("pygame not installed, audio alerts may be limited")

try:
    import winsound
    WINSOUND_AVAILABLE = True
except ImportError:
    WINSOUND_AVAILABLE = False


class AlertSystem:
    """
    Alert system for drowsiness notifications.
    
    Supports both audio and visual alerts with cooldown to prevent spam.
    
    Attributes:
        sound_enabled: Whether audio alerts are enabled
        sound_file: Path to alert sound file
        cooldown_seconds: Minimum time between alerts
    """
    
    def __init__(
        self,
        sound_enabled: bool = True,
        sound_file: str = "alarm.wav",
        visual_enabled: bool = True,
        cooldown_seconds: float = 3.0
    ):
        """
        Initialize alert system.
        
        Args:
            sound_enabled: Enable audio alerts
            sound_file: Path to sound file
            visual_enabled: Enable visual alerts
            cooldown_seconds: Cooldown between alerts
        """
        self.sound_enabled = sound_enabled
        self.sound_file = Path(sound_file)
        self.visual_enabled = visual_enabled
        self.cooldown_seconds = cooldown_seconds
        
        # State tracking
        self.is_alerting = False
        self.last_alert_time = 0.0
        self.alert_count = 0
        
        # Threading
        self._stop_event = Event()
        self._alert_thread: Optional[Thread] = None
        
        # Initialize audio
        self._init_audio()
        
        logger.info(f"Alert system initialized (sound={sound_enabled}, visual={visual_enabled})")
    
    def _init_audio(self) -> None:
        """Initialize audio system."""
        if not self.sound_enabled:
            return
        
        if PYGAME_AVAILABLE:
            try:
                pygame.mixer.init()
                if self.sound_file.exists():
                    pygame.mixer.music.load(str(self.sound_file))
                    logger.info("Audio initialized with pygame")
                else:
                    logger.warning(f"Sound file not found: {self.sound_file}")
            except Exception as e:
                logger.error(f"Failed to initialize pygame audio: {e}")
                self.sound_enabled = False
    
    def trigger_alert(self, alert_type: str = "drowsiness") -> bool:
        """
        Trigger an alert.
        
        Args:
            alert_type: Type of alert (for logging/tracking)
            
        Returns:
            True if alert was triggered, False if on cooldown
        """
        current_time = time.time()
        
        # Check cooldown
        if current_time - self.last_alert_time < self.cooldown_seconds:
            return False
        
        # Already alerting
        if self.is_alerting:
            return False
        
        logger.warning(f"Alert triggered: {alert_type}")
        
        self.is_alerting = True
        self.last_alert_time = current_time
        self.alert_count += 1
        
        # Start alert in background thread
        if self.sound_enabled:
            self._stop_event.clear()
            self._alert_thread = Thread(target=self._play_alert, daemon=True)
            self._alert_thread.start()
        
        return True
    
    def _play_alert(self) -> None:
        """Play alert sound in background thread."""
        try:
            if PYGAME_AVAILABLE and self.sound_file.exists():
                pygame.mixer.music.play(-1)  # Loop until stopped
                
                while not self._stop_event.is_set():
                    time.sleep(0.1)
                
                pygame.mixer.music.stop()
                
            elif WINSOUND_AVAILABLE:
                # Windows beep as fallback
                while not self._stop_event.is_set():
                    winsound.Beep(1000, 500)  # 1000Hz for 500ms
                    time.sleep(0.5)
            else:
                # Console beep as last resort
                while not self._stop_event.is_set():
                    print('\a', end='', flush=True)  # ASCII bell
                    time.sleep(1.0)
                    
        except Exception as e:
            logger.error(f"Error playing alert: {e}")
        
        finally:
            self.is_alerting = False
    
    def stop_alert(self) -> None:
        """Stop the current alert."""
        if not self.is_alerting:
            return
        
        logger.info("Stopping alert")
        self._stop_event.set()
        
        if self._alert_thread is not None:
            self._alert_thread.join(timeout=1.0)
        
        self.is_alerting = False
    
    def should_show_visual_alert(self) -> bool:
        """
        Check if visual alert should be displayed.
        
        Returns:
            True if visual alert should be shown
        """
        return self.visual_enabled and self.is_alerting
    
    def get_alert_stats(self) -> dict:
        """
        Get alert statistics.
        
        Returns:
            Dictionary with alert stats
        """
        return {
            'total_alerts': self.alert_count,
            'is_alerting': self.is_alerting,
            'last_alert_time': self.last_alert_time,
        }
    
    def reset(self) -> None:
        """Reset alert system state."""
        self.stop_alert()
        self.alert_count = 0
        self.last_alert_time = 0.0
        logger.info("Alert system reset")
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self.stop_alert()
        
        if PYGAME_AVAILABLE:
            try:
                pygame.mixer.quit()
            except Exception:
                pass
        
        logger.info("Alert system cleaned up")
