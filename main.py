#!/usr/bin/env python3
"""
Driver Drowsiness Detection System - CLI Entry Point

Real-time drowsiness detection using computer vision and facial landmark analysis.

Usage:
    python main.py              # Run with default settings
    python main.py --config config.yaml    # Run with custom config
    python main.py --no-sound   # Run without sound alerts
    python main.py --record output.avi     # Record output video

Controls:
    ESC or 'q' - Quit
    'r' - Reset detection state
    's' - Toggle sound alerts
    'p' - Pause/Resume
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.core import DrowsinessDetector, VideoStream
from src.core.drowsiness_detector import Config
from src.core.video_stream import VideoWriter

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Driver Drowsiness Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    
    parser.add_argument(
        '--camera', '-cam',
        type=int,
        default=0,
        help='Camera device ID (default: 0)'
    )
    
    parser.add_argument(
        '--no-sound',
        action='store_true',
        help='Disable sound alerts'
    )
    
    parser.add_argument(
        '--record', '-r',
        type=str,
        default=None,
        help='Record output to video file'
    )
    
    parser.add_argument(
        '--no-display',
        action='store_true',
        help='Run without display (headless mode)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("""
    +--------------------------------------------------------------+
    |         Driver Drowsiness Detection System                   |
    |                                                              |
    |  Monitoring for: Eye closure, Yawning, Head position        |
    |                                                              |
    |  Controls: ESC/q=Quit | r=Reset | s=Sound | p=Pause         |
    +--------------------------------------------------------------+
    """)
    
    # Load configuration
    config_path = Path(args.config)
    if config_path.exists():
        logger.info(f"Loading configuration from {config_path}")
        config = Config.from_yaml(str(config_path))
    else:
        logger.warning(f"Config file not found: {config_path}, using defaults")
        config = Config()
    
    # Apply command line overrides
    config.camera_device_id = args.camera
    if args.no_sound:
        config.sound_enabled = False
    
    # Initialize detector
    try:
        detector = DrowsinessDetector(config=config)
    except Exception as e:
        logger.error(f"Failed to initialize detector: {e}")
        return 1
    
    # Initialize video stream
    try:
        video_stream = VideoStream(
            device_id=config.camera_device_id,
            width=config.camera_width,
            height=config.camera_height,
            fps=config.camera_fps
        ).start()
    except Exception as e:
        logger.error(f"Failed to start video stream: {e}")
        return 1
    
    # Initialize video writer if recording
    video_writer = None
    if args.record:
        video_writer = VideoWriter(args.record, fps=config.camera_fps)
        logger.info(f"Recording to {args.record}")
    
    # Main loop state
    paused = False
    running = True
    
    print("\n[OK] System initialized. Starting detection...\n")
    
    try:
        while running:
            # Read frame
            grabbed, frame = video_stream.read()
            
            if not grabbed or frame is None:
                logger.warning("Failed to grab frame")
                time.sleep(0.1)
                continue
            
            if not paused:
                # Process frame
                result = detector.process_frame(frame)
                
                # Draw visualization
                if not args.no_display:
                    display_frame = detector.draw_visualization(frame, result)
                    cv2.imshow("Drowsiness Detection", display_frame)
                
                # Record if enabled
                if video_writer:
                    video_writer.write(display_frame if not args.no_display else frame)
                
                # Log periodic status
                if detector.frame_count % 300 == 0:  # Every ~10 seconds at 30fps
                    stats = detector.get_stats()
                    logger.info(
                        f"Stats: FPS={stats['avg_fps']:.1f}, "
                        f"Blinks={stats['blink_count']}, "
                        f"Yawns={stats['yawn_count']}, "
                        f"Alerts={stats['alert_count']}"
                    )
            else:
                # Just show frame when paused
                if not args.no_display:
                    cv2.putText(
                        frame, "PAUSED - Press 'p' to resume", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2
                    )
                    cv2.imshow("Drowsiness Detection", frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27 or key == ord('q'):  # ESC or 'q'
                logger.info("Quit requested")
                running = False
            
            elif key == ord('r'):  # Reset
                logger.info("Reset requested")
                detector.reset()
                print("[RESET] Detection reset")
            
            elif key == ord('s'):  # Toggle sound
                config.sound_enabled = not config.sound_enabled
                detector.alert_system.sound_enabled = config.sound_enabled
                status = "ON" if config.sound_enabled else "OFF"
                print(f"[SOUND] Sound alerts: {status}")
            
            elif key == ord('p'):  # Pause
                paused = not paused
                status = "PAUSED" if paused else "RUNNING"
                print(f"[STATUS] {status}")
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    
    finally:
        # Cleanup
        print("\n[STOP] Shutting down...")
        
        video_stream.stop()
        
        if video_writer:
            video_writer.release()
        
        detector.cleanup()
        cv2.destroyAllWindows()
        
        # Print final stats
        stats = detector.get_stats()
        print(f"""
    +--------------------------------------------------------------+
    |                     Session Summary                          |
    +--------------------------------------------------------------+
    |  Frames Processed: {stats['frames_processed']:<40}|
    |  Duration: {stats['elapsed_seconds']:.1f} seconds{' ' * 34}|
    |  Average FPS: {stats['avg_fps']:.1f}{' ' * 40}|
    |  Total Blinks: {stats['blink_count']:<38}|
    |  Total Yawns: {stats['yawn_count']:<39}|
    |  Alerts Triggered: {stats['alert_count']:<35}|
    +--------------------------------------------------------------+
        """)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
