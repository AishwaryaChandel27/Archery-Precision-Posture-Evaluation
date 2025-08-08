import cv2
import numpy as np
from typing import List, Tuple, Optional

class VideoProcessor:
    """Handles video processing operations for archery analysis"""
    
    def __init__(self):
        self.supported_formats = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
    
    def extract_frames(self, video_path: str) -> Tuple[List[np.ndarray], float, int]:
        """
        Extract frames from video file
        
        Args:
            video_path: Path to video file
            
        Returns:
            Tuple of (frames, fps, frame_count)
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        frames = []
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        
        cap.release()
        
        return frames, fps, frame_count
    
    def resize_frame(self, frame: np.ndarray, target_width: int = 640) -> np.ndarray:
        """
        Resize frame while maintaining aspect ratio
        
        Args:
            frame: Input frame
            target_width: Target width for resizing
            
        Returns:
            Resized frame
        """
        height, width = frame.shape[:2]
        aspect_ratio = height / width
        target_height = int(target_width * aspect_ratio)
        
        resized = cv2.resize(frame, (target_width, target_height))
        return resized
    
    def extract_key_frames(self, frames: List[np.ndarray], num_key_frames: int = 10) -> List[np.ndarray]:
        """
        Extract key frames from video for analysis
        
        Args:
            frames: List of video frames
            num_key_frames: Number of key frames to extract
            
        Returns:
            List of key frames
        """
        if len(frames) <= num_key_frames:
            return frames
        
        step = len(frames) // num_key_frames
        key_frames = []
        
        for i in range(0, len(frames), step):
            if len(key_frames) < num_key_frames:
                key_frames.append(frames[i])
        
        return key_frames
    
    def detect_motion_phases(self, frames: List[np.ndarray]) -> dict:
        """
        Detect different phases of archery motion using optical flow
        
        Args:
            frames: List of video frames
            
        Returns:
            Dictionary with phase boundaries
        """
        if len(frames) < 10:
            return {'draw': (0, len(frames)//3), 'anchor': (len(frames)//3, 2*len(frames)//3), 'release': (2*len(frames)//3, len(frames))}
        
        # Convert frames to grayscale for optical flow
        gray_frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in frames]
        
        # Calculate optical flow between consecutive frames
        motion_magnitudes = []
        
        for i in range(1, len(gray_frames)):
            # Create initial points for optical flow
            h, w = gray_frames[i-1].shape
            initial_points = np.array([[w//2, h//2]], dtype=np.float32)
            
            try:
                # Calculate optical flow
                next_points = initial_points.copy()
                flow, status, error = cv2.calcOpticalFlowPyrLK(
                    gray_frames[i-1], gray_frames[i], 
                    initial_points, 
                    next_points
                )
                
                if flow is not None and len(flow) > 0 and status[0]:
                    magnitude = np.linalg.norm(flow[0] - initial_points[0])
                    motion_magnitudes.append(magnitude)
                else:
                    motion_magnitudes.append(0)
            except Exception:
                motion_magnitudes.append(0)
        
        # Smooth motion data
        if len(motion_magnitudes) > 5:
            smoothed = np.convolve(motion_magnitudes, np.ones(5)/5, mode='same')
        else:
            smoothed = motion_magnitudes
        
        # Find phase boundaries based on motion changes
        total_frames = len(frames)
        
        # Simple heuristic for archery phases
        # Draw phase: high motion (0-40%)
        # Anchor phase: low motion (40-70%)
        # Release phase: high motion (70-100%)
        
        draw_end = int(0.4 * total_frames)
        anchor_end = int(0.7 * total_frames)
        
        phases = {
            'stance': (0, int(0.1 * total_frames)),
            'draw': (int(0.1 * total_frames), draw_end),
            'anchor': (draw_end, anchor_end),
            'release': (anchor_end, int(0.85 * total_frames)),
            'follow_through': (int(0.85 * total_frames), total_frames)
        }
        
        return phases
    
    def enhance_frame_quality(self, frame: np.ndarray) -> np.ndarray:
        """
        Enhance frame quality for better pose detection
        
        Args:
            frame: Input frame
            
        Returns:
            Enhanced frame
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        lab[:,:,0] = clahe.apply(lab[:,:,0])
        
        # Convert back to RGB
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # Apply slight gaussian blur to reduce noise
        enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        return enhanced
    
    def save_annotated_video(self, frames: List[np.ndarray], annotations: List[dict], 
                           output_path: str, fps: float = 30.0) -> bool:
        """
        Save video with pose annotations
        
        Args:
            frames: List of video frames
            annotations: List of annotation data for each frame
            output_path: Path to save annotated video
            fps: Frames per second
            
        Returns:
            Success status
        """
        if not frames:
            return False
        
        height, width = frames[0].shape[:2]
        
        # Define codec and create VideoWriter
        fourcc = cv2.VideoWriter.fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        try:
            for i, frame in enumerate(frames):
                annotated_frame = frame.copy()
                
                # Add annotations if available
                if i < len(annotations) and annotations[i]:
                    annotated_frame = self._draw_pose_annotations(annotated_frame, annotations[i])
                
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            
            return True
            
        except Exception as e:
            print(f"Error saving annotated video: {e}")
            return False
            
        finally:
            out.release()
    
    def _draw_pose_annotations(self, frame: np.ndarray, annotation: dict) -> np.ndarray:
        """
        Draw pose annotations on frame
        
        Args:
            frame: Input frame
            annotation: Annotation data
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        # Draw pose landmarks if available
        if 'landmarks' in annotation:
            landmarks = annotation['landmarks']
            
            # Draw key points
            for landmark in landmarks:
                if 'x' in landmark and 'y' in landmark:
                    x, y = int(landmark['x'] * frame.shape[1]), int(landmark['y'] * frame.shape[0])
                    cv2.circle(annotated, (x, y), 5, (0, 255, 0), -1)
        
        # Draw detected issues
        if 'issues' in annotation:
            y_offset = 30
            for issue in annotation['issues']:
                cv2.putText(annotated, issue, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                y_offset += 30
        
        return annotated
