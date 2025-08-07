import os
import tempfile
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import math

def create_temp_file(suffix: str = '.mp4') -> str:
    """Create a temporary file and return its path"""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_file.close()
    return temp_file.name

def cleanup_temp_file(file_path: str) -> None:
    """Clean up temporary file"""
    try:
        if os.path.exists(file_path):
            os.unlink(file_path)
    except Exception as e:
        print(f"Error cleaning up temp file {file_path}: {e}")

def calculate_distance_2d(point1: Dict[str, float], point2: Dict[str, float]) -> float:
    """Calculate 2D distance between two points"""
    dx = point1['x'] - point2['x']
    dy = point1['y'] - point2['y']
    return math.sqrt(dx*dx + dy*dy)

def calculate_angle_3points(p1: Dict[str, float], p2: Dict[str, float], p3: Dict[str, float]) -> float:
    """Calculate angle at p2 formed by p1-p2-p3"""
    # Convert to numpy arrays
    point1 = np.array([p1['x'], p1['y']])
    point2 = np.array([p2['x'], p2['y']])
    point3 = np.array([p3['x'], p3['y']])
    
    # Calculate vectors
    v1 = point1 - point2
    v2 = point3 - point2
    
    # Calculate angle
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = math.degrees(math.acos(cos_angle))
    
    return angle

def smooth_data(data: List[float], window_size: int = 5) -> List[float]:
    """Apply moving average smoothing to data"""
    if len(data) < window_size:
        return data
    
    smoothed = []
    for i in range(len(data)):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(data), i + window_size // 2 + 1)
        window_data = data[start_idx:end_idx]
        smoothed.append(sum(window_data) / len(window_data))
    
    return smoothed

def normalize_coordinates(landmarks: Dict[str, Dict], frame_width: int, frame_height: int) -> Dict[str, Dict]:
    """Normalize landmark coordinates to 0-1 range"""
    normalized = {}
    
    for name, landmark in landmarks.items():
        normalized[name] = {
            'x': landmark['pixel_x'] / frame_width,
            'y': landmark['pixel_y'] / frame_height,
            'visibility': landmark.get('visibility', 1.0)
        }
    
    return normalized

def calculate_frame_similarity(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """Calculate similarity between two frames using histogram comparison"""
    # Convert to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY) if len(frame1.shape) == 3 else frame1
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY) if len(frame2.shape) == 3 else frame2
    
    # Calculate histograms
    hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
    
    # Compare histograms
    similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    
    return similarity

def detect_motion_intensity(frames: List[np.ndarray]) -> List[float]:
    """Detect motion intensity across frames"""
    if len(frames) < 2:
        return [0.0] * len(frames)
    
    motion_intensities = [0.0]  # First frame has no motion
    
    for i in range(1, len(frames)):
        # Convert frames to grayscale
        gray1 = cv2.cvtColor(frames[i-1], cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
        
        # Calculate frame difference
        diff = cv2.absdiff(gray1, gray2)
        
        # Calculate motion intensity as mean of differences
        intensity = np.mean(diff) / 255.0  # Normalize to 0-1
        motion_intensities.append(intensity)
    
    return motion_intensities

def find_keyframes(frames: List[np.ndarray], num_keyframes: int = 10) -> List[int]:
    """Find key frames based on motion and content"""
    if len(frames) <= num_keyframes:
        return list(range(len(frames)))
    
    # Calculate motion intensities
    motion_intensities = detect_motion_intensity(frames)
    
    # Find frames with significant motion changes
    motion_changes = []
    for i in range(1, len(motion_intensities)):
        change = abs(motion_intensities[i] - motion_intensities[i-1])
        motion_changes.append((i, change))
    
    # Sort by motion change magnitude
    motion_changes.sort(key=lambda x: x[1], reverse=True)
    
    # Select top frames with motion changes
    keyframe_indices = [0]  # Always include first frame
    for idx, _ in motion_changes[:num_keyframes-2]:
        keyframe_indices.append(idx)
    
    keyframe_indices.append(len(frames)-1)  # Always include last frame
    
    # Sort and remove duplicates
    keyframe_indices = sorted(list(set(keyframe_indices)))
    
    return keyframe_indices[:num_keyframes]

def calculate_pose_stability(pose_data: List[Dict]) -> float:
    """Calculate overall pose stability across frames"""
    if len(pose_data) < 2:
        return 1.0
    
    stability_scores = []
    
    # Check stability of key landmarks
    key_landmarks = ['nose', 'left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
    
    for landmark_name in key_landmarks:
        positions = []
        for frame_data in pose_data:
            if (frame_data.get('pose_detected', False) and 
                'landmarks' in frame_data and 
                landmark_name in frame_data['landmarks']):
                landmark = frame_data['landmarks'][landmark_name]
                positions.append((landmark['x'], landmark['y']))
        
        if len(positions) > 1:
            # Calculate position variance
            x_coords = [pos[0] for pos in positions]
            y_coords = [pos[1] for pos in positions]
            
            x_var = np.var(x_coords)
            y_var = np.var(y_coords)
            
            # Convert variance to stability score (lower variance = higher stability)
            stability = max(0, 1 - (x_var + y_var) * 10)
            stability_scores.append(stability)
    
    return np.mean(stability_scores) if stability_scores else 0.0

def format_analysis_results(results: Dict[str, Any]) -> str:
    """Format analysis results for display"""
    formatted = "ARCHERY TECHNIQUE ANALYSIS\n"
    formatted += "=" * 50 + "\n\n"
    
    # Overall assessment
    if 'overall_assessment' in results:
        formatted += "OVERALL ASSESSMENT:\n"
        formatted += results['overall_assessment'] + "\n\n"
    
    # Priority issues
    if 'priority_issues' in results and results['priority_issues']:
        formatted += "PRIORITY ISSUES TO ADDRESS:\n"
        formatted += "-" * 30 + "\n"
        for i, issue in enumerate(results['priority_issues'][:5], 1):
            formatted += f"{i}. {issue['phase'].upper()}: {issue['description']}\n"
            formatted += f"   Recommendation: {issue['recommendation']}\n\n"
    
    # Strengths
    if 'strengths' in results and results['strengths']:
        formatted += "TECHNIQUE STRENGTHS:\n"
        formatted += "-" * 20 + "\n"
        for strength in results['strengths']:
            formatted += f"• {strength['phase'].upper()}: {strength['description']}\n"
        formatted += "\n"
    
    return formatted

def validate_video_format(file_path: str) -> bool:
    """Validate if file is a supported video format"""
    supported_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
    
    # Check file extension
    _, ext = os.path.splitext(file_path.lower())
    if ext not in supported_extensions:
        return False
    
    # Try to open with OpenCV
    try:
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            return False
        
        # Read first frame to verify
        ret, frame = cap.read()
        cap.release()
        
        return ret and frame is not None
    
    except Exception:
        return False

def resize_frame_aspect_ratio(frame: np.ndarray, max_width: int = 640, max_height: int = 480) -> np.ndarray:
    """Resize frame while maintaining aspect ratio"""
    height, width = frame.shape[:2]
    
    # Calculate scaling factor
    scale_w = max_width / width
    scale_h = max_height / height
    scale = min(scale_w, scale_h)
    
    # Calculate new dimensions
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    # Resize frame
    resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    return resized

def create_progress_callback(progress_bar, status_text, total_steps: int):
    """Create a progress callback function"""
    current_step = 0
    
    def update_progress(step_description: str):
        nonlocal current_step
        current_step += 1
        progress = min(current_step / total_steps, 1.0)
        
        if progress_bar:
            progress_bar.progress(progress)
        if status_text:
            status_text.text(step_description)
    
    return update_progress

def extract_video_metadata(file_path: str) -> Dict[str, Any]:
    """Extract metadata from video file"""
    metadata = {
        'duration': 0,
        'fps': 0,
        'frame_count': 0,
        'width': 0,
        'height': 0,
        'format': 'unknown'
    }
    
    try:
        cap = cv2.VideoCapture(file_path)
        
        if cap.isOpened():
            metadata['fps'] = cap.get(cv2.CAP_PROP_FPS)
            metadata['frame_count'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            metadata['width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            metadata['height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if metadata['fps'] > 0:
                metadata['duration'] = metadata['frame_count'] / metadata['fps']
            
            # Get format from file extension
            _, ext = os.path.splitext(file_path)
            metadata['format'] = ext.lower()
        
        cap.release()
    
    except Exception as e:
        print(f"Error extracting video metadata: {e}")
    
    return metadata
