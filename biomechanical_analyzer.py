import numpy as np
import math
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass

@dataclass
class PhaseAnalysis:
    """Data structure for phase analysis results"""
    phase_name: str
    score: float
    issues: List[str]
    recommendations: List[str]
    metrics: Dict[str, Any]

class BiomechanicalAnalyzer:
    """Analyzes biomechanical aspects of archery technique"""
    
    def __init__(self):
        # Define optimal ranges for various measurements
        self.optimal_ranges = {
            'stance_width': (0.15, 0.25),  # Relative to body width
            'shoulder_alignment': (0.0, 0.02),  # Shoulder level difference
            'spine_angle': (0, 10),  # Degrees from vertical
            'bow_arm_angle': (170, 180),  # Elbow extension
            'draw_length_consistency': 0.95,  # Minimum consistency ratio
            'anchor_point_consistency': 0.90,  # Minimum consistency
            'release_smoothness': 0.85,  # Minimum smoothness score
            'follow_through_duration': (0.5, 1.5)  # Seconds
        }
        
        # Phase-specific weights for scoring
        self.phase_weights = {
            'stance': {'posture': 0.4, 'balance': 0.3, 'alignment': 0.3},
            'draw': {'path': 0.3, 'symmetry': 0.3, 'speed': 0.2, 'form': 0.2},
            'anchor': {'consistency': 0.4, 'stability': 0.3, 'alignment': 0.3},
            'release': {'smoothness': 0.4, 'timing': 0.3, 'follow_through': 0.3},
            'follow_through': {'stability': 0.4, 'duration': 0.3, 'form': 0.3}
        }
    
    def analyze_archery_phases(self, pose_data: List[Dict], fps: float) -> Dict[str, PhaseAnalysis]:
        """
        Comprehensive analysis of all archery phases
        
        Args:
            pose_data: List of pose analysis data for each frame
            fps: Video frame rate
            
        Returns:
            Dictionary of phase analysis results
        """
        if not pose_data:
            return {}
        
        # Detect phase boundaries
        phases = self._detect_phase_boundaries(pose_data, fps)
        
        results = {}
        
        # Analyze each phase
        for phase_name, (start_frame, end_frame) in phases.items():
            phase_data = pose_data[start_frame:end_frame] if end_frame <= len(pose_data) else pose_data[start_frame:]
            
            if phase_name == 'stance':
                results[phase_name] = self._analyze_stance_phase(phase_data)
            elif phase_name == 'draw':
                results[phase_name] = self._analyze_draw_phase(phase_data, fps)
            elif phase_name == 'anchor':
                results[phase_name] = self._analyze_anchor_phase(phase_data)
            elif phase_name == 'release':
                results[phase_name] = self._analyze_release_phase(phase_data, fps)
            elif phase_name == 'follow_through':
                results[phase_name] = self._analyze_follow_through_phase(phase_data, fps)
        
        return results
    
    def _detect_phase_boundaries(self, pose_data: List[Dict], fps: float) -> Dict[str, Tuple[int, int]]:
        """Detect phase boundaries in the archery sequence"""
        total_frames = len(pose_data)
        
        # Extract motion data
        motion_data = []
        for i in range(len(pose_data)):
            if pose_data[i].get('pose_detected', False):
                draw_length = pose_data[i].get('draw_length', 0)
                motion_data.append(draw_length)
            else:
                motion_data.append(0)
        
        if len(motion_data) < 10:
            # Fallback to simple time-based division
            return {
                'stance': (0, int(0.1 * total_frames)),
                'draw': (int(0.1 * total_frames), int(0.5 * total_frames)),
                'anchor': (int(0.5 * total_frames), int(0.7 * total_frames)),
                'release': (int(0.7 * total_frames), int(0.8 * total_frames)),
                'follow_through': (int(0.8 * total_frames), total_frames)
            }
        
        # Smooth motion data
        smoothed_motion = np.convolve(motion_data, np.ones(5)/5, mode='same')
        
        # Find draw length peaks and valleys
        draw_start = self._find_motion_start(smoothed_motion.tolist())
        anchor_start = self._find_anchor_phase(smoothed_motion.tolist())
        release_start = self._find_release_point(smoothed_motion.tolist())
        
        # Define phase boundaries
        phases = {
            'stance': (0, draw_start),
            'draw': (draw_start, anchor_start),
            'anchor': (anchor_start, release_start),
            'release': (release_start, min(release_start + int(0.5 * fps), total_frames)),
            'follow_through': (min(release_start + int(0.5 * fps), total_frames), total_frames)
        }
        
        return phases
    
    def _find_motion_start(self, motion_data: List[float]) -> int:
        """Find when significant motion begins (draw start)"""
        if len(motion_data) < 5:
            return 0
        
        baseline = np.mean(motion_data[:5])
        threshold = baseline + 0.02  # Small threshold above baseline
        
        for i in range(5, len(motion_data)):
            if motion_data[i] > threshold:
                return max(0, i - 2)  # Start slightly before motion detected
        
        return int(0.1 * len(motion_data))  # Fallback
    
    def _find_anchor_phase(self, motion_data: List[float]) -> int:
        """Find when anchor phase begins (draw length stabilizes)"""
        if len(motion_data) < 10:
            return int(0.5 * len(motion_data))
        
        # Find peak draw length
        max_draw = max(motion_data)
        anchor_threshold = 0.9 * max_draw
        
        # Find first time we reach near-maximum draw
        for i in range(len(motion_data)):
            if motion_data[i] >= anchor_threshold:
                return i
        
        return int(0.5 * len(motion_data))  # Fallback
    
    def _find_release_point(self, motion_data: List[float]) -> int:
        """Find release point (sudden drop in draw length)"""
        if len(motion_data) < 10:
            return int(0.7 * len(motion_data))
        
        # Find maximum draw length point
        max_idx = np.argmax(motion_data)
        
        # Look for sudden drop after maximum
        for i in range(max_idx + 1, len(motion_data)):
            if i + 3 < len(motion_data):
                # Check for sustained decrease
                recent_avg = np.mean(motion_data[i:i+3])
                if recent_avg < 0.7 * motion_data[max_idx]:
                    return i
        
        return int(0.7 * len(motion_data))  # Fallback
    
    def _analyze_stance_phase(self, phase_data: List[Dict]) -> PhaseAnalysis:
        """Analyze stance and posture phase"""
        issues = []
        recommendations = []
        metrics = {}
        
        # Extract stance metrics
        stance_widths = []
        spine_angles = []
        shoulder_alignments = []
        
        for frame in phase_data:
            if frame.get('pose_detected', False):
                positions = frame.get('positions', {})
                angles = frame.get('angles', {})
                
                if 'stance_width' in positions:
                    stance_widths.append(positions['stance_width'])
                
                if 'spine_angle' in angles:
                    spine_angles.append(angles['spine_angle'])
                
                if 'shoulder_level_difference' in positions:
                    shoulder_alignments.append(positions['shoulder_level_difference'])
        
        # Calculate metrics
        if stance_widths:
            avg_stance_width = np.mean(stance_widths)
            metrics['average_stance_width'] = avg_stance_width
            
            optimal_min, optimal_max = self.optimal_ranges['stance_width']
            if avg_stance_width < optimal_min:
                issues.append("Stance too narrow")
                recommendations.append("Widen your stance to shoulder-width apart")
            elif avg_stance_width > optimal_max:
                issues.append("Stance too wide")
                recommendations.append("Narrow your stance slightly")
        
        if spine_angles:
            avg_spine_angle = np.mean(spine_angles)
            metrics['average_spine_angle'] = avg_spine_angle
            
            if avg_spine_angle > self.optimal_ranges['spine_angle'][1]:
                issues.append("Excessive body lean")
                recommendations.append("Keep your torso more upright and centered")
        
        if shoulder_alignments:
            avg_shoulder_alignment = np.mean(shoulder_alignments)
            metrics['shoulder_level_difference'] = avg_shoulder_alignment
            
            if avg_shoulder_alignment > self.optimal_ranges['shoulder_alignment'][1]:
                issues.append("Uneven shoulder alignment")
                recommendations.append("Level your shoulders and maintain square posture")
        
        # Calculate score
        score = self._calculate_stance_score(metrics)
        
        return PhaseAnalysis(
            phase_name="Stance & Posture",
            score=score,
            issues=issues,
            recommendations=recommendations,
            metrics=metrics
        )
    
    def _analyze_draw_phase(self, phase_data: List[Dict], fps: float) -> PhaseAnalysis:
        """Analyze draw phase biomechanics"""
        issues = []
        recommendations = []
        metrics = {}
        
        # Extract draw metrics
        draw_lengths = []
        elbow_angles = []
        shoulder_angles = []
        draw_speeds = []
        
        for frame in phase_data:
            if frame.get('pose_detected', False):
                if 'draw_length' in frame:
                    draw_lengths.append(frame['draw_length'])
                
                angles = frame.get('angles', {})
                if 'right_elbow_angle' in angles:  # Assuming right-handed
                    elbow_angles.append(angles['right_elbow_angle'])
                if 'right_shoulder_angle' in angles:
                    shoulder_angles.append(angles['right_shoulder_angle'])
        
        # Calculate draw speed
        if len(draw_lengths) > 1:
            for i in range(1, len(draw_lengths)):
                speed = (draw_lengths[i] - draw_lengths[i-1]) * fps
                draw_speeds.append(abs(speed))
        
        # Analyze draw path consistency
        if draw_lengths:
            draw_consistency = self._calculate_consistency(draw_lengths)
            metrics['draw_path_consistency'] = draw_consistency
            
            if draw_consistency < 0.8:
                issues.append("Inconsistent draw path")
                recommendations.append("Focus on smooth, consistent draw motion")
        
        # Analyze draw speed
        if draw_speeds:
            avg_speed = np.mean(draw_speeds)
            speed_variation = np.std(draw_speeds)
            metrics['average_draw_speed'] = avg_speed
            metrics['draw_speed_variation'] = speed_variation
            
            if speed_variation > avg_speed * 0.5:
                issues.append("Inconsistent draw speed")
                recommendations.append("Maintain steady, controlled draw speed")
        
        # Analyze elbow positioning
        if elbow_angles:
            avg_elbow_angle = np.mean(elbow_angles)
            metrics['average_drawing_elbow_angle'] = avg_elbow_angle
            
            if avg_elbow_angle < 120:
                issues.append("Drawing elbow too low")
                recommendations.append("Raise your drawing elbow to shoulder height")
        
        score = self._calculate_draw_score(metrics)
        
        return PhaseAnalysis(
            phase_name="Draw Phase",
            score=score,
            issues=issues,
            recommendations=recommendations,
            metrics=metrics
        )
    
    def _analyze_anchor_phase(self, phase_data: List[Dict]) -> PhaseAnalysis:
        """Analyze anchor and aiming phase"""
        issues = []
        recommendations = []
        metrics = {}
        
        # Extract anchor metrics
        anchor_positions = []
        head_positions = []
        bow_arm_angles = []
        
        for frame in phase_data:
            if frame.get('pose_detected', False):
                if 'head_to_string_distance' in frame:
                    anchor_positions.append(frame['head_to_string_distance'])
                
                landmarks = frame.get('landmarks', {})
                if 'nose' in landmarks:
                    head_positions.append((landmarks['nose']['x'], landmarks['nose']['y']))
                
                angles = frame.get('angles', {})
                if 'left_elbow_angle' in angles:  # Bow arm
                    bow_arm_angles.append(angles['left_elbow_angle'])
        
        # Analyze anchor point consistency
        if anchor_positions:
            anchor_consistency = self._calculate_consistency(anchor_positions)
            metrics['anchor_point_consistency'] = anchor_consistency
            
            if anchor_consistency < self.optimal_ranges['anchor_point_consistency']:
                issues.append("Inconsistent anchor point")
                recommendations.append("Practice finding the same anchor point every time")
        
        # Analyze head stability
        if head_positions:
            head_stability = self._calculate_position_stability(head_positions)
            metrics['head_stability'] = head_stability
            
            if head_stability < 0.85:
                issues.append("Head movement during anchor")
                recommendations.append("Keep your head still during aiming")
        
        # Analyze bow arm extension
        if bow_arm_angles:
            avg_bow_arm_angle = np.mean(bow_arm_angles)
            metrics['bow_arm_extension'] = avg_bow_arm_angle
            
            optimal_min, optimal_max = self.optimal_ranges['bow_arm_angle']
            if avg_bow_arm_angle < optimal_min:
                issues.append("Bow arm not fully extended")
                recommendations.append("Extend your bow arm more fully")
        
        score = self._calculate_anchor_score(metrics)
        
        return PhaseAnalysis(
            phase_name="Anchor & Aiming",
            score=score,
            issues=issues,
            recommendations=recommendations,
            metrics=metrics
        )
    
    def _analyze_release_phase(self, phase_data: List[Dict], fps: float) -> PhaseAnalysis:
        """Analyze release execution"""
        issues = []
        recommendations = []
        metrics = {}
        
        # Extract release metrics
        string_hand_positions = []
        bow_hand_positions = []
        release_speeds = []
        
        for frame in phase_data:
            if frame.get('pose_detected', False):
                landmarks = frame.get('landmarks', {})
                
                if 'right_wrist' in landmarks:  # String hand
                    pos = landmarks['right_wrist']
                    string_hand_positions.append((pos['x'], pos['y']))
                
                if 'left_wrist' in landmarks:  # Bow hand
                    pos = landmarks['left_wrist']
                    bow_hand_positions.append((pos['x'], pos['y']))
        
        # Analyze release smoothness
        if string_hand_positions:
            release_smoothness = self._calculate_release_smoothness(string_hand_positions)
            metrics['release_smoothness'] = release_smoothness
            
            if release_smoothness < self.optimal_ranges['release_smoothness']:
                issues.append("Jerky or inconsistent release")
                recommendations.append("Focus on smooth, relaxed finger release")
        
        # Analyze bow hand stability
        if bow_hand_positions:
            bow_hand_stability = self._calculate_position_stability(bow_hand_positions)
            metrics['bow_hand_stability'] = bow_hand_stability
            
            if bow_hand_stability < 0.8:
                issues.append("Bow hand movement during release")
                recommendations.append("Keep bow hand steady through release")
        
        score = self._calculate_release_score(metrics)
        
        return PhaseAnalysis(
            phase_name="Release",
            score=score,
            issues=issues,
            recommendations=recommendations,
            metrics=metrics
        )
    
    def _analyze_follow_through_phase(self, phase_data: List[Dict], fps: float) -> PhaseAnalysis:
        """Analyze follow-through execution"""
        issues = []
        recommendations = []
        metrics = {}
        
        # Calculate follow-through duration
        duration = len(phase_data) / fps
        metrics['follow_through_duration'] = duration
        
        optimal_min, optimal_max = self.optimal_ranges['follow_through_duration']
        if duration < optimal_min:
            issues.append("Follow-through too short")
            recommendations.append("Hold your form longer after release")
        elif duration > optimal_max:
            issues.append("Follow-through too long")
            recommendations.append("Natural follow-through without forcing it")
        
        # Analyze body stability during follow-through
        head_positions = []
        torso_angles = []
        
        for frame in phase_data:
            if frame.get('pose_detected', False):
                landmarks = frame.get('landmarks', {})
                angles = frame.get('angles', {})
                
                if 'nose' in landmarks:
                    head_positions.append((landmarks['nose']['x'], landmarks['nose']['y']))
                
                if 'spine_angle' in angles:
                    torso_angles.append(angles['spine_angle'])
        
        if head_positions:
            head_stability = self._calculate_position_stability(head_positions)
            metrics['head_stability_follow_through'] = head_stability
            
            if head_stability < 0.8:
                issues.append("Head movement during follow-through")
                recommendations.append("Keep your head position stable after release")
        
        score = self._calculate_follow_through_score(metrics)
        
        return PhaseAnalysis(
            phase_name="Follow-through",
            score=score,
            issues=issues,
            recommendations=recommendations,
            metrics=metrics
        )
    
    def _calculate_consistency(self, values: List[float]) -> float:
        """Calculate consistency score for a series of values"""
        if len(values) < 2:
            return 1.0
        
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        if mean_val == 0:
            return 1.0 if std_val == 0 else 0.0
        
        coefficient_of_variation = std_val / mean_val
        consistency = max(0, 1 - coefficient_of_variation)
        
        return float(consistency)
    
    def _calculate_position_stability(self, positions: List[Tuple[float, float]]) -> float:
        """Calculate stability score for position data"""
        if len(positions) < 2:
            return 1.0
        
        # Calculate movement distances
        movements = []
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            movement = math.sqrt(dx*dx + dy*dy)
            movements.append(movement)
        
        avg_movement = np.mean(movements)
        stability = max(0, 1 - avg_movement * 10)  # Scale factor
        
        return float(stability)
    
    def _calculate_release_smoothness(self, positions: List[Tuple[float, float]]) -> float:
        """Calculate smoothness of release motion"""
        if len(positions) < 3:
            return 1.0
        
        # Calculate acceleration changes (jerk)
        accelerations = []
        for i in range(1, len(positions) - 1):
            # Simple acceleration approximation
            acc_x = positions[i+1][0] - 2*positions[i][0] + positions[i-1][0]
            acc_y = positions[i+1][1] - 2*positions[i][1] + positions[i-1][1]
            acc_magnitude = math.sqrt(acc_x*acc_x + acc_y*acc_y)
            accelerations.append(acc_magnitude)
        
        # Lower acceleration variance indicates smoother motion
        if accelerations:
            smoothness = max(0, 1 - np.std(accelerations) * 100)  # Scale factor
            return float(smoothness)
        
        return 1.0
    
    def _calculate_stance_score(self, metrics: Dict) -> float:
        """Calculate stance phase score"""
        score = 100.0
        
        # Penalize stance width issues
        if 'average_stance_width' in metrics:
            width = metrics['average_stance_width']
            optimal_min, optimal_max = self.optimal_ranges['stance_width']
            if width < optimal_min or width > optimal_max:
                score -= 20
        
        # Penalize spine angle issues
        if 'average_spine_angle' in metrics:
            angle = metrics['average_spine_angle']
            if angle > self.optimal_ranges['spine_angle'][1]:
                score -= 15
        
        # Penalize shoulder alignment issues
        if 'shoulder_level_difference' in metrics:
            diff = metrics['shoulder_level_difference']
            if diff > self.optimal_ranges['shoulder_alignment'][1]:
                score -= 15
        
        return max(0, score)
    
    def _calculate_draw_score(self, metrics: Dict) -> float:
        """Calculate draw phase score"""
        score = 100.0
        
        if 'draw_path_consistency' in metrics:
            consistency = metrics['draw_path_consistency']
            score *= consistency
        
        if 'draw_speed_variation' in metrics and 'average_draw_speed' in metrics:
            speed_var = metrics['draw_speed_variation']
            avg_speed = metrics['average_draw_speed']
            if avg_speed > 0 and speed_var / avg_speed > 0.5:
                score -= 20
        
        return max(0, score)
    
    def _calculate_anchor_score(self, metrics: Dict) -> float:
        """Calculate anchor phase score"""
        score = 100.0
        
        if 'anchor_point_consistency' in metrics:
            consistency = metrics['anchor_point_consistency']
            score *= consistency
        
        if 'head_stability' in metrics:
            stability = metrics['head_stability']
            score *= stability
        
        if 'bow_arm_extension' in metrics:
            extension = metrics['bow_arm_extension']
            optimal_min, optimal_max = self.optimal_ranges['bow_arm_angle']
            if extension < optimal_min:
                score -= 20
        
        return max(0, score)
    
    def _calculate_release_score(self, metrics: Dict) -> float:
        """Calculate release phase score"""
        score = 100.0
        
        if 'release_smoothness' in metrics:
            smoothness = metrics['release_smoothness']
            score *= smoothness
        
        if 'bow_hand_stability' in metrics:
            stability = metrics['bow_hand_stability']
            score *= stability
        
        return max(0, score)
    
    def _calculate_follow_through_score(self, metrics: Dict) -> float:
        """Calculate follow-through phase score"""
        score = 100.0
        
        if 'follow_through_duration' in metrics:
            duration = metrics['follow_through_duration']
            optimal_min, optimal_max = self.optimal_ranges['follow_through_duration']
            if duration < optimal_min or duration > optimal_max:
                score -= 25
        
        if 'head_stability_follow_through' in metrics:
            stability = metrics['head_stability_follow_through']
            score *= stability
        
        return max(0, score)
