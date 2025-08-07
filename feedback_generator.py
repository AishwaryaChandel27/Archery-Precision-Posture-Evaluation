from typing import Dict, List, Any
import numpy as np

class FeedbackGenerator:
    """Generates comprehensive feedback and recommendations for archery technique"""
    
    def __init__(self):
        # Define feedback templates and recommendations
        self.feedback_templates = {
            'stance': {
                'narrow_stance': {
                    'description': "Your stance is too narrow, which affects balance and stability",
                    'recommendation': "Widen your stance to shoulder-width apart with feet perpendicular to the target",
                    'drill': "Practice the T-square stance position daily"
                },
                'wide_stance': {
                    'description': "Your stance is too wide, reducing shooting stability",
                    'recommendation': "Narrow your stance to approximately shoulder-width",
                    'drill': "Use floor markers to practice consistent foot placement"
                },
                'body_lean': {
                    'description': "Excessive body lean detected, affecting balance",
                    'recommendation': "Keep your torso upright and centered over your feet",
                    'drill': "Practice against a wall to maintain proper posture"
                },
                'shoulder_alignment': {
                    'description': "Uneven shoulder alignment compromises shooting form",
                    'recommendation': "Level your shoulders and maintain square posture to the target",
                    'drill': "Use a mirror to check shoulder alignment during practice"
                }
            },
            'draw': {
                'inconsistent_path': {
                    'description': "Draw path lacks consistency between shots",
                    'recommendation': "Focus on a smooth, straight draw path every time",
                    'drill': "Practice slow-motion draw exercises with emphasis on form"
                },
                'speed_variation': {
                    'description': "Draw speed varies significantly between shots",
                    'recommendation': "Maintain steady, controlled draw speed throughout",
                    'drill': "Count to 3 during each draw phase for timing consistency"
                },
                'low_elbow': {
                    'description': "Drawing elbow is positioned too low",
                    'recommendation': "Raise your drawing elbow to shoulder height",
                    'drill': "Practice drawing with elbow at shoulder level using a mirror"
                }
            },
            'anchor': {
                'inconsistent_anchor': {
                    'description': "Anchor point varies between shots",
                    'recommendation': "Find a consistent anchor point and use it every time",
                    'drill': "Practice finding your anchor point with eyes closed"
                },
                'head_movement': {
                    'description': "Head moves during the anchor and aiming phase",
                    'recommendation': "Keep your head still and stable during aiming",
                    'drill': "Practice aiming while balancing an object on your head"
                },
                'bow_arm_collapse': {
                    'description': "Bow arm is not fully extended",
                    'recommendation': "Maintain firm, extended bow arm throughout the shot",
                    'drill': "Practice bow arm strength exercises and extension drills"
                }
            },
            'release': {
                'jerky_release': {
                    'description': "Release motion is jerky or forced",
                    'recommendation': "Focus on smooth, relaxed finger release",
                    'drill': "Practice release exercises with a resistance band"
                },
                'bow_hand_movement': {
                    'description': "Bow hand moves during release",
                    'recommendation': "Keep bow hand steady and relaxed through release",
                    'drill': "Practice dead release with bow hand follow-through"
                }
            },
            'follow_through': {
                'short_follow_through': {
                    'description': "Follow-through is too brief",
                    'recommendation': "Hold your form for 2-3 seconds after release",
                    'drill': "Practice counting to 3 after each release"
                },
                'head_drop': {
                    'description': "Head moves immediately after release",
                    'recommendation': "Keep your head position stable until arrow hits target",
                    'drill': "Focus on watching arrow flight without moving your head"
                }
            }
        }
        
        # Define strength recognition patterns
        self.strength_patterns = {
            'excellent_stance': "Maintains excellent stance stability and alignment",
            'smooth_draw': "Demonstrates smooth and consistent draw technique",
            'stable_anchor': "Shows excellent anchor point consistency",
            'clean_release': "Executes smooth and controlled release",
            'good_follow_through': "Maintains proper follow-through form"
        }
        
        # Priority scoring for issues
        self.issue_priorities = {
            'stance': {'safety': 3, 'accuracy': 2, 'consistency': 2},
            'draw': {'safety': 2, 'accuracy': 3, 'consistency': 3},
            'anchor': {'safety': 1, 'accuracy': 3, 'consistency': 3},
            'release': {'safety': 2, 'accuracy': 3, 'consistency': 2},
            'follow_through': {'safety': 1, 'accuracy': 2, 'consistency': 2}
        }
    
    def generate_comprehensive_feedback(self, biomech_results: Dict) -> Dict[str, Any]:
        """
        Generate comprehensive feedback based on biomechanical analysis
        
        Args:
            biomech_results: Results from biomechanical analysis
            
        Returns:
            Comprehensive feedback dictionary
        """
        feedback = {
            'priority_issues': [],
            'improvements': [],
            'strengths': [],
            'overall_assessment': '',
            'training_plan': []
        }
        
        # Analyze each phase and generate feedback
        for phase_name, phase_data in biomech_results.items():
            if hasattr(phase_data, 'issues') and phase_data.issues:
                # Process issues for this phase
                phase_feedback = self._generate_phase_feedback(phase_name, phase_data)
                
                # Categorize by priority
                priority_issues = self._categorize_issues_by_priority(phase_name, phase_feedback)
                feedback['priority_issues'].extend(priority_issues)
                
                # Add improvement suggestions
                improvements = self._generate_improvement_suggestions(phase_name, phase_data)
                feedback['improvements'].extend(improvements)
            
            # Identify strengths
            strengths = self._identify_strengths(phase_name, phase_data)
            feedback['strengths'].extend(strengths)
        
        # Generate overall assessment
        feedback['overall_assessment'] = self._generate_overall_assessment(biomech_results)
        
        # Create training plan
        feedback['training_plan'] = self._create_training_plan(feedback['priority_issues'])
        
        # Sort issues by priority
        feedback['priority_issues'] = sorted(feedback['priority_issues'], 
                                           key=lambda x: x['priority_score'], reverse=True)
        
        return feedback
    
    def _generate_phase_feedback(self, phase_name: str, phase_data: Any) -> List[Dict]:
        """Generate feedback for a specific phase"""
        phase_feedback = []
        
        if not hasattr(phase_data, 'issues') or not phase_data.issues:
            return phase_feedback
        
        # Map detected issues to feedback templates
        issue_mapping = {
            'stance': {
                'Stance too narrow': 'narrow_stance',
                'Stance too wide': 'wide_stance',
                'Excessive body lean': 'body_lean',
                'Uneven shoulder alignment': 'shoulder_alignment'
            },
            'draw': {
                'Inconsistent draw path': 'inconsistent_path',
                'Inconsistent draw speed': 'speed_variation',
                'Drawing elbow too low': 'low_elbow'
            },
            'anchor': {
                'Inconsistent anchor point': 'inconsistent_anchor',
                'Head movement during anchor': 'head_movement',
                'Bow arm not fully extended': 'bow_arm_collapse'
            },
            'release': {
                'Jerky or inconsistent release': 'jerky_release',
                'Bow hand movement during release': 'bow_hand_movement'
            },
            'follow_through': {
                'Follow-through too short': 'short_follow_through',
                'Head movement during follow-through': 'head_drop'
            }
        }
        
        phase_mapping = issue_mapping.get(phase_name, {})
        
        for issue in phase_data.issues:
            # Find matching template
            template_key = None
            for issue_pattern, template in phase_mapping.items():
                if issue_pattern.lower() in issue.lower():
                    template_key = template
                    break
            
            if template_key and phase_name in self.feedback_templates:
                template = self.feedback_templates[phase_name].get(template_key)
                if template:
                    feedback_item = {
                        'phase': phase_name,
                        'issue': issue,
                        'description': template['description'],
                        'recommendation': template['recommendation'],
                        'drill': template['drill']
                    }
                    phase_feedback.append(feedback_item)
        
        return phase_feedback
    
    def _categorize_issues_by_priority(self, phase_name: str, phase_feedback: List[Dict]) -> List[Dict]:
        """Categorize issues by priority level"""
        priority_issues = []
        
        for feedback_item in phase_feedback:
            # Calculate priority score
            base_priorities = self.issue_priorities.get(phase_name, {'safety': 1, 'accuracy': 1, 'consistency': 1})
            priority_score = sum(base_priorities.values())
            
            # Add safety multiplier for certain issues
            if 'safety' in feedback_item['description'].lower():
                priority_score *= 1.5
            
            priority_issues.append({
                'phase': phase_name,
                'description': feedback_item['description'],
                'recommendation': feedback_item['recommendation'],
                'drill': feedback_item['drill'],
                'priority_score': priority_score
            })
        
        return priority_issues
    
    def _generate_improvement_suggestions(self, phase_name: str, phase_data: Any) -> List[Dict]:
        """Generate general improvement suggestions"""
        improvements = []
        
        if not hasattr(phase_data, 'score'):
            return improvements
        
        score = phase_data.score
        
        # General improvement suggestions based on score
        if score < 60:
            improvements.append({
                'phase': phase_name,
                'description': f"Significant improvement needed in {phase_name} technique",
                'suggestion': "Focus on fundamental form and consider working with a coach"
            })
        elif score < 80:
            improvements.append({
                'phase': phase_name,
                'description': f"Good foundation in {phase_name}, but refinement needed",
                'suggestion': "Practice specific drills to improve consistency"
            })
        
        # Phase-specific suggestions
        phase_suggestions = {
            'stance': "Work on balance and posture exercises",
            'draw': "Focus on smooth, controlled movement patterns",
            'anchor': "Develop muscle memory for consistent anchor point",
            'release': "Practice relaxation and timing techniques",
            'follow_through': "Develop patience and form holding"
        }
        
        if score < 85 and phase_name in phase_suggestions:
            improvements.append({
                'phase': phase_name,
                'description': f"Consider additional {phase_name} training",
                'suggestion': phase_suggestions[phase_name]
            })
        
        return improvements
    
    def _identify_strengths(self, phase_name: str, phase_data: Any) -> List[Dict]:
        """Identify technique strengths"""
        strengths = []
        
        if not hasattr(phase_data, 'score'):
            return strengths
        
        score = phase_data.score
        
        # Identify strengths based on scores
        if score >= 90:
            strengths.append({
                'phase': phase_name,
                'description': f"Excellent {phase_name} technique demonstrated"
            })
        elif score >= 80:
            strengths.append({
                'phase': phase_name,
                'description': f"Good {phase_name} form with minor areas for refinement"
            })
        
        # Specific strength patterns
        if hasattr(phase_data, 'metrics'):
            metrics = phase_data.metrics
            
            # Check for specific strong metrics
            if phase_name == 'stance' and 'average_spine_angle' in metrics:
                if metrics['average_spine_angle'] < 5:
                    strengths.append({
                        'phase': phase_name,
                        'description': "Excellent posture and body alignment"
                    })
            
            if phase_name == 'anchor' and 'anchor_point_consistency' in metrics:
                if metrics['anchor_point_consistency'] > 0.95:
                    strengths.append({
                        'phase': phase_name,
                        'description': "Outstanding anchor point consistency"
                    })
        
        return strengths
    
    def _generate_overall_assessment(self, biomech_results: Dict) -> str:
        """Generate overall assessment of technique"""
        total_score = 0
        phase_count = 0
        
        # Calculate overall score
        for phase_data in biomech_results.values():
            if hasattr(phase_data, 'score'):
                total_score += phase_data.score
                phase_count += 1
        
        if phase_count == 0:
            return "Unable to generate assessment due to insufficient data"
        
        overall_score = total_score / phase_count
        
        # Generate assessment based on overall score
        if overall_score >= 90:
            assessment = "Excellent archery technique with only minor refinements needed. "
            assessment += "Focus on maintaining consistency and consider advanced training techniques."
        elif overall_score >= 80:
            assessment = "Good archery form with solid fundamentals. "
            assessment += "Continue practicing identified areas for improvement to reach advanced level."
        elif overall_score >= 70:
            assessment = "Developing archery technique with good potential. "
            assessment += "Focus on fundamental corrections and consistent practice."
        elif overall_score >= 60:
            assessment = "Basic archery form established but needs significant improvement. "
            assessment += "Consider working with a coach to address fundamental issues."
        else:
            assessment = "Archery technique needs substantial work on fundamentals. "
            assessment += "Recommend focusing on basic form and safety before advancing."
        
        assessment += f" Overall technique score: {overall_score:.1f}/100"
        
        return assessment
    
    def _create_training_plan(self, priority_issues: List[Dict]) -> List[Dict]:
        """Create a structured training plan based on identified issues"""
        training_plan = []
        
        # Group issues by phase
        phase_issues = {}
        for issue in priority_issues[:5]:  # Top 5 priority issues
            phase = issue['phase']
            if phase not in phase_issues:
                phase_issues[phase] = []
            phase_issues[phase].append(issue)
        
        # Create weekly training plan
        week_plan = {
            'Week 1-2': 'Foundation Building',
            'Week 3-4': 'Consistency Development',
            'Week 5-6': 'Refinement and Integration'
        }
        
        for week, focus in week_plan.items():
            week_activities = []
            
            # Add phase-specific activities
            for phase, issues in phase_issues.items():
                for issue in issues:
                    activity = {
                        'focus_area': phase,
                        'activity': issue['drill'],
                        'goal': issue['recommendation'],
                        'frequency': '3-4 times per week',
                        'duration': '15-20 minutes'
                    }
                    week_activities.append(activity)
            
            if week_activities:
                training_plan.append({
                    'period': week,
                    'focus': focus,
                    'activities': week_activities[:3]  # Limit to 3 activities per week
                })
        
        return training_plan
    
    def generate_drill_recommendations(self, phase_name: str, specific_issues: List[str]) -> List[Dict]:
        """Generate specific drill recommendations for a phase"""
        drills = []
        
        # Phase-specific drill database
        drill_database = {
            'stance': [
                {
                    'name': 'Mirror Alignment Check',
                    'description': 'Practice stance in front of mirror to check alignment',
                    'duration': '5 minutes daily',
                    'equipment': 'Mirror'
                },
                {
                    'name': 'Balance Board Training',
                    'description': 'Practice stance on balance board to improve stability',
                    'duration': '10 minutes, 3x per week',
                    'equipment': 'Balance board'
                }
            ],
            'draw': [
                {
                    'name': 'Slow Motion Draw',
                    'description': 'Practice draw in slow motion focusing on path consistency',
                    'duration': '15 minutes daily',
                    'equipment': 'Bow or rubber band'
                },
                {
                    'name': 'Wall Draw Exercise',
                    'description': 'Practice draw against wall to ensure straight path',
                    'duration': '10 minutes daily',
                    'equipment': 'Wall space'
                }
            ]
        }
        
        if phase_name in drill_database:
            drills = drill_database[phase_name]
        
        return drills
