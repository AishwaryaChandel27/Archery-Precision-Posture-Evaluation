import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from video_processor import VideoProcessor
from pose_analyzer import PoseAnalyzer
from biomechanical_analyzer import BiomechanicalAnalyzer
from feedback_generator import FeedbackGenerator
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image

# Configure page
st.set_page_config(
    page_title="AI Archery Form Analysis",
    page_icon="🏹",
    layout="wide"
)

# Initialize session state
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

def main():
    st.title("🏹 AI-Powered Archery Form Analysis")
    st.markdown("Upload a video of your archery technique to receive detailed biomechanical feedback and form corrections.")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["Video Analysis", "Analysis Results", "About"])
    
    if page == "Video Analysis":
        video_analysis_page()
    elif page == "Analysis Results":
        results_page()
    else:
        about_page()

def video_analysis_page():
    st.header("Video Upload and Analysis")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=['mp4', 'avi', 'mov', 'mkv', 'wmv'],
        help="Upload a video of archery technique for analysis"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name
        
        # Display video info
        st.success(f"Video uploaded: {uploaded_file.name}")
        
        # Show video
        st.video(uploaded_file)
        
        # Analysis button
        if st.button("🔍 Analyze Archery Form", type="primary"):
            analyze_video(video_path)
        
        # Clean up
        if os.path.exists(video_path):
            os.unlink(video_path)

def analyze_video(video_path):
    """Analyze the uploaded video"""
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Initialize components
        video_processor = VideoProcessor()
        pose_analyzer = PoseAnalyzer()
        biomech_analyzer = BiomechanicalAnalyzer()
        feedback_gen = FeedbackGenerator()
        
        # OpenCV-based pose analysis is always available
        
        # Step 1: Process video
        status_text.text("Processing video...")
        progress_bar.progress(10)
        frames, fps, frame_count = video_processor.extract_frames(video_path)
        
        # Step 2: Analyze poses
        status_text.text("Detecting poses...")
        progress_bar.progress(30)
        pose_data = pose_analyzer.analyze_video_poses(frames)
        
        # Step 3: Biomechanical analysis
        status_text.text("Analyzing biomechanics...")
        progress_bar.progress(60)
        biomech_results = biomech_analyzer.analyze_archery_phases(pose_data, fps)
        
        # Step 4: Generate feedback
        status_text.text("Generating feedback...")
        progress_bar.progress(80)
        feedback = feedback_gen.generate_comprehensive_feedback(biomech_results)
        
        # Step 5: Prepare results
        status_text.text("Preparing results...")
        progress_bar.progress(90)
        
        # Store results in session state
        st.session_state.analysis_results = {
            'pose_data': pose_data,
            'biomech_results': biomech_results,
            'feedback': feedback,
            'fps': fps,
            'frame_count': frame_count
        }
        st.session_state.analysis_complete = True
        
        progress_bar.progress(100)
        status_text.text("Analysis complete!")
        
        st.success("✅ Analysis complete! View results in the 'Analysis Results' tab.")
        
        # Auto-navigate to results
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Error during analysis: {str(e)}")
        progress_bar.empty()
        status_text.empty()

def results_page():
    st.header("Analysis Results")
    
    if not st.session_state.analysis_complete or st.session_state.analysis_results is None:
        st.info("📹 Please upload and analyze a video first.")
        return
    
    results = st.session_state.analysis_results
    
    # Overall score
    overall_score = calculate_overall_score(results['biomech_results'])
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Overall Form Score", f"{overall_score:.1f}/100")
    with col2:
        st.metric("Video FPS", results['fps'])
    with col3:
        st.metric("Total Frames", results['frame_count'])
    
    # Tabs for different analysis aspects
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Phase Analysis", "🎯 Feedback", "📈 Pose Tracking", "📋 Report"])
    
    with tab1:
        display_phase_analysis(results['biomech_results'])
    
    with tab2:
        display_feedback(results['feedback'])
    
    with tab3:
        display_pose_tracking(results['pose_data'])
    
    with tab4:
        display_comprehensive_report(results)

def display_phase_analysis(biomech_results):
    """Display analysis of archery phases"""
    st.subheader("Archery Phase Analysis")
    
    phases = ['stance', 'draw', 'anchor', 'release', 'follow_through']
    phase_names = ['Stance & Posture', 'Draw Phase', 'Anchor & Aiming', 'Release', 'Follow-through']
    
    for phase, phase_name in zip(phases, phase_names):
        if phase in biomech_results:
            with st.expander(f"{phase_name} Analysis", expanded=True):
                phase_data = biomech_results[phase]
                
                # Score
                score = getattr(phase_data, 'score', 0)
                st.metric(f"{phase_name} Score", f"{score:.1f}/100")
                
                # Issues
                issues = getattr(phase_data, 'issues', [])
                if issues:
                    st.write("**Issues Identified:**")
                    for issue in issues:
                        st.write(f"⚠️ {issue}")
                else:
                    st.write("✅ No major issues detected")
                
                # Metrics
                metrics = getattr(phase_data, 'metrics', {})
                if metrics:
                    st.write("**Key Metrics:**")
                    metric_cols = st.columns(min(len(metrics), 4))
                    for i, (metric, value) in enumerate(metrics.items()):
                        with metric_cols[i % 4]:
                            if isinstance(value, (int, float)):
                                st.metric(metric, f"{value:.2f}")
                            else:
                                st.metric(metric, str(value))

def display_feedback(feedback):
    """Display detailed feedback and recommendations"""
    st.subheader("Detailed Feedback & Recommendations")
    
    # Priority issues
    if 'priority_issues' in feedback:
        st.write("### 🚨 Priority Issues to Address")
        for issue in feedback['priority_issues']:
            st.error(f"**{issue['phase']}**: {issue['description']}")
            if 'recommendation' in issue:
                st.write(f"💡 **Recommendation**: {issue['recommendation']}")
        st.divider()
    
    # Strengths
    if 'strengths' in feedback:
        st.write("### ✅ Technique Strengths")
        for strength in feedback['strengths']:
            st.success(f"**{strength['phase']}**: {strength['description']}")
        st.divider()
    
    # Improvement suggestions
    if 'improvements' in feedback:
        st.write("### 📈 Areas for Improvement")
        for improvement in feedback['improvements']:
            st.info(f"**{improvement['phase']}**: {improvement['description']}")
            if 'drill' in improvement:
                st.write(f"🏃 **Suggested Drill**: {improvement['drill']}")

def display_pose_tracking(pose_data):
    """Display pose tracking visualization"""
    st.subheader("Pose Tracking Visualization")
    
    if not pose_data:
        st.warning("No pose data available for visualization.")
        return
    
    # Create tracking plots
    frame_numbers = list(range(len(pose_data)))
    
    # Key joint angles over time
    if frame_numbers:
        # Extract key angles
        shoulder_angles = []
        elbow_angles = []
        spine_angles = []
        
        for frame_data in pose_data:
            if 'angles' in frame_data:
                angles = frame_data['angles']
                shoulder_angles.append(angles.get('shoulder_angle', 0))
                elbow_angles.append(angles.get('elbow_angle', 0))
                spine_angles.append(angles.get('spine_angle', 0))
            else:
                shoulder_angles.append(0)
                elbow_angles.append(0)
                spine_angles.append(0)
        
        # Plot angles
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=frame_numbers, y=shoulder_angles, name='Shoulder Angle', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=frame_numbers, y=elbow_angles, name='Elbow Angle', line=dict(color='red')))
        fig.add_trace(go.Scatter(x=frame_numbers, y=spine_angles, name='Spine Angle', line=dict(color='green')))
        
        fig.update_layout(
            title="Key Joint Angles Over Time",
            xaxis_title="Frame Number",
            yaxis_title="Angle (degrees)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

def display_comprehensive_report(results):
    """Display comprehensive analysis report"""
    st.subheader("Comprehensive Analysis Report")
    
    # Summary statistics
    st.write("### Analysis Summary")
    
    biomech_results = results['biomech_results']
    feedback = results['feedback']
    
    # Create summary table
    import pandas as pd
    
    phases = ['stance', 'draw', 'anchor', 'release', 'follow_through']
    phase_names = ['Stance & Posture', 'Draw Phase', 'Anchor & Aiming', 'Release', 'Follow-through']
    
    summary_data = []
    for phase, phase_name in zip(phases, phase_names):
        if phase in biomech_results:
            phase_data = biomech_results[phase]
            score = getattr(phase_data, 'score', 0)
            issues_count = len(getattr(phase_data, 'issues', []))
            summary_data.append({
                'Phase': phase_name,
                'Score': f"{score:.1f}/100",
                'Issues': issues_count
            })
    
    if summary_data:
        df = pd.DataFrame(summary_data)
        st.dataframe(df, use_container_width=True)
    
    # Export functionality
    st.write("### Export Results")
    if st.button("📄 Export Analysis Report"):
        export_report(results)

def export_report(results):
    """Export analysis report as text"""
    report = generate_text_report(results)
    
    st.download_button(
        label="Download Report",
        data=report,
        file_name="archery_analysis_report.txt",
        mime="text/plain"
    )

def generate_text_report(results):
    """Generate a text-based analysis report"""
    report = "ARCHERY FORM ANALYSIS REPORT\n"
    report += "=" * 40 + "\n\n"
    
    # Overall score
    overall_score = calculate_overall_score(results['biomech_results'])
    report += f"Overall Form Score: {overall_score:.1f}/100\n\n"
    
    # Phase analysis
    report += "PHASE ANALYSIS:\n"
    report += "-" * 20 + "\n"
    
    phases = ['stance', 'draw', 'anchor', 'release', 'follow_through']
    phase_names = ['Stance & Posture', 'Draw Phase', 'Anchor & Aiming', 'Release', 'Follow-through']
    
    for phase, phase_name in zip(phases, phase_names):
        if phase in results['biomech_results']:
            phase_data = results['biomech_results'][phase]
            report += f"\n{phase_name}:\n"
            report += f"  Score: {getattr(phase_data, 'score', 0):.1f}/100\n"
            
            issues = getattr(phase_data, 'issues', [])
            if issues:
                report += "  Issues:\n"
                for issue in issues:
                    report += f"    - {issue}\n"
            else:
                report += "  No major issues detected\n"
    
    # Feedback
    if 'priority_issues' in results['feedback']:
        report += "\nPRIORITY ISSUES:\n"
        report += "-" * 20 + "\n"
        for issue in results['feedback']['priority_issues']:
            report += f"- {issue['phase']}: {issue['description']}\n"
            if 'recommendation' in issue:
                report += f"  Recommendation: {issue['recommendation']}\n"
    
    return report

def calculate_overall_score(biomech_results):
    """Calculate overall form score"""
    phases = ['stance', 'draw', 'anchor', 'release', 'follow_through']
    total_score = 0
    phase_count = 0
    
    for phase in phases:
        if phase in biomech_results and hasattr(biomech_results[phase], 'score'):
            total_score += biomech_results[phase].score
            phase_count += 1
    
    return total_score / phase_count if phase_count > 0 else 0

def about_page():
    st.header("About AI Archery Form Analysis")
    
    st.markdown("""
    This application uses advanced computer vision and biomechanical analysis to evaluate archery technique
    and provide actionable feedback for improvement.
    
    ### Key Features:
    - **Pose Detection**: Uses MediaPipe for accurate human pose estimation
    - **Phase Analysis**: Analyzes key archery phases including stance, draw, anchor, release, and follow-through
    - **Biomechanical Evaluation**: Identifies form inefficiencies and technique issues
    - **Detailed Feedback**: Provides specific recommendations for improvement
    - **Visual Analysis**: Shows pose tracking and joint angle analysis over time
    
    ### Archery Phases Analyzed:
    
    **1. Stance & Posture**
    - Foot placement and alignment
    - Center of gravity distribution
    - Body orientation to target
    
    **2. Draw Phase**
    - Draw path symmetry
    - Shoulder positioning
    - Elbow tracking
    
    **3. Anchor & Aiming**
    - Anchor point consistency
    - Head stability
    - Bow canting
    
    **4. Release**
    - Release smoothness
    - Follow-through direction
    - Bow hand reaction
    
    **5. Follow-through**
    - Post-release posture
    - Head movement
    - Body stability
    
    ### How to Use:
    1. Upload a video of your archery technique
    2. Click "Analyze Archery Form" to process the video
    3. Review the detailed analysis results
    4. Implement the suggested improvements
    5. Export your analysis report for future reference
    
    ### Tips for Best Results:
    - Record from the side view for optimal pose detection
    - Ensure good lighting and clear visibility
    - Include the full shooting sequence from stance to follow-through
    - Keep the archer in frame throughout the video
    """)

if __name__ == "__main__":
    main()
