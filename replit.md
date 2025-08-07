# Overview

This project is an AI-powered archery form analysis application built with Streamlit that provides detailed biomechanical feedback to help archers improve their technique. The system uses computer vision and pose estimation to analyze archery videos, breaking down the shooting process into distinct phases (stance, draw, anchor, release, follow-through) and providing personalized recommendations for improvement.

The application leverages MediaPipe for pose detection and analysis, combined with custom biomechanical analysis algorithms to evaluate form quality, consistency, and technique across multiple archery phases. Users upload videos of their archery practice, and the system generates comprehensive feedback reports with visualizations and specific drills for improvement.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture
- **Framework**: Streamlit web application providing an intuitive user interface
- **Navigation**: Multi-page application with sidebar navigation for Video Analysis, Results, and About pages
- **File Upload**: Direct video file upload support for multiple formats (MP4, AVI, MOV, MKV, WMV)
- **Visualization**: Plotly integration for interactive charts and graphs displaying analysis results
- **Session State**: Maintains analysis results and application state across page interactions

## Backend Architecture
- **Modular Design**: Component-based architecture with specialized classes for different analysis aspects:
  - `VideoProcessor`: Handles video file processing and frame extraction
  - `PoseAnalyzer`: Performs pose detection and landmark extraction using MediaPipe
  - `BiomechanicalAnalyzer`: Analyzes biomechanical aspects across archery phases
  - `FeedbackGenerator`: Generates personalized recommendations and feedback

## Computer Vision Pipeline
- **Pose Detection**: MediaPipe Pose solution with high-complexity model for accurate landmark detection
- **Frame Processing**: Batch processing of video frames with RGB conversion and resizing capabilities
- **Landmark Analysis**: Focus on key anatomical landmarks relevant to archery form (shoulders, elbows, wrists, hips, knees, ankles)

## Analysis Engine
- **Phase Segmentation**: Automatic detection and analysis of five archery phases
- **Scoring System**: Weighted scoring algorithms based on biomechanical principles
- **Consistency Tracking**: Frame-by-frame analysis to identify inconsistencies in form
- **Feedback Generation**: Template-based feedback system with specific recommendations and practice drills

## Data Processing
- **Temporal Analysis**: Time-series analysis of pose data to track movement patterns
- **Statistical Evaluation**: Consistency metrics, variance analysis, and smoothness calculations
- **Geometric Calculations**: Angle measurements, distance calculations, and alignment assessments

# External Dependencies

## Core Libraries
- **Streamlit**: Web application framework for user interface
- **OpenCV (cv2)**: Video processing and computer vision operations
- **MediaPipe**: Google's pose estimation and landmark detection library
- **NumPy**: Numerical computing and array operations
- **Plotly**: Interactive data visualization and charting

## Additional Dependencies
- **PIL (Pillow)**: Image processing and manipulation
- **tempfile/os**: File system operations and temporary file management
- **math**: Mathematical calculations and geometric operations
- **typing**: Type hints and annotations for better code documentation

## MediaPipe Configuration
- **Model Complexity**: Level 2 for high-accuracy pose detection
- **Detection Confidence**: 50% minimum threshold
- **Tracking Confidence**: 50% minimum threshold for consistent landmark tracking
- **Segmentation**: Disabled to focus on pose landmarks only

## File System Integration
- **Temporary File Management**: Automatic cleanup of uploaded video files
- **Multi-format Support**: Comprehensive video format compatibility
- **Frame Extraction**: Efficient video-to-frame conversion with metadata preservation