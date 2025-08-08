# Application Dependencies

## Core Dependencies
- **streamlit>=1.28.0** - Web application framework
- **opencv-python>=4.8.0** - Computer vision and video processing
- **numpy>=1.21.0** - Numerical computing
- **plotly>=5.0.0** - Interactive data visualization
- **pandas>=1.3.0** - Data manipulation and analysis
- **pillow>=8.0.0** - Image processing

## Optional Dependencies
- **mediapipe>=0.10.0** - Pose estimation (falls back to demo mode if unavailable)
- **matplotlib>=3.5.0** - Additional plotting capabilities
- **seaborn>=0.11.0** - Statistical data visualization

## Installation Notes
If MediaPipe fails to install due to platform compatibility issues, the application will automatically fall back to demonstration mode with simulated pose data. This allows you to test the full application functionality without MediaPipe.

To install dependencies manually:
```bash
pip install streamlit opencv-python numpy plotly pandas pillow
```

For full functionality (if platform supports it):
```bash
pip install mediapipe matplotlib seaborn
```