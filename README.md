# Archery Form Analysis — Video Demo

A Streamlit demonstration of an archery-video analysis workflow: upload footage, inspect estimated poses, and explore phase scores and rule-based feedback.

## What it implements

- Process uploaded video into frames.
- Calculate joint angles and position metrics from generated landmarks.
- Group analysis into stance, draw, anchor, release and follow-through.
- Present scores, visualizations and feedback templates.

## How it works

`video_processor.py` extracts frames; `pose_analyzer.py` estimates a body region from contours and generates proportional landmarks with random variations. `biomechanical_analyzer.py` calculates phase metrics, and `feedback_generator.py` maps issues to written suggestions. No trained pose network is loaded: `pose_net` remains `None`.

## Current limitations

The landmarks are synthetic approximations, and pose detection is assumed successful. Scores are demonstrations, not measured biomechanics or coaching validation. Results can vary between runs because of randomness. The manifest includes both OpenCV GUI and headless packages; resolve that overlap in a separate dependency change if installation fails.

## Run locally

The commands below were checked against source paths and manifests; application startup was not tested unless explicitly stated.
Python 3.11 or newer is declared in `pyproject.toml`. Use uv for the checked-in project/lockfile. Set environment variables in the shell or process manager; the Python entry points do not explicitly load `.env`.

```bash
git clone --branch main https://github.com/AishwaryaChandel27/Archery-Precision-Posture-Evaluation.git
cd Archery-Precision-Posture-Evaluation
uv sync
uv run streamlit run app.py --server.port 5000
```
Open `http://localhost:5000`. 

## Usage

Upload a short archery clip, run analysis, and inspect phase-level results. Repository sample clips are in `video/`. Use the output to review the software pipeline, not to judge an archer’s technique.

## Suggested next work

Replace synthetic landmarks with a real pose estimator, validate against annotated videos, and measure phase segmentation and landmark errors before making accuracy claims.

## Source map

| Path | Responsibility |
| --- | --- |
| [app.py](app.py) | Streamlit interface |
| [video_processor.py](video_processor.py) | Frame extraction |
| [pose_analyzer.py](pose_analyzer.py) | Contour-based synthetic landmarks |
| [biomechanical_analyzer.py](biomechanical_analyzer.py) | Phase scoring |
| [feedback_generator.py](feedback_generator.py) | Rule-based feedback |

## Validation

Documentation was checked against the repository snapshot on 28 October 2026. No application test suite was run for this repository. Provider availability, browser behavior, builds and deployment are unverified.

## Demo capture

Suggested addition: a short recording of the usage workflow above, with synthetic inputs and visible limitations. No new screenshot or recording was produced in this audit.

## Contributing

Open an issue with reproduction steps or submit a focused pull request. Keep prototype behavior and future plans clearly distinguished, and include verification steps for changes.

## License

No license file was found. A manifest or older README license label does not replace an explicit license grant.

Maintained in [AishwaryaChandel27’s repositories](https://github.com/AishwaryaChandel27).
