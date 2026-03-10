# Outstanding

A real-time NBA pose twin app. Strike a pose and it matches you to the NBA player whose body position most closely resembles yours.

## How it works

- Uses **MediaPipe Pose** to extract upper body landmarks (nose, shoulders, elbows, wrists, hands) from both your webcam feed and each NBA player photo
- Normalizes skeletons relative to torso size so body proportions don't skew the match
- Finds the closest match using **cosine similarity** on the pose embeddings

## Requirements

- Python 3.9–3.12 (3.11 recommended)
- A webcam
- macOS (tested on Apple Silicon)

## Setup

```bash
git clone https://github.com/Simonmann17/Outstanding.git
cd Outstanding

python3.11 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

On first run the app will automatically download the MediaPipe pose model (~6MB).

## Run

```bash
python3 main.py
```

- A camera feed window opens alongside the matched player photo
- Press **ESC** to quit

## Adding players

Drop any `.jpeg` or `.jpg` image into the `hangitup/` folder. Action shots with a visible torso work best — tight headshots won't have a detectable pose and will be skipped at startup.

## Project structure

```
hangitup/        # NBA player photos used for matching
detector/
  camera.py      # Webcam capture
  pose_matcher.py # Pose extraction, normalization, and cosine similarity matching
display/
  image_display.py # Renders camera feed and matched player side by side
audio/
  music.mp3      # Background music
main.py          # Entry point
```
