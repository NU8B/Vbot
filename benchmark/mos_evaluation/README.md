# TTS Model MOS Evaluation System

This is a web-based Mean Opinion Score (MOS) evaluation system for TTS models. It allows users to:
- Listen to original and generated audio samples
- Rate voice similarity
- Rate emotion expression quality
- Provide additional comments
- View aggregated results

## Deployment Instructions

### 1. Host Audio Files on GitHub

1. Create a new GitHub repository or use an existing one
2. Push your audio files to the repository with this structure:
   ```
   your-repo/
   ├── Data_prep/
   │   └── raw_data/
   │       └── raw_audio2/
   │           └── *.wav
   └── asset/
       └── outputs/
           ├── new_ft_StyleTTS2/
           │   └── *.wav
           └── Amelia10_ft_StyleTTS2/
               └── *.wav
   ```
3. Enable GitHub Pages:
   - Go to repository Settings
   - Scroll to "GitHub Pages" section
   - Select main branch as source
   - Save changes

4. Update the `AUDIO_BASE_URL` in `app.py` with your repository URL:
   ```python
   AUDIO_BASE_URL = "https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO/main"
   ```

### 2. Deploy Flask App to PythonAnywhere

1. Sign up for a free account at [PythonAnywhere.com](https://www.pythonanywhere.com)

2. Create a new web app:
   - Go to the "Web" tab
   - Click "Add a new web app"
   - Choose "Flask" as your framework
   - Select Python 3.8 or later

3. Upload app files (excluding audio files):
   ```
   mysite/
   ├── app.py
   ├── requirements.txt
   └── templates/
       ├── index.html
       └── results.html
   ```

4. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

5. Configure the web app:
   - In "Web" tab, set working directory to `/home/yourusername/mysite`
   - Set WSGI configuration file to use `app.py`
   - Click "Reload" to apply changes

Your evaluation form will be available at:
```
http://yourusername.pythonanywhere.com
```

## Features

- Easy-to-use web interface
- Slider-based rating system (1-10)
- Audio playback for all samples
- Results visualization with charts
- Comments section for additional feedback
- Automatic data storage in JSON format

## Directory Structure

```
mos_evaluation/
├── app.py              # Main Flask application
├── requirements.txt    # Python dependencies
├── static/
│   └── audio/         # Audio files for evaluation
└── templates/
    ├── index.html     # Evaluation form
    └── results.html   # Results visualization
```

## Data Storage

All evaluation results are stored in `results/mos_evaluations/mos_results.json`

## Evaluation Process

1. Users listen to original and generated audio samples
2. Rate voice similarity on a scale of 1-10
3. Rate emotion expression quality for each emotion (joy, sadness, anger, surprise, neutral)
4. Optionally provide additional comments
5. Submit evaluation
6. View results page with aggregated statistics

## Results Analysis

The results page shows:
- Average ratings for each model and emotion
- Bar charts for visual comparison
- Recent user comments
- Overall statistics

## Local Development

1. Create a Python virtual environment:
```bash
python -m venv venv
.\venv\Scripts\activate  # On Windows
source venv/bin/activate  # On Linux/Mac
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

4. Open your browser:
```
http://localhost:5000
``` 