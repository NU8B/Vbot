# TTS Emotion Recognition Evaluation Form

A simplified web-based evaluation system for comparing emotion recognition accuracy between old and new TTS models.

## Overview

This evaluation form allows users to:
- Listen to 100 audio samples from 4 different TTS models
- Identify the emotion expressed in each sample
- Compare accuracy between old models (Amelia, Eveland) and new models (Gura, Amelia_new)

## Features

- **Single-page evaluation**: All 100 audio samples on one scrollable page
- **Progress tracking**: Visual progress bar showing completion status
- **Simplified workflow**: Just audio + emotion selection (no confidence ratings)
- **Comprehensive results**: Compare old vs new models with detailed statistics
- **Beautiful visualizations**: Charts showing overall and per-emotion accuracy

## Setup Instructions

### 1. Prepare Audio Files

Create the following directory structure in the `static` folder:

```
new_tts_eval_form/
├── static/
│   └── audio/
│       ├── Amelia/
│       │   ├── specific_joy_1.wav
│       │   ├── specific_joy_2.wav
│       │   ├── specific_joy_3.wav
│       │   ├── specific_joy_4.wav
│       │   ├── generic_joy.wav
│       │   ├── specific_sadness_1.wav
│       │   ... (25 files total: 4 specific + 1 generic per emotion × 5 emotions)
│       ├── Eveland/
│       │   ... (25 files total)
│       ├── Gura/
│       │   ... (25 files total)
│       └── Amelia_new/
│           ... (25 files total)
```

**Total: 100 audio files** (4 models × 25 files each)
- Each model has 20 specific files (4 prompts × 5 emotions)
- Each model has 5 generic files (1 generic × 5 emotions)

You can copy the audio files generated from `utils/TEST_multiple_model_inference.py`:
```bash
# From the Vbot project root
# Use the automated setup script (recommended)
python new_tts_eval_form/setup_audio_files.py

# Or manually copy files:
cp asset/outputs/Amelia/*.wav new_tts_eval_form/static/audio/Amelia/
cp asset/outputs/Eveland/*.wav new_tts_eval_form/static/audio/Eveland/
cp asset/outputs/Gura/*.wav new_tts_eval_form/static/audio/Gura/
cp asset/outputs/Amelia_new/*.wav new_tts_eval_form/static/audio/Amelia_new/
```

### 2. Install Dependencies

Create a virtual environment and install requirements:

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt

# Linux/Mac
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Run the Application

```bash
python app.py
```

The application will be available at: `http://localhost:5000`

## Usage

### For Evaluators

1. Open the evaluation form at `http://localhost:5000`
2. Listen to each audio sample
3. Select the emotion you think is being expressed
4. Progress is tracked automatically
5. Click "Submit Evaluation" when all 100 samples are completed
6. View the results page with comparison statistics

### View Results

Navigate to `http://localhost:5000/results` to see:
- Overall accuracy comparison (Old vs New models)
- Accuracy breakdown by emotion type
- Individual model performance
- Detailed statistics with charts

## Data Storage

All evaluation results are stored in:
```
results/evaluation_results.json
```

Each submission includes:
- Timestamp
- User IP address
- All 100 emotion selections
- Model and emotion metadata

## Model Groups

- **Old Models**: Amelia, Eveland
- **New Models**: Gura, Amelia_new

## Emotions

The system evaluates recognition of 5 emotions:
1. Joy
2. Sadness
3. Anger
4. Surprise
5. Neutral

## Deployment

### Deploy to PythonAnywhere

1. Create a free account at [PythonAnywhere.com](https://www.pythonanywhere.com)
2. Upload all files (excluding audio files)
3. Create a web app with Flask framework
4. Upload audio files to the static/audio directory
5. Install requirements: `pip install -r requirements.txt`
6. Set WSGI configuration to use `app.py`
7. Reload the web app

### Deploy to Heroku

1. Add a `Procfile`:
   ```
   web: gunicorn app:app
   ```
2. Create a Heroku app
3. Deploy via Git or GitHub integration
4. Note: Free tier may have storage limitations for audio files

## Results Analysis

The results page provides:

### Overall Comparison
- Side-by-side accuracy comparison
- Winner indication with percentage difference

### Emotion-wise Analysis
- Bar chart comparing accuracy for each emotion
- Identifies which emotions are easier/harder to recognize

### Individual Model Performance
- Radar chart for each model showing per-emotion accuracy
- Overall accuracy percentage for each model

## File Structure

```
new_tts_eval_form/
├── app.py                  # Main Flask application
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── templates/
│   ├── index.html         # Evaluation form
│   └── results.html       # Results visualization
├── static/
│   └── audio/            # Audio files (not included in repo)
└── results/
    └── evaluation_results.json  # Stored evaluations
```

## Technical Details

- **Backend**: Flask (Python)
- **Frontend**: Bootstrap 5, vanilla JavaScript
- **Charts**: Chart.js
- **Data Format**: JSON
- **Audio Format**: WAV files

## Differences from Old MOS Evaluation

1. **Single page**: All samples on one page vs. multiple sections
2. **No confidence ratings**: Only emotion selection required
3. **No naturalness evaluation**: Focus only on emotion recognition
4. **Model comparison**: Built-in old vs. new model comparison
5. **Simplified workflow**: Faster evaluation process

## Tips for Evaluators

- Use headphones for better audio quality
- Take breaks if needed - your progress is saved in the browser
- You can replay audio samples as many times as needed
- The evaluation typically takes 20-30 minutes to complete

## Support

For issues or questions, please check:
- Audio files are in the correct directory structure
- All dependencies are installed
- Flask server is running on the correct port

