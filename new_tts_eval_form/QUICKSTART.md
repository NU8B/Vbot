# Quick Start Guide

## Prerequisites

1. You've already run `utils/TEST_multiple_model_inference.py` to generate the 100 audio files
2. The audio files are in `asset/outputs/` directory for each model

## Setup Steps (Windows)

### Step 1: Copy Audio Files

From the **Vbot project root** directory, run:

```bash
python new_tts_eval_form/setup_audio_files.py
```

This will copy all 100 audio files to the correct location.

### Step 2: Install Dependencies

```bash
cd new_tts_eval_form
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### Step 3: Run the Application

```bash
python app.py
```

### Step 4: Open in Browser

Navigate to: `http://localhost:5000`

## Usage

1. **Evaluation Form** (`/`)
   - Listen to 100 audio samples
   - Select the emotion for each sample
   - Submit when all are completed

2. **Results Page** (`/results`)
   - View overall comparison between old and new models
   - See accuracy breakdown by emotion
   - Analyze individual model performance

## Expected File Structure After Setup

```
new_tts_eval_form/
├── app.py
├── requirements.txt
├── README.md
├── QUICKSTART.md
├── setup_audio_files.py
├── templates/
│   ├── index.html
│   └── results.html
├── static/
│   └── audio/
│       ├── Amelia/          (25 .wav files: 20 specific + 5 generic)
│       ├── Eveland/         (25 .wav files: 20 specific + 5 generic)
│       ├── Gura/            (25 .wav files: 20 specific + 5 generic)
│       └── Amelia_new/      (25 .wav files: 20 specific + 5 generic)
└── results/
    └── evaluation_results.json  (created after first submission)
```

## Troubleshooting

### No audio files found
- Make sure you've run `utils/TEST_multiple_model_inference.py` first
- Check that files exist in `asset/outputs/[Model]/specific_*.wav`
- Run `setup_audio_files.py` again

### Port already in use
- Change the port in `app.py`: `app.run(debug=True, host="0.0.0.0", port=5001)`

### Audio not playing
- Make sure your browser supports WAV files
- Check browser console for errors
- Verify audio files are valid WAV format

## Model Comparison

- **Old Models**: Amelia, Eveland (baseline)
- **New Models**: Gura, Amelia_new (improved models)

The results page will show which model group performs better at emotion recognition!

## Tips

- The evaluation takes about 20-30 minutes
- Use headphones for better audio quality
- You can replay each audio sample as many times as needed
- Progress is saved in the browser (but not submitted until you click Submit)

