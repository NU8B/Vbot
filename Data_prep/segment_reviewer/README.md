# 🎵 Audio Segment Reviewer

A web-based interface for manually reviewing and approving audio segments for training data. This tool helps you efficiently review the segments that passed automated quality checks to ensure they're truly suitable for your training dataset.

## 🚀 Quick Start

### Simple Setup
```bash
# Navigate to this folder
cd Data_prep/segment_reviewer

# Install dependencies (one time only)
pip install -r requirements.txt

# Run the reviewer
python segment_reviewer.py
```

Then open your browser and go to: **http://localhost:5000**

## 📋 Features

### 🎧 Audio Playback
- Built-in audio player with controls
- Auto-plays each segment when loaded
- High-quality WAV playback

### 📊 Quality Metrics Display
- **STOI** (Speech Intelligibility) - How clear the speech is
- **PESQ** (Perceptual Speech Quality) - Overall audio quality
- **Duration** - Segment length
- **RMS Energy** - Audio volume levels
- **Spectral Centroid** - Frequency characteristics
- **Peak Ratio** - Detects pops/clicks

### 📝 Transcription Review & Editing
- View the AI-generated transcription
- **Edit transcriptions directly** in the interface
- Automatic saving of transcription changes
- Track original vs edited text
- Verify accuracy against what you hear
- Easily spot and fix transcription errors

### ⚡ Fast Review Controls
- **Approve** - Good for training ✅
- **Reject** - Not suitable ❌
- **Skip** - Review later ⏭️

### ⌨️ Keyboard Shortcuts
- **Space** - Play/Pause audio
- **A** - Approve segment
- **R** - Reject segment
- **S** - Skip segment
- **← →** - Navigate between segments

*Note: Keyboard shortcuts are disabled when editing transcription text*

### 📈 Progress Tracking
- Real-time progress bar
- Statistics dashboard
- Session persistence (resume where you left off)
- Built-in review analysis

### 📝 Review Notes
- Add optional notes for each decision
- Helpful for tracking specific issues
- Saved with the segment for future reference

## 📁 File Organization

The reviewer automatically organizes your files:

```
Data_prep/raw_data/segments/
├── passed_segments/           # Original passed segments (gets emptied as you review)
├── reviewed_approved/         # ✅ Segments you approved
├── reviewed_rejected/         # ❌ Segments you rejected
├── reviewed_skipped/          # ⏭️ Segments you skipped
├── metadata.json             # Original metadata
├── summary.json              # Processing summary
├── review_state.json         # Your review progress (auto-saved)
└── approved_segments_metadata.json  # Final approved metadata (downloadable)
```

## 🎯 Review Guidelines

### ✅ **Approve** if:
- Clear, intelligible speech
- Accurate transcription
- Good audio quality (no distortion, noise)
- Natural speaking pace
- Single speaker
- No background noise or interference

### ❌ **Reject** if:
- Poor audio quality (distorted, noisy, muffled)
- Incorrect or incomplete transcription
- Multiple speakers or overlapping speech
- Background music, noise, or interference
- Unnatural speech (robotic, synthetic)
- Inappropriate content

### ⏭️ **Skip** if:
- Unsure about quality
- Need to come back later
- Want a second opinion
- Borderline case requiring more thought

## 📊 Quality Metrics Explained

| Metric | Good Range | What It Means |
|--------|------------|---------------|
| **STOI** | > 0.95 | Speech intelligibility (higher = clearer) |
| **PESQ** | > 4.0 | Perceptual quality (1-4.5 scale) |
| **RMS Energy** | 0.05-0.95 | Audio volume (not too quiet/loud) |
| **Peak Ratio** | < 8 | Detects pops/clicks (lower = better) |

Green indicators = Good quality
Yellow indicators = Acceptable quality  
Red indicators = Poor quality

## 🔄 Workflow Tips

1. **Start Fresh**: Begin with segments that clearly pass or fail to build momentum
2. **Use Keyboard Shortcuts**: Much faster than clicking buttons
3. **Add Notes**: For borderline cases, note why you made the decision
4. **Take Breaks**: Audio review can be tiring - take regular breaks
5. **Review Skipped**: Come back to skipped segments when you're fresh

## 📥 Exporting Results

The reviewer automatically maintains compatibility with the StyleTTS2 preparation pipeline:

### Auto-Export (Recommended)
- Approved metadata is automatically saved as `approved_segments_metadata.json`
- This file is automatically used by `data_StyleTTS2.py` 
- No manual export needed - just continue to the next step!

### Manual Export
Click **"📥 Download Approved Metadata"** to manually download:
- JSON file with all approved segments
- Ready for StyleTTS2 training pipeline  
- Includes all quality metrics and review notes

### Next Step: StyleTTS2 Preparation
After reviewing, run the StyleTTS2 data preparation:
```bash
python Data_prep/data_StyleTTS2.py
```
It will automatically use your approved segments!

## 📈 Built-in Analysis

The reviewer includes built-in analysis features:
- Review pattern analysis
- Quality metrics comparison between approved/rejected segments
- Approval rate insights
- Recommendations for improving review process

Access analysis via: `http://localhost:5000/analysis` (when server is running)

## 🐛 Troubleshooting

### Audio Won't Play
- Check browser audio permissions
- Try refreshing the page
- Ensure WAV files are not corrupted

### "Segments directory not found"
- Make sure you've run the audio segmenter first
- Check that `Data_prep/raw_data/segments/` exists
- The script will auto-detect the correct path

### Missing Dependencies
```bash
pip install Flask numpy
```

### Progress Lost
- Review state is auto-saved to `review_state.json`
- If corrupted, delete the file to start fresh
- Your organized files remain intact

### Performance Issues
- Close other browser tabs
- Use Chrome/Firefox for best performance
- Consider reviewing in smaller batches

## 🔧 Customization

You can modify the reviewer by editing `segment_reviewer.py`:

- Change port: `app.run(port=8080)`
- Modify quality thresholds in the HTML templates
- Add custom review categories
- Adjust file paths if needed

## 🎯 Best Practices

1. **Consistency**: Establish clear criteria and stick to them
2. **Speed**: Don't over-analyze - trust your first impression
3. **Quality**: When in doubt, reject rather than approve
4. **Documentation**: Use notes for edge cases and patterns
5. **Validation**: Spot-check your decisions periodically



---

**Happy reviewing! 🎵 Your careful review will ensure high-quality training data.** 