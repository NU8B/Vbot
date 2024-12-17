import os
import sys
import subprocess
import shutil
from pathlib import Path
import yt_dlp
import time

START_TIME = time.time()
# Get the root directory
ROOT_DIR = Path(__file__).parent.parent
DOWNLOADS_DIR = ROOT_DIR / "cache" / "downloads"
RAW_AUDIO_DIR = ROOT_DIR / "Data_prep" / "raw_data" / "full_audio"
PROCESSED_DIR = ROOT_DIR / "Data_prep" / "raw_data" / "processed_audio"
SEGMENTS_DIR = ROOT_DIR / "Data_prep" / "raw_data" / "2hour_amelia"
DATA_DIR = ROOT_DIR / "Data_prep" / "Data"

# FFmpeg path configuration
FFMPEG_PATHS = [
    r"D:\ffmpeg-2024-12-11-git-a518b5540d-full_build\bin",  # Your FFmpeg path
    os.environ.get("FFMPEG_PATH", ""),  # Environment variable if set
    "",  # System PATH
]


def run_pipeline(youtube_url):
    """Run the complete pipeline from YouTube to dataset"""
    try:
        # Setup
        print("Setting up environment...")
        setup_directories()

        # Clean up previous run files first
        cleanup_paths = [
            ROOT_DIR / "Data_prep" / "raw_data" / "full_audio" / "vocals.wav",
            ROOT_DIR / "Data_prep" / "raw_data" / "full_audio" / "input.mp3",
            ROOT_DIR / "Data_prep" / "raw_data" / "2hour_amelia",
            ROOT_DIR / "Data_prep" / "Data" / "wavs",
            ROOT_DIR / "Data_prep" / "Data" / "train_list.txt",
            ROOT_DIR / "Data_prep" / "Data" / "val_list.txt",
            DOWNLOADS_DIR,
        ]

        print("Cleaning up previous run files...")
        for path in cleanup_paths:
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
                path.mkdir(parents=True, exist_ok=True)

        # Step 1: Download from YouTube
        print("\nStep 1: Downloading from YouTube...")
        input_file = download_from_youtube(youtube_url)

        # Step 2: Isolate vocals
        print("\nStep 2: Isolating vocals...")
        vocals_file = RAW_AUDIO_DIR / "vocals.wav"
        subprocess.run(
            [
                sys.executable,
                "Data_prep/audio_preprocessor/vocal_isolator.py",
                "--input",
                str(input_file),
                "--output",
                str(vocals_file),
            ],
            check=True,
        )

        # Step 3: Segment audio
        print("\nStep 3: Segmenting audio...")
        subprocess.run(
            [
                sys.executable,
                "Data_prep/audio_preprocessor/audio_segmenter.py",
                "--input",
                str(vocals_file),
                "--output",
                str(SEGMENTS_DIR),
            ],
            check=True,
        )

        # Step 4: Prepare dataset
        print("\nStep 4: Preparing dataset...")
        subprocess.run(
            [
                sys.executable,
                "Data_prep/data_StyleTTS2.py",
                "--input",
                str(SEGMENTS_DIR),
                "--output",
                str(DATA_DIR),
                "--sr",
                "24000",
                "--max-tokens",
                "377",
            ],
            check=True,
        )

        print("\nPipeline completed successfully!")

    except Exception as e:
        print(f"\nError in pipeline: {e}")
        raise


def find_ffmpeg():
    """Find FFmpeg executable in the system"""
    for base_path in FFMPEG_PATHS:
        if not base_path:
            continue

        ffmpeg_path = Path(base_path) / "ffmpeg.exe"
        ffprobe_path = Path(base_path) / "ffprobe.exe"

        if ffmpeg_path.exists() and ffprobe_path.exists():
            return str(ffmpeg_path.parent)

    return None


def setup_directories():
    """Create necessary directories"""
    directories = [DOWNLOADS_DIR, RAW_AUDIO_DIR, PROCESSED_DIR, SEGMENTS_DIR, DATA_DIR]
    for dir_path in directories:
        dir_path.mkdir(parents=True, exist_ok=True)


def install_dependencies():
    """Install required dependencies"""
    try:
        print("Installing/Updating dependencies...")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "yt-dlp",  # For YouTube download
                "--upgrade",  # Ensure we have the latest version
            ],
            check=True,
        )

        # Check FFmpeg installation
        ffmpeg_path = find_ffmpeg()
        if not ffmpeg_path:
            print("\nWARNING: FFmpeg not found in standard locations.")
            print("Please ensure FFmpeg is installed and in your system PATH")
            print("You can download FFmpeg from: https://ffmpeg.org/download.html")
            print("Expected paths:")
            for path in FFMPEG_PATHS:
                if path:
                    print(f"- {path}")
            raise RuntimeError("FFmpeg not found")
        else:
            print(f"\nFound FFmpeg at: {ffmpeg_path}")
            # Add FFmpeg to system PATH for this session
            os.environ["PATH"] = ffmpeg_path + os.pathsep + os.environ["PATH"]

    except Exception as e:
        print(f"Error installing dependencies: {e}")
        raise


def download_from_youtube(url):
    """Download audio from YouTube with best quality"""
    ffmpeg_path = find_ffmpeg()
    if not ffmpeg_path:
        raise RuntimeError("FFmpeg not found. Please install FFmpeg and add it to PATH")

    output_template = str(DOWNLOADS_DIR / "%(title)s.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "320",
            }
        ],
        "outtmpl": output_template,
        "quiet": False,
        "no_warnings": True,
        "extract_audio": True,
        "audio_format": "mp3",
        "audio_quality": 0,  # Best quality
        "prefer_ffmpeg": True,
        "ffmpeg_location": ffmpeg_path,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Get video info first
            info = ydl.extract_info(url, download=False)
            video_title = info.get("title", "video")
            duration = info.get("duration", 0)

            # Check duration (limit to 3 hours to prevent massive downloads)
            if duration > 10800:  # 3 hours in seconds
                raise ValueError(
                    f"Video too long: {duration/3600:.1f} hours. Maximum allowed is 3 hours."
                )

            print(f"\nDownloading: {video_title}")
            print(f"Duration: {duration/60:.1f} minutes")

            # Download the video
            ydl.download([url])

            # Find the downloaded file
            downloaded_files = list(DOWNLOADS_DIR.glob("*.mp3"))
            if not downloaded_files:
                raise FileNotFoundError("Download completed but no MP3 file found")

            # Move the file to raw_audio directory
            downloaded_file = downloaded_files[0]
            target_file = RAW_AUDIO_DIR / "input.mp3"
            shutil.move(str(downloaded_file), str(target_file))

            return target_file

    except Exception as e:
        print(f"Error downloading from YouTube: {e}")
        raise


if __name__ == "__main__":
    # Example YouTube URL
    youtube_url = (
        "https://www.youtube.com/watch?v=F-9KWQByeU0"  # Replace with actual URL
    )

    try:
        run_pipeline(youtube_url)
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        raise
    print(f"Total time taken: {time.time() - START_TIME:.2f} seconds")
    print(f"Total time taken: {(time.time() - START_TIME) / 60:.2f} minutes")
