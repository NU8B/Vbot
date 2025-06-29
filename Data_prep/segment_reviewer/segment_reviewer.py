import os
import json
import shutil
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
import numpy as np
from collections import defaultdict, Counter

app = Flask(__name__)


class SegmentReviewer:
    def __init__(self, segments_dir=None):
        # Auto-detect segments directory
        if segments_dir is None:
            # Try different possible paths
            current_dir = Path(__file__).parent
            possible_paths = [
                current_dir / "../raw_data/segments",  # From segment_reviewer folder
                current_dir / "../../raw_data/segments",  # If deeper nested
                Path("Data_prep/raw_data/segments"),  # From project root
                Path("raw_data/segments"),  # Direct path
            ]

            for path in possible_paths:
                if path.exists():
                    segments_dir = path
                    break

            if segments_dir is None:
                # Create the expected directory structure
                segments_dir = current_dir / "../raw_data/segments"
                segments_dir.mkdir(parents=True, exist_ok=True)

        self.segments_dir = Path(segments_dir).resolve()
        self.passed_dir = self.segments_dir / "passed_segments"
        self.failed_dir = self.segments_dir / "failed_segments"
        self.srts_dir = self.segments_dir / "srts"

        # Create review directories
        self.approved_dir = self.segments_dir / "reviewed_approved"
        self.rejected_dir = self.segments_dir / "reviewed_rejected"
        self.skipped_dir = self.segments_dir / "reviewed_skipped"

        # Ensure all directories exist
        for dir_path in [
            self.passed_dir,
            self.failed_dir,
            self.srts_dir,
            self.approved_dir,
            self.rejected_dir,
            self.skipped_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Load metadata
        self.metadata = self.load_metadata()
        self.review_state = self.load_review_state()

    def load_metadata(self):
        """Load segment metadata"""
        metadata_file = self.segments_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, "r", encoding="utf-8") as f:
                return {item["filename"]: item for item in json.load(f)}
        return {}

    def load_review_state(self):
        """Load review state from file"""
        state_file = self.segments_dir / "review_state.json"
        default_state = {
            "current_index": 0,
            "approved": [],
            "rejected": [],
            "skipped": [],
            "last_reviewed": None,
        }

        if state_file.exists():
            try:
                with open(state_file, "r", encoding="utf-8") as f:
                    return {**default_state, **json.load(f)}
            except:
                return default_state
        return default_state

    def save_review_state(self):
        """Save current review state"""
        state_file = self.segments_dir / "review_state.json"
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(self.review_state, f, indent=2)

    def get_segment_list(self):
        """Get list of all segments to review"""
        segments = []

        # Get all wav files from passed_segments
        for wav_file in self.passed_dir.glob("*.wav"):
            filename = wav_file.name

            # Skip if already reviewed
            if (
                filename in self.review_state["approved"]
                or filename in self.review_state["rejected"]
                or filename in self.review_state["skipped"]
            ):
                continue

            segments.append(
                {
                    "filename": filename,
                    "path": str(wav_file),
                    "metadata": self.metadata.get(filename, {}),
                }
            )

        return segments

    def get_current_segment(self):
        """Get current segment for review"""
        segments = self.get_segment_list()
        if not segments:
            return None

        current_index = min(self.review_state["current_index"], len(segments) - 1)
        if current_index < 0:
            return None

        return segments[current_index]

    def review_segment(self, filename, action, notes="", edited_text=None):
        """Record review decision for a segment"""
        segment_path = self.passed_dir / filename

        if not segment_path.exists():
            return False

        # Update transcription if edited
        if edited_text is not None and edited_text.strip():
            # Clean and validate the edited text
            cleaned_text = edited_text.strip()
            original_text = self.metadata[filename].get("text", "")

            if cleaned_text != original_text:
                # Store original text if not already stored
                if not self.metadata[filename].get("transcription_edited", False):
                    self.metadata[filename]["original_text"] = original_text

                # Update the metadata with the edited transcription
                self.metadata[filename]["text"] = cleaned_text
                self.metadata[filename]["transcription_edited"] = True

                # Save updated metadata
                self.save_metadata()

        # Record the decision
        if action == "approve":
            self.review_state["approved"].append(filename)
            dest_dir = self.approved_dir
        elif action == "reject":
            self.review_state["rejected"].append(filename)
            dest_dir = self.rejected_dir
        elif action == "skip":
            self.review_state["skipped"].append(filename)
            dest_dir = self.skipped_dir
        else:
            return False

        # Move the file
        try:
            dest_path = dest_dir / filename
            shutil.move(str(segment_path), str(dest_path))

            # Save review notes if provided
            if notes:
                notes_file = dest_path.with_suffix(".review_notes.txt")
                with open(notes_file, "w", encoding="utf-8") as f:
                    f.write(notes)

            # Move to next segment
            self.review_state["current_index"] += 1
            self.review_state["last_reviewed"] = filename
            self.save_review_state()

            # Auto-save approved metadata for StyleTTS2 compatibility
            self.save_approved_metadata()

            return True
        except Exception as e:
            print(f"Error moving file: {e}")
            return False

    def get_review_stats(self):
        """Get review statistics"""
        total_segments = len(list(self.passed_dir.glob("*.wav")))
        reviewed = (
            len(self.review_state["approved"])
            + len(self.review_state["rejected"])
            + len(self.review_state["skipped"])
        )
        remaining = total_segments

        return {
            "total_passed": total_segments + reviewed,
            "remaining": remaining,
            "approved": len(self.review_state["approved"]),
            "rejected": len(self.review_state["rejected"]),
            "skipped": len(self.review_state["skipped"]),
            "progress_percent": (
                (reviewed / (total_segments + reviewed)) * 100
                if (total_segments + reviewed) > 0
                else 0
            ),
        }

    def analyze_review_patterns(self):
        """Analyze review patterns and quality metrics"""
        approved = self.review_state.get("approved", [])
        rejected = self.review_state.get("rejected", [])
        skipped = self.review_state.get("skipped", [])

        total_reviewed = len(approved) + len(rejected) + len(skipped)

        if total_reviewed == 0:
            return {"error": "No reviews completed yet"}

        # Quality metrics analysis
        approved_metrics = []
        rejected_metrics = []

        for filename in approved:
            if (
                filename in self.metadata
                and "quality_metrics" in self.metadata[filename]
            ):
                approved_metrics.append(self.metadata[filename]["quality_metrics"])

        for filename in rejected:
            if (
                filename in self.metadata
                and "quality_metrics" in self.metadata[filename]
            ):
                rejected_metrics.append(self.metadata[filename]["quality_metrics"])

        analysis = {
            "total_reviewed": total_reviewed,
            "approved_count": len(approved),
            "rejected_count": len(rejected),
            "skipped_count": len(skipped),
            "approval_rate": len(approved) / total_reviewed * 100,
            "metrics_comparison": {},
        }

        if approved_metrics and rejected_metrics:
            metrics_to_compare = [
                "stoi",
                "pesq",
                "rms_energy",
                "peak_ratio",
                "duration",
            ]

            for metric in metrics_to_compare:
                approved_values = [m[metric] for m in approved_metrics if metric in m]
                rejected_values = [m[metric] for m in rejected_metrics if metric in m]

                if approved_values and rejected_values:
                    analysis["metrics_comparison"][metric] = {
                        "approved_avg": np.mean(approved_values),
                        "rejected_avg": np.mean(rejected_values),
                        "difference": np.mean(approved_values)
                        - np.mean(rejected_values),
                    }

        return analysis

    def save_approved_metadata(self):
        """Save approved metadata for StyleTTS2 compatibility"""
        approved_metadata = []

        for filename in self.review_state["approved"]:
            if filename in self.metadata:
                # Keep the original format that data_StyleTTS2.py expects
                metadata = self.metadata[filename].copy()
                # Add review information but keep original structure
                metadata["reviewed"] = True
                metadata["review_action"] = "approved"
                approved_metadata.append(metadata)

        # Save approved metadata in the format expected by data_StyleTTS2.py
        output_file = self.segments_dir / "approved_segments_metadata.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(approved_metadata, f, indent=2, ensure_ascii=False)

    def save_metadata(self):
        """Save updated metadata back to file"""
        metadata_list = list(self.metadata.values())
        metadata_file = self.segments_dir / "metadata.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata_list, f, indent=2, ensure_ascii=False)


# Global reviewer instance
reviewer = SegmentReviewer()


@app.route("/")
def index():
    """Main review interface"""
    current_segment = reviewer.get_current_segment()
    stats = reviewer.get_review_stats()

    if not current_segment:
        return render_template("completed.html", stats=stats)

    return render_template("review.html", segment=current_segment, stats=stats)


@app.route("/audio/<filename>")
def serve_audio(filename):
    """Serve audio files"""
    # Check in passed_segments first
    audio_path = reviewer.passed_dir / filename
    if audio_path.exists():
        return send_file(audio_path)

    # Check in other directories
    for directory in [
        reviewer.approved_dir,
        reviewer.rejected_dir,
        reviewer.skipped_dir,
    ]:
        audio_path = directory / filename
        if audio_path.exists():
            return send_file(audio_path)

    return "File not found", 404


@app.route("/review", methods=["POST"])
def review():
    """Handle review decision"""
    data = request.json
    filename = data.get("filename")
    action = data.get("action")
    notes = data.get("notes", "")
    edited_text = data.get("edited_text", "")

    if reviewer.review_segment(filename, action, notes, edited_text):
        return jsonify({"success": True})
    else:
        return jsonify({"success": False, "error": "Failed to process review"})


@app.route("/skip")
def skip_to_next():
    """Skip to next segment without reviewing"""
    reviewer.review_state["current_index"] += 1
    reviewer.save_review_state()
    return redirect(url_for("index"))


@app.route("/previous")
def go_to_previous():
    """Go to previous segment"""
    reviewer.review_state["current_index"] = max(
        0, reviewer.review_state["current_index"] - 1
    )
    reviewer.save_review_state()
    return redirect(url_for("index"))


@app.route("/stats")
def stats():
    """Get current statistics"""
    return jsonify(reviewer.get_review_stats())


@app.route("/analysis")
def analysis():
    """Get review analysis"""
    return jsonify(reviewer.analyze_review_patterns())


@app.route("/export_approved")
def export_approved():
    """Export approved segments metadata"""
    # Use the existing method to ensure consistency
    reviewer.save_approved_metadata()

    output_file = reviewer.segments_dir / "approved_segments_metadata.json"
    return send_file(output_file, as_attachment=True)


def check_dependencies():
    """Check if required dependencies are installed"""
    missing = []

    try:
        import flask
    except ImportError:
        missing.append("Flask")

    try:
        import numpy
    except ImportError:
        missing.append("numpy")

    if missing:
        print("❌ Missing required packages:")
        for pkg in missing:
            print(f"   - {pkg}")
        print("\n💡 Install with:")
        print(f"   pip install {' '.join(missing)}")
        return False

    return True


if __name__ == "__main__":
    print("🎵 Audio Segment Reviewer Starting...")
    print("=" * 50)

    # Check dependencies
    if not check_dependencies():
        input("Press Enter to exit...")
        exit(1)

    print(f"📁 Segments directory: {reviewer.segments_dir}")

    # Check if segments exist
    if not reviewer.segments_dir.exists():
        print("❌ Segments directory not found!")
        print("   Run the audio segmenter first to create segments")
        input("Press Enter to exit...")
        exit(1)

    stats = reviewer.get_review_stats()
    print(f"📊 Found {stats['remaining']} segments ready for review")

    if stats["remaining"] == 0:
        print("⚠️  No segments found to review!")
        print("   Either all segments have been reviewed or none passed quality checks")

    print("\n🌐 Starting web server...")
    print("   Open your browser and go to: http://localhost:5000")
    print("\n⌨️  Keyboard Shortcuts:")
    print("   Space - Play/Pause audio")
    print("   A - Approve segment")
    print("   R - Reject segment")
    print("   S - Skip segment")
    print("   ← → - Navigate segments")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 50)

    try:
        app.run(debug=False, host="0.0.0.0", port=5000)
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        print("\n💡 Common solutions:")
        print("   - Make sure port 5000 is not in use")
        print("   - Try running as administrator")
        print("   - Check if Flask is properly installed")
        input("\nPress Enter to exit...")
