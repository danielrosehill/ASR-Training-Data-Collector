#!/usr/bin/env python3
"""
Export collected data to standard Hugging Face dataset format for Whisper fine-tuning.

Creates a dataset compatible with the common speech recognition format used by
Hugging Face's datasets library.
"""

import json
import shutil
from pathlib import Path
from datetime import datetime

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
AUDIO_DIR = DATA_DIR / "audio"
MANIFEST_FILE = DATA_DIR / "manifest.json"
EXPORT_DIR = BASE_DIR / "exported_dataset"


def export_to_hf_format():
    """
    Export the dataset to Hugging Face format.

    Creates a directory structure compatible with:
    - datasets.load_dataset("audiofolder")
    - Standard ASR training pipelines

    Format:
    exported_dataset/
    ├── metadata.csv
    ├── audio/
    │   ├── sample1.wav
    │   └── sample2.wav
    └── dataset_info.json
    """

    if not MANIFEST_FILE.exists():
        print("No manifest.json found. No data to export.")
        return

    with open(MANIFEST_FILE, 'r') as f:
        manifest = json.load(f)

    samples = manifest.get("samples", [])

    if not samples:
        print("No samples in manifest. Nothing to export.")
        return

    # Create export directory
    export_audio_dir = EXPORT_DIR / "audio"
    export_audio_dir.mkdir(parents=True, exist_ok=True)

    # Create metadata.csv (Hugging Face audiofolder format)
    metadata_lines = ["file_name,transcription,duration,style"]

    for sample in samples:
        audio_file = sample["audio_file"]
        text = sample["text"].replace('"', '""')  # Escape quotes for CSV
        duration = sample.get("duration_seconds", 0)
        style = sample.get("style", "unknown")

        # Copy audio file
        src_path = AUDIO_DIR / audio_file
        if src_path.exists():
            dst_path = export_audio_dir / audio_file
            shutil.copy2(src_path, dst_path)

            # Add to metadata
            metadata_lines.append(f'audio/{audio_file},"{text}",{duration},{style}')
        else:
            print(f"Warning: Audio file not found: {audio_file}")

    # Write metadata.csv
    metadata_path = EXPORT_DIR / "metadata.csv"
    with open(metadata_path, 'w') as f:
        f.write('\n'.join(metadata_lines))

    # Create dataset_info.json
    total_duration = sum(s.get("duration_seconds", 0) for s in samples)
    dataset_info = {
        "dataset_name": "whisper-fine-tuning-data",
        "description": "Custom speech-to-text training data for Whisper fine-tuning",
        "num_samples": len(samples),
        "total_duration_seconds": round(total_duration, 2),
        "total_duration_formatted": f"{int(total_duration // 3600)}h {int((total_duration % 3600) // 60)}m",
        "styles": list(set(s.get("style", "unknown") for s in samples)),
        "exported_at": datetime.now().isoformat(),
        "format": "audiofolder",
        "features": {
            "audio": {"dtype": "audio", "sampling_rate": 16000},
            "transcription": {"dtype": "string"},
            "duration": {"dtype": "float32"},
            "style": {"dtype": "string"}
        }
    }

    info_path = EXPORT_DIR / "dataset_info.json"
    with open(info_path, 'w') as f:
        json.dump(dataset_info, f, indent=2)

    # Create a README for the dataset
    readme_content = f"""# Whisper Fine-Tuning Dataset

Custom speech-to-text training data collected for Whisper fine-tuning.

## Dataset Statistics

- **Total Samples:** {len(samples)}
- **Total Duration:** {int(total_duration // 3600)}h {int((total_duration % 3600) // 60)}m {int(total_duration % 60)}s
- **Styles:** {', '.join(dataset_info['styles'])}

## Usage

Load with Hugging Face datasets:

```python
from datasets import load_dataset

dataset = load_dataset("audiofolder", data_dir="./exported_dataset")
```

## Format

- Audio files: WAV format, mono, various sample rates (will be resampled during training)
- Transcriptions: Plain text in metadata.csv
- Each sample includes: audio path, transcription, duration, style tag

## Export Date

{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

    readme_path = EXPORT_DIR / "README.md"
    with open(readme_path, 'w') as f:
        f.write(readme_content)

    print(f"Dataset exported to: {EXPORT_DIR}")
    print(f"Total samples: {len(samples)}")
    print(f"Total duration: {int(total_duration // 60)}m {int(total_duration % 60)}s")
    print("\nFiles created:")
    print(f"  - {metadata_path}")
    print(f"  - {info_path}")
    print(f"  - {readme_path}")
    print(f"  - {len(list(export_audio_dir.glob('*.wav')))} audio files")


if __name__ == "__main__":
    export_to_hf_format()
