#!/usr/bin/env python3
"""
Export collected data to standard Hugging Face dataset format for Whisper fine-tuning.

Creates a dataset compatible with the common speech recognition format used by
Hugging Face's datasets library.

IMPORTANT: This script creates JSONL files (whisper_train.jsonl, whisper_validation.jsonl,
whisper_test.jsonl) which is the required format for Hugging Face audio datasets.
"""

import json
import shutil
import random
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
    Export the dataset to Hugging Face JSONL format.

    Creates JSONL files with train/validation/test splits compatible with:
    - Hugging Face audio datasets
    - Whisper fine-tuning pipelines
    - Standard ASR training workflows

    Format:
    exported_dataset/
    ├── whisper_train.jsonl       # 70% of samples
    ├── whisper_validation.jsonl  # 15% of samples
    ├── whisper_test.jsonl        # 15% of samples
    ├── audio/
    │   ├── sample1.wav
    │   └── sample2.wav
    └── README.md
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

    # Shuffle samples for random splits
    random.seed(42)  # For reproducibility
    shuffled_samples = samples.copy()
    random.shuffle(shuffled_samples)

    # Calculate split sizes (70% train, 15% validation, 15% test)
    total = len(shuffled_samples)
    train_size = int(total * 0.70)
    val_size = int(total * 0.15)

    train_samples = shuffled_samples[:train_size]
    val_samples = shuffled_samples[train_size:train_size + val_size]
    test_samples = shuffled_samples[train_size + val_size:]

    print(f"\nDataset splits:")
    print(f"  Train: {len(train_samples)} samples ({len(train_samples)/total*100:.1f}%)")
    print(f"  Validation: {len(val_samples)} samples ({len(val_samples)/total*100:.1f}%)")
    print(f"  Test: {len(test_samples)} samples ({len(test_samples)/total*100:.1f}%)")
    print(f"  Total: {total} samples\n")

    # Process and copy audio files, create JSONL entries
    def process_sample(sample):
        """Process a sample and copy audio file."""
        audio_file = sample["audio_file"]

        # Copy audio file
        src_path = AUDIO_DIR / audio_file
        if src_path.exists():
            dst_path = export_audio_dir / audio_file
            shutil.copy2(src_path, dst_path)

            # Return JSONL-formatted entry
            return {
                "id": sample["id"],
                "audio_filepath": f"audio/{audio_file}",
                "text": sample["text"],
                "duration_seconds": sample.get("duration_seconds", 0),
                "sample_rate": sample.get("sample_rate", 16000),
                "style": sample.get("style", "unknown"),
                "source": sample.get("metadata", {}).get("source", "user"),
            }
        else:
            print(f"Warning: Audio file not found: {audio_file}")
            return None

    # Create JSONL files for each split
    splits = {
        "whisper_train.jsonl": train_samples,
        "whisper_validation.jsonl": val_samples,
        "whisper_test.jsonl": test_samples,
    }

    for filename, split_samples in splits.items():
        output_path = EXPORT_DIR / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in split_samples:
                entry = process_sample(sample)
                if entry:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        print(f"Created {filename} with {len(split_samples)} samples")

    # Create a README for the dataset
    total_duration = sum(s.get("duration_seconds", 0) for s in samples)
    styles = list(set(s.get("style", "unknown") for s in samples))

    readme_content = f"""---
license: mit
language:
- en
pretty_name: "Custom ASR Training Dataset"
size_categories:
- n<1K
task_categories:
- automatic-speech-recognition
---

# Custom ASR Training Dataset

Custom speech-to-text training data collected for Whisper fine-tuning.

## Dataset Statistics

- **Total Samples:** {len(samples)}
- **Total Duration:** {int(total_duration // 3600)}h {int((total_duration % 3600) // 60)}m {int(total_duration % 60)}s
- **Styles:** {', '.join(styles)}

## Dataset Structure

```
exported_dataset/
├── whisper_train.jsonl       # Training split (70%)
├── whisper_validation.jsonl  # Validation split (15%)
├── whisper_test.jsonl        # Test split (15%)
├── audio/                    # WAV audio files
└── README.md
```

## Dataset Splits

- **Training:** {len(train_samples)} samples
- **Validation:** {len(val_samples)} samples
- **Test:** {len(test_samples)} samples

## JSONL Schema

Each entry in the JSONL files contains:

```json
{{
  "id": "unique_id",
  "audio_filepath": "audio/filename.wav",
  "text": "transcription text",
  "duration_seconds": 15.5,
  "sample_rate": 16000,
  "style": "style_category",
  "source": "user"
}}
```

## Usage

This dataset is ready to upload to Hugging Face:

1. Create a new dataset on Hugging Face
2. Upload the JSONL files and audio directory
3. The dataset will automatically display with audio player

Load with Hugging Face datasets:

```python
from datasets import load_dataset

dataset = load_dataset("your-username/dataset-name")
```

## Format

- **Audio files:** WAV format, 16kHz sample rate
- **Transcriptions:** Plain text in JSONL files
- **Each sample includes:** audio path, transcription, duration, style tag, source

## Export Date

{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

*This dataset was created using the ASR Training Data Collector app.*
"""

    readme_path = EXPORT_DIR / "README.md"
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)

    print(f"\n✓ Dataset exported to: {EXPORT_DIR}")
    print(f"✓ Total samples: {len(samples)}")
    print(f"✓ Total duration: {int(total_duration // 60)}m {int(total_duration % 60)}s")
    print("\nFiles created:")
    print(f"  - whisper_train.jsonl ({len(train_samples)} samples)")
    print(f"  - whisper_validation.jsonl ({len(val_samples)} samples)")
    print(f"  - whisper_test.jsonl ({len(test_samples)} samples)")
    print(f"  - README.md")
    print(f"  - audio/ ({len(list(export_audio_dir.glob('*.wav')))} files)")
    print("\n✓ Dataset is ready to upload to Hugging Face!")


if __name__ == "__main__":
    export_to_hf_format()
