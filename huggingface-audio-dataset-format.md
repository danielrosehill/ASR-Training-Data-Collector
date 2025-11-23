# Hugging Face Audio Dataset Format Guide

## Overview

This document describes the **required format** for audio datasets to properly display on Hugging Face with the audio player interface.

## The Secret Sauce 🎯

The key to a properly-displaying audio dataset on Hugging Face is:

**Use JSONL files with train/validation/test splits, NOT a single JSON manifest file.**

## Required Structure

```
your-dataset/
├── whisper_train.jsonl       # Training split (typically 70%)
├── whisper_validation.jsonl  # Validation split (typically 15%)
├── whisper_test.jsonl        # Test split (typically 15%)
├── audio/                    # Directory containing audio files
│   ├── sample1.wav
│   ├── sample2.wav
│   └── ...
├── text/                     # (Optional) Directory with text files
│   ├── sample1.txt
│   └── ...
└── README.md                 # Dataset documentation
```

## JSONL Format

Each line in the JSONL files must be a valid JSON object with these fields:

### Required Fields

```json
{
  "id": "unique_identifier",
  "audio_filepath": "audio/filename.wav",
  "text": "The transcription of the audio"
}
```

### Recommended Additional Fields

```json
{
  "id": "sample_001",
  "audio_filepath": "audio/sample_001.wav",
  "text": "The transcription of the audio",
  "duration_seconds": 15.5,
  "sample_rate": 16000,
  "style": "casual_conversation",
  "source": "user",
  "language": "en"
}
```

## Example JSONL Entry

```json
{"id": "a8b55d7b", "audio_filepath": "audio/20251123_192556_a8b55d7b.wav", "text": "In our recent project, we utilized Docker to streamline the deployment process.", "duration_seconds": 19.77, "sample_rate": 16000, "style": "formal_technical_explanation", "source": "llm"}
```

## What DOESN'T Work ❌

### ❌ Single JSON Manifest

This format will **NOT** display properly as an audio dataset:

```json
{
  "samples": [
    {
      "id": "sample1",
      "audio_file": "sample1.wav",
      "text": "Hello world"
    }
  ]
}
```

**Problem:** Hugging Face expects JSONL format with splits, not a JSON array.

### ❌ CSV Metadata Only

While metadata.csv works for some purposes, it doesn't provide the audio player functionality that JSONL files do.

## Split Ratios

Common split ratios for audio datasets:

- **Training:** 70% of samples
- **Validation:** 15% of samples
- **Test:** 15% of samples

For small datasets (<100 samples), you can adjust these ratios, but maintain all three files.

## Audio File Requirements

- **Format:** WAV, MP3, or FLAC
- **Sample Rate:** 16000 Hz is standard for ASR (Automatic Speech Recognition)
- **Channels:** Mono is preferred
- **Bit Depth:** 16-bit PCM

## README.md Front Matter

Include proper YAML front matter in your README.md:

```yaml
---
license: mit
language:
- en
pretty_name: "Your Dataset Name"
size_categories:
- n<1K
task_categories:
- automatic-speech-recognition
---
```

## Converting Existing Datasets

If you have a `manifest.json` file, use the conversion script:

```python
import json
import random

# Load manifest
with open('manifest.json', 'r') as f:
    data = json.load(f)

samples = data['samples']

# Shuffle and split
random.seed(42)
random.shuffle(samples)

total = len(samples)
train_size = int(total * 0.70)
val_size = int(total * 0.15)

train_samples = samples[:train_size]
val_samples = samples[train_size:train_size + val_size]
test_samples = samples[train_size + val_size:]

# Write JSONL files
splits = {
    "whisper_train.jsonl": train_samples,
    "whisper_validation.jsonl": val_samples,
    "whisper_test.jsonl": test_samples,
}

for filename, split_samples in splits.items():
    with open(filename, 'w', encoding='utf-8') as f:
        for sample in split_samples:
            # Convert to required format
            entry = {
                "id": sample["id"],
                "audio_filepath": f"audio/{sample['audio_file']}",
                "text": sample["text"],
                "duration_seconds": sample.get("duration_seconds", 0),
                "sample_rate": sample.get("sample_rate", 16000),
            }
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
```

## Verification Checklist

Before uploading to Hugging Face, verify:

- [ ] Three JSONL files exist: `whisper_train.jsonl`, `whisper_validation.jsonl`, `whisper_test.jsonl`
- [ ] Each JSONL file contains one JSON object per line
- [ ] `audio_filepath` field uses relative paths (e.g., `audio/filename.wav`)
- [ ] All referenced audio files exist in the `audio/` directory
- [ ] README.md includes proper YAML front matter
- [ ] Audio files are in a supported format (WAV/MP3/FLAC)

## Working Example

See the `English-Hebrew-Mixed-Sentences` dataset in this repository for a working reference implementation.

## Common Mistakes

1. **Using absolute paths** - Use `audio/file.wav`, not `/full/path/to/audio/file.wav`
2. **Missing splits** - All three JSONL files are required, even for small datasets
3. **Wrong file extension** - Must be `.jsonl`, not `.json`
4. **Nested JSON objects** - Each line should be a flat JSON object, not nested
5. **Missing audio directory prefix** - Use `audio/file.wav`, not just `file.wav`

## Testing Locally

Test your dataset structure before uploading:

```python
from datasets import load_dataset

# Load from local directory
dataset = load_dataset("json", data_files={
    "train": "whisper_train.jsonl",
    "validation": "whisper_validation.jsonl",
    "test": "whisper_test.jsonl"
})

print(dataset)
```

## Additional Resources

- [Hugging Face Audio Dataset Documentation](https://huggingface.co/docs/datasets/audio_dataset)
- [Datasets Library Documentation](https://huggingface.co/docs/datasets)
- [Whisper Fine-tuning Guide](https://huggingface.co/blog/fine-tune-whisper)

---

**Last Updated:** November 23, 2025
**Author:** Daniel Rosehill
