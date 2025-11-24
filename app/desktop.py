#!/usr/bin/env python3
"""
Whisper Fine-Tuning Data Collection - Desktop Application

A PyQt6 desktop application for collecting speech-to-text training data.
"""

import sys
import json
import os
import re
import uuid
import wave
import numpy as np
from datetime import datetime
from pathlib import Path
import requests
from dotenv import load_dotenv, dotenv_values
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QComboBox, QGroupBox, QMessageBox,
    QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
    QLineEdit, QDialog, QDialogButtonBox, QFormLayout, QCheckBox, QProgressBar,
    QFileDialog, QListWidget
)
from PyQt6.QtCore import Qt, QTimer, QBuffer, QIODevice, QUrl, QThread, pyqtSignal, QProcess, QSettings
from PyQt6.QtMultimedia import QMediaDevices, QAudioSource, QAudioFormat, QMediaPlayer, QAudioOutput
from PyQt6.QtGui import QShortcut, QKeySequence
import subprocess

API_KEY_PATTERN = re.compile(r"^[A-Za-z0-9_\-]+$")


def sanitize_api_key(raw_key):
    """Remove whitespace/invalid characters from API key and validate."""
    if not raw_key:
        return ""

    cleaned = raw_key.strip().strip('"').strip("'")
    cleaned = cleaned.replace("\r", "").replace("\n", "")

    if not API_KEY_PATTERN.fullmatch(cleaned):
        return ""

    return cleaned

# Paths - support both development and installed locations
APP_DIR = Path(__file__).parent

# Check for user data directory (set by launcher when installed as package)
if os.environ.get("WHISPER_FT_DATA_DIR"):
    DATA_DIR = Path(os.environ["WHISPER_FT_DATA_DIR"])
    BASE_DIR = DATA_DIR
else:
    # Development mode - use project directory
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"

# Dataset profiles configuration - must be in writable location
# In installed mode: ~/.local/share/whisper-finetuning-data/dataset_profiles.json
# In dev mode: project_root/data/dataset_profiles.json (for backwards compatibility with existing profiles)
if os.environ.get("WHISPER_FT_DATA_DIR"):
    # Installed mode - store in user data directory
    DATASET_PROFILES_FILE = DATA_DIR / "dataset_profiles.json"
else:
    # Development mode - check both locations for backwards compatibility
    # First try app dir (old location), then fall back to data dir
    old_location = APP_DIR / "dataset_profiles.json"
    new_location = DATA_DIR / "dataset_profiles.json"

    if old_location.exists() and not new_location.exists():
        # Migrate from old location to new location
        import shutil
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(old_location, new_location)
        print(f"Migrated dataset profiles from {old_location} to {new_location}")

    DATASET_PROFILES_FILE = new_location

# Initial paths (will be updated when profile is selected)
AUDIO_DIR = DATA_DIR / "audio"
TEXT_DIR = DATA_DIR / "text"
METADATA_DIR = DATA_DIR / "metadata"
MANIFEST_FILE = DATA_DIR / "manifest.json"

# Load environment configuration (data dir overrides, app dir fallback)
DATA_ENV_PATH = DATA_DIR / ".env"
APP_ENV_PATH = APP_DIR / ".env"

if DATA_ENV_PATH.exists():
    load_dotenv(DATA_ENV_PATH, override=True)

DEFAULT_API_KEY = ""
if APP_ENV_PATH.exists():
    DEFAULT_API_KEY = sanitize_api_key(
        dotenv_values(APP_ENV_PATH).get("OPENROUTER_API_KEY", "")
    )
    if not sanitize_api_key(os.environ.get("OPENROUTER_API_KEY", "")):
        load_dotenv(APP_ENV_PATH, override=True)
else:
    DEFAULT_API_KEY = ""

if not sanitize_api_key(os.environ.get("OPENROUTER_API_KEY", "")) and DEFAULT_API_KEY:
    os.environ["OPENROUTER_API_KEY"] = DEFAULT_API_KEY

# Ensure directories exist
for dir_path in [AUDIO_DIR, TEXT_DIR, METADATA_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

def cleanup_placeholder_files():
    """Remove leftover .gitkeep placeholders so counts match actual data."""
    for dir_path in [AUDIO_DIR, TEXT_DIR, METADATA_DIR]:
        placeholder = dir_path / ".gitkeep"
        if placeholder.exists():
            try:
                placeholder.unlink()
            except OSError:
                pass

cleanup_placeholder_files()

# Dataset profile management functions
def load_dataset_profiles():
    """Load dataset profiles from JSON file."""
    if DATASET_PROFILES_FILE.exists():
        try:
            with open(DATASET_PROFILES_FILE, 'r') as f:
                data = json.load(f)

                # Migrate old profiles without categories
                for profile in data.get("profiles", []):
                    if "categories" not in profile:
                        profile["categories"] = get_default_categories()

                return data
        except Exception as e:
            print(f"Error loading dataset profiles: {e}")

    # Default profile based on current DATA_DIR (relative to app location)
    # Use a relative path for portability
    default_path = DATA_DIR
    return {
        "profiles": [
            {
                "name": "Default Dataset",
                "path": str(default_path.resolve()),  # Use absolute path but based on runtime location
                "remote_url": "",
                "is_default": True,
                "categories": get_default_categories()
            }
        ],
        "active_profile": "Default Dataset"
    }

def save_dataset_profiles(profiles_data):
    """Save dataset profiles to JSON file."""
    try:
        with open(DATASET_PROFILES_FILE, 'w') as f:
            json.dump(profiles_data, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving dataset profiles: {e}")
        return False

# Default generalized categories for text generation
# These are used if no custom categories are defined for a profile

def get_default_categories():
    """Get default generalized categories for audio collection."""
    return {
        "styles": {
            "casual": "Casual, conversational tone",
            "formal": "Professional, business-appropriate tone",
            "informal": "Very informal, chatty",
            "narrative": "Story-telling, descriptive",
            "instructional": "Educational, step-by-step",
            "conversational": "Natural dialogue style"
        },
        "content_types": {
            "general": "General everyday topics",
            "technical": "Technical or specialized subject matter",
            "personal": "Personal notes and thoughts",
            "business": "Business and professional content",
            "creative": "Creative writing and ideas",
            "educational": "Learning and teaching content",
            "communication": "Messages and correspondence",
            "documentation": "Documentation and reference material"
        },
        "formats": {
            "monologue": "Single speaker, continuous",
            "conversation": "One side of a conversation",
            "note": "Quick note or memo",
            "explanation": "Explaining or describing something",
            "list": "List or enumeration",
            "message": "Written message format",
            "dictation": "Dictating text to be written"
        }
    }

# Legacy global dictionaries for backwards compatibility
STYLES = get_default_categories()["styles"]
CONTENT_TYPES = get_default_categories()["content_types"]
FORMATS = get_default_categories()["formats"]

# Hebrew-English terms (transliterated Hebrew used in English speech)
HEBREW_ENGLISH_TERMS = [
    "Teudat Zehut", "Kupat Holim", "Misrad Hapnim", "Bituach Leumi",
    "Arnona", "Vaad Bayit", "Iriya", "Misrad Haklita", "Sal Klita",
    "Ulpan", "Sherut Leumi", "Tzav Rishon", "Miluim", "Knesset",
    "Mashkanta", "Tik Pensiya", "Keren Hishtalmut", "Mas Hachnasa",
    "Beit Din", "Rabbanut", "Chuppa", "Brit Mila", "Bar Mitzvah",
    "Shabbat", "Chag", "Kibbutz", "Moshav", "Makolet", "Shuk",
    "Tremp", "Sherut", "Monit", "Rosh Hashana", "Yom Kippur",
    "Pesach", "Sukkot", "Chanukah", "Purim", "Tisha B'Av"
]

# Development/Technical terms
DEV_TECHNICAL_TERMS = [
    "Docker", "Kubernetes", "GitHub", "GitLab", "Hugging Face",
    "Python", "JavaScript", "TypeScript", "Node.js", "React", "Vue",
    "API", "REST", "GraphQL", "OAuth", "JWT", "SSL", "TLS",
    "PostgreSQL", "MongoDB", "Redis", "Elasticsearch", "RabbitMQ",
    "AWS", "Azure", "GCP", "Vercel", "Netlify", "Cloudflare",
    "CI/CD", "DevOps", "MLOps", "LLM", "GPT", "BERT", "transformer",
    "PyTorch", "TensorFlow", "Pandas", "NumPy", "Scikit-learn",
    "Linux", "Ubuntu", "Bash", "SSH", "SFTP", "nginx", "Apache",
    "VSCode", "Vim", "Neovim", "tmux", "conda", "pip", "npm", "yarn",
    "JSON", "YAML", "XML", "CSV", "Parquet", "Markdown", "LaTeX",
    "regex", "cron", "systemd", "journalctl", "htop", "grep", "awk"
]

class TextGeneratorWorker(QThread):
    """Worker thread for generating text via API."""
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, api_key, system_prompt, user_prompt):
        super().__init__()
        self.api_key = api_key
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt

    def run(self):
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "openai/gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": self.user_prompt}
                    ],
                    "max_tokens": 150,
                    "temperature": 1.6,
                    "top_p": 0.98,
                    "frequency_penalty": 1.2,
                    "presence_penalty": 1.0
                },
                timeout=30
            )

            if response.status_code == 200:
                text = response.json()["choices"][0]["message"]["content"].strip().strip('"\'')
                self.finished.emit(text)
            else:
                self.error.emit(f"API error: {response.status_code}")

        except Exception as e:
            self.error.emit(str(e))


class OllamaTextGeneratorWorker(QThread):
    """Worker thread for generating text via local Ollama."""
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, model, system_prompt, user_prompt):
        super().__init__()
        self.model = model
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt

    def run(self):
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.model,
                    "prompt": f"{self.system_prompt}\n\n{self.user_prompt}",
                    "stream": False,
                    "options": {
                        "temperature": 1.0,
                        "top_p": 0.9,
                        "num_predict": 100
                    }
                },
                timeout=60
            )

            if response.status_code == 200:
                text = response.json()["response"].strip().strip('"\'')
                self.finished.emit(text)
            else:
                self.error.emit(f"Ollama error: {response.status_code}")

        except Exception as e:
            self.error.emit(f"Ollama error: {str(e)}")


class HFSyncWorker(QThread):
    """Worker thread for syncing data to Hugging Face."""
    progress = pyqtSignal(str)  # Progress updates
    finished = pyqtSignal(dict)  # Final result with stats
    error = pyqtSignal(str)

    def __init__(self, data_dir):
        super().__init__()
        self.data_dir = data_dir

    def run(self):
        try:
            # Count local files before sync
            audio_dir = self.data_dir / "audio"
            local_audio_count = len(list(audio_dir.glob("*.wav"))) if audio_dir.exists() else 0

            self.progress.emit(f"Local files: {local_audio_count} audio samples")

            # Check git status to see what will be added
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=self.data_dir,
                capture_output=True,
                text=True
            )

            # Count new/modified files
            status_lines = [l for l in result.stdout.strip().split('\n') if l]
            new_files = len([l for l in status_lines if l.startswith('?') or l.startswith('A')])
            modified_files = len([l for l in status_lines if l.startswith('M')])

            if not status_lines or result.stdout.strip() == '':
                self.progress.emit("No changes to sync")
                # Get remote count anyway
                remote_count = self._get_remote_count()
                self.finished.emit({
                    "success": True,
                    "new_files": 0,
                    "modified_files": 0,
                    "local_count": local_audio_count,
                    "remote_count": remote_count,
                    "message": "Already up to date"
                })
                return

            self.progress.emit(f"Found {new_files} new, {modified_files} modified files")

            # Stage all changes with timeout (can be slow for many audio files)
            self.progress.emit(f"Staging {new_files + modified_files} files (this may take a moment)...")
            stage_result = subprocess.run(
                ["git", "add", "-A"],
                cwd=self.data_dir,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout for large batches
            )
            if stage_result.returncode != 0:
                self.error.emit(f"Staging failed: {stage_result.stderr}")
                return

            self.progress.emit("Files staged successfully")

            # Commit with timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            commit_msg = f"Sync {new_files} new files - {timestamp}"

            self.progress.emit("Committing...")
            subprocess.run(
                ["git", "commit", "-m", commit_msg],
                cwd=self.data_dir,
                capture_output=True,
                check=True
            )

            # Push to HF
            self.progress.emit("Pushing to Hugging Face...")
            push_result = subprocess.run(
                ["git", "push"],
                cwd=self.data_dir,
                capture_output=True,
                text=True,
                timeout=120
            )

            if push_result.returncode != 0:
                self.error.emit(f"Push failed: {push_result.stderr}")
                return

            # Verify remote count
            self.progress.emit("Verifying remote...")
            remote_count = self._get_remote_count()

            self.finished.emit({
                "success": True,
                "new_files": new_files,
                "modified_files": modified_files,
                "local_count": local_audio_count,
                "remote_count": remote_count,
                "message": f"Successfully synced {new_files} files"
            })

        except subprocess.TimeoutExpired:
            self.error.emit("Sync timed out - check your connection")
        except subprocess.CalledProcessError as e:
            self.error.emit(f"Git error: {e.stderr if e.stderr else str(e)}")
        except Exception as e:
            self.error.emit(str(e))

    def _get_remote_count(self):
        """Get count of audio files in the remote HF dataset."""
        try:
            # First fetch to update remote refs
            subprocess.run(
                ["git", "fetch", "origin"],
                cwd=self.data_dir,
                capture_output=True,
                timeout=30
            )

            # Use git ls-tree on origin/main to count files actually on remote
            result = subprocess.run(
                ["git", "ls-tree", "-r", "origin/main", "--name-only", "audio/"],
                cwd=self.data_dir,
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                files = [f for f in result.stdout.strip().split('\n') if f.endswith('.wav')]
                return len(files)
        except Exception:
            pass
        return -1  # Unknown


class SettingsDialog(QDialog):
    """Dialog for application settings."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(400)

        self.settings = QSettings("WhisperFinetuning", "DataCollector")

        layout = QVBoxLayout(self)

        # Form layout for settings
        form = QFormLayout()

        # API Key input
        self.api_key_input = QLineEdit()
        self.api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_input.setPlaceholderText("Enter your OpenRouter API key...")

        # Load existing key
        saved_key = sanitize_api_key(self.settings.value("openrouter_api_key", ""))
        if saved_key:
            self.api_key_input.setText(saved_key)
        else:
            env_key = sanitize_api_key(os.environ.get("OPENROUTER_API_KEY", ""))
            fallback_key = env_key or DEFAULT_API_KEY
            if fallback_key:
                self.api_key_input.setText(fallback_key)

        form.addRow("OpenRouter API Key:", self.api_key_input)

        # Show/hide toggle
        self.show_key_btn = QPushButton("Show")
        self.show_key_btn.setCheckable(True)
        self.show_key_btn.toggled.connect(self.toggle_key_visibility)
        form.addRow("", self.show_key_btn)

        layout.addLayout(form)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.save_settings)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def toggle_key_visibility(self, checked):
        if checked:
            self.api_key_input.setEchoMode(QLineEdit.EchoMode.Normal)
            self.show_key_btn.setText("Hide")
        else:
            self.api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
            self.show_key_btn.setText("Show")

    def save_settings(self):
        raw_key = self.api_key_input.text()

        if not raw_key.strip():
            self.settings.remove("openrouter_api_key")
            if DEFAULT_API_KEY:
                os.environ["OPENROUTER_API_KEY"] = DEFAULT_API_KEY
            else:
                os.environ.pop("OPENROUTER_API_KEY", None)
            self.accept()
            return

        api_key = sanitize_api_key(raw_key)
        if not api_key:
            QMessageBox.warning(
                self,
                "Invalid API Key",
                "The OpenRouter API key you entered contains invalid characters.\n"
                "Please copy only the key itself (looks like 'sk-...') and try again."
            )
            return

        self.settings.setValue("openrouter_api_key", api_key)
        os.environ["OPENROUTER_API_KEY"] = api_key
        self.accept()


class DatasetProfileDialog(QDialog):
    """Dialog for adding/editing a dataset profile."""

    def __init__(self, parent=None, profile=None):
        super().__init__(parent)
        self.profile = profile
        self.setWindowTitle("Add Dataset Profile" if not profile else "Edit Dataset Profile")
        self.setMinimumWidth(500)

        layout = QVBoxLayout(self)

        # Info label
        info = QLabel(
            "Dataset profiles let you work with multiple datasets. "
            "Each profile has its own audio categories that you can customize."
        )
        info.setWordWrap(True)
        info.setStyleSheet("padding: 8px; background: #e3f2fd; border-radius: 5px; margin-bottom: 10px;")
        layout.addWidget(info)

        # Form layout
        form = QFormLayout()

        # Profile name
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("e.g., Personal Dataset, Work Dataset")
        if profile:
            self.name_input.setText(profile.get("name", ""))
        form.addRow("Profile Name:", self.name_input)

        # Dataset path with browse button
        path_row = QHBoxLayout()
        self.path_input = QLineEdit()
        self.path_input.setPlaceholderText("/path/to/dataset/folder")
        if profile:
            self.path_input.setText(profile.get("path", ""))
        path_row.addWidget(self.path_input)

        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_path)
        path_row.addWidget(browse_btn)
        form.addRow("Dataset Path:", path_row)

        # Remote URL (auto-detected)
        self.remote_input = QLineEdit()
        self.remote_input.setPlaceholderText("Will be auto-detected from git remote")
        self.remote_input.setReadOnly(True)
        if profile:
            self.remote_input.setText(profile.get("remote_url", ""))
        form.addRow("Remote URL:", self.remote_input)

        detect_btn = QPushButton("Detect Remote from Path")
        detect_btn.clicked.connect(self.detect_remote)
        form.addRow("", detect_btn)

        layout.addLayout(form)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.validate_and_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def browse_path(self):
        """Open folder browser for dataset path."""
        path = QFileDialog.getExistingDirectory(
            self,
            "Select Dataset Folder",
            str(Path.home())
        )
        if path:
            self.path_input.setText(path)
            # Auto-detect remote when path is selected
            self.detect_remote()

    def detect_remote(self):
        """Detect git remote URL from the selected path."""
        path = self.path_input.text().strip()
        if not path:
            QMessageBox.warning(self, "Warning", "Please select a dataset path first")
            return

        path_obj = Path(path)
        if not path_obj.exists():
            QMessageBox.warning(self, "Warning", "Selected path does not exist")
            return

        # Check if it's a git repo
        try:
            result = subprocess.run(
                ["git", "remote", "get-url", "origin"],
                cwd=path_obj,
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                remote_url = result.stdout.strip()
                self.remote_input.setText(remote_url)
                QMessageBox.information(
                    self,
                    "Remote Detected",
                    f"Found remote:\n{remote_url}"
                )
            else:
                QMessageBox.warning(
                    self,
                    "No Remote Found",
                    "This folder is not a git repository or has no remote configured.\n\n"
                    "The dataset should be a git repository with a remote URL configured."
                )
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to detect remote: {e}")

    def validate_and_accept(self):
        """Validate inputs before accepting."""
        name = self.name_input.text().strip()
        path = self.path_input.text().strip()

        if not name:
            QMessageBox.warning(self, "Warning", "Please enter a profile name")
            return

        if not path:
            QMessageBox.warning(self, "Warning", "Please select a dataset path")
            return

        path_obj = Path(path)
        if not path_obj.exists():
            reply = QMessageBox.question(
                self,
                "Path Does Not Exist",
                f"The path '{path}' does not exist.\n\nCreate it?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                try:
                    path_obj.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to create directory: {e}")
                    return
            else:
                return

        self.accept()

    def get_profile_data(self):
        """Get the profile data from inputs."""
        # Get categories from profile or use defaults
        categories = self.profile.get("categories", get_default_categories()) if self.profile else get_default_categories()

        return {
            "name": self.name_input.text().strip(),
            "path": self.path_input.text().strip(),
            "remote_url": self.remote_input.text().strip(),
            "is_default": False,
            "categories": categories
        }


class DatasetProfileManager(QDialog):
    """Dialog for managing dataset profiles."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Manage Dataset Profiles")
        self.setMinimumSize(600, 400)

        self.profiles_data = load_dataset_profiles()

        layout = QVBoxLayout(self)

        # Info label
        info = QLabel(
            "Dataset profiles allow you to switch between different dataset locations.\n"
            "Each profile stores its location and git remote URL."
        )
        info.setWordWrap(True)
        info.setStyleSheet("padding: 10px; background: #e3f2fd; border-radius: 5px;")
        layout.addWidget(info)

        # List of profiles
        list_label = QLabel("Saved Profiles:")
        list_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        layout.addWidget(list_label)

        self.profiles_list = QListWidget()
        self.refresh_profiles_list()
        layout.addWidget(self.profiles_list)

        # Buttons
        btn_row = QHBoxLayout()

        add_btn = QPushButton("➕ Add Profile")
        add_btn.clicked.connect(self.add_profile)
        btn_row.addWidget(add_btn)

        edit_btn = QPushButton("✏️ Edit Profile")
        edit_btn.clicked.connect(self.edit_profile)
        btn_row.addWidget(edit_btn)

        delete_btn = QPushButton("🗑️ Delete Profile")
        delete_btn.clicked.connect(self.delete_profile)
        btn_row.addWidget(delete_btn)

        btn_row.addStretch()
        layout.addLayout(btn_row)

        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)

    def refresh_profiles_list(self):
        """Refresh the profiles list display."""
        self.profiles_list.clear()
        active_profile = self.profiles_data.get("active_profile", "")

        for profile in self.profiles_data.get("profiles", []):
            name = profile.get("name", "Unnamed")
            path = profile.get("path", "")
            is_active = (name == active_profile)

            display_text = f"{'✓ ' if is_active else ''}{name} - {path}"
            if is_active:
                display_text = f"<b>{display_text}</b>"

            self.profiles_list.addItem(display_text)

    def add_profile(self):
        """Add a new profile."""
        dialog = DatasetProfileDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            new_profile = dialog.get_profile_data()

            # Check for duplicate names
            existing_names = [p.get("name", "") for p in self.profiles_data.get("profiles", [])]
            if new_profile["name"] in existing_names:
                QMessageBox.warning(
                    self,
                    "Duplicate Name",
                    f"A profile named '{new_profile['name']}' already exists."
                )
                return

            self.profiles_data["profiles"].append(new_profile)
            if not save_dataset_profiles(self.profiles_data):
                QMessageBox.critical(
                    self,
                    "Save Failed",
                    f"Could not save dataset profiles to:\n{DATASET_PROFILES_FILE}\n\n"
                    "Check that the directory is writable."
                )
                return
            self.refresh_profiles_list()
            QMessageBox.information(self, "Success", f"Profile '{new_profile['name']}' added!")

    def edit_profile(self):
        """Edit the selected profile."""
        current_row = self.profiles_list.currentRow()
        if current_row < 0:
            QMessageBox.warning(self, "Warning", "Please select a profile to edit")
            return

        profile = self.profiles_data["profiles"][current_row]
        dialog = DatasetProfileDialog(self, profile)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            updated_profile = dialog.get_profile_data()

            # Check for duplicate names (excluding current profile)
            existing_names = [
                p.get("name", "") for i, p in enumerate(self.profiles_data["profiles"])
                if i != current_row
            ]
            if updated_profile["name"] in existing_names:
                QMessageBox.warning(
                    self,
                    "Duplicate Name",
                    f"A profile named '{updated_profile['name']}' already exists."
                )
                return

            self.profiles_data["profiles"][current_row] = updated_profile
            if not save_dataset_profiles(self.profiles_data):
                QMessageBox.critical(
                    self,
                    "Save Failed",
                    f"Could not save dataset profiles to:\n{DATASET_PROFILES_FILE}\n\n"
                    "Check that the directory is writable."
                )
                return
            self.refresh_profiles_list()
            QMessageBox.information(self, "Success", "Profile updated!")

    def delete_profile(self):
        """Delete the selected profile."""
        current_row = self.profiles_list.currentRow()
        if current_row < 0:
            QMessageBox.warning(self, "Warning", "Please select a profile to delete")
            return

        profile = self.profiles_data["profiles"][current_row]
        profile_name = profile.get("name", "")

        # Don't allow deleting if it's the only profile
        if len(self.profiles_data["profiles"]) == 1:
            QMessageBox.warning(
                self,
                "Cannot Delete",
                "You must have at least one dataset profile."
            )
            return

        # Confirm deletion
        reply = QMessageBox.question(
            self,
            "Confirm Delete",
            f"Delete profile '{profile_name}'?\n\nThis will not delete the actual dataset files.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        # If deleting active profile, switch to first remaining profile
        if self.profiles_data.get("active_profile") == profile_name:
            remaining_profiles = [
                p for i, p in enumerate(self.profiles_data["profiles"]) if i != current_row
            ]
            if remaining_profiles:
                self.profiles_data["active_profile"] = remaining_profiles[0]["name"]

        del self.profiles_data["profiles"][current_row]
        if not save_dataset_profiles(self.profiles_data):
            QMessageBox.critical(
                self,
                "Save Failed",
                f"Could not save dataset profiles to:\n{DATASET_PROFILES_FILE}\n\n"
                "Check that the directory is writable."
            )
            return
        self.refresh_profiles_list()
        QMessageBox.information(self, "Success", f"Profile '{profile_name}' deleted!")


class AudioRecorder:
    """Handle audio recording using Qt Multimedia."""

    def __init__(self):
        self.audio_source = None
        self.buffer = None
        self.is_recording = False
        self.sample_rate = 16000
        self.selected_device = None

    def set_device(self, device):
        """Set the audio input device to use."""
        self.selected_device = device

    def start_recording(self):
        """Start recording audio."""
        format = QAudioFormat()
        format.setSampleRate(self.sample_rate)
        format.setChannelCount(1)
        format.setSampleFormat(QAudioFormat.SampleFormat.Int16)

        # Use selected device or default
        if self.selected_device and not self.selected_device.isNull():
            device = self.selected_device
        else:
            device = QMediaDevices.defaultAudioInput()

        if device.isNull():
            raise RuntimeError("No audio input device found")

        self.buffer = QBuffer()
        self.buffer.open(QIODevice.OpenModeFlag.WriteOnly)

        self.audio_source = QAudioSource(device, format)
        self.audio_source.start(self.buffer)
        self.is_recording = True

    def stop_recording(self):
        """Stop recording and return audio data."""
        if not self.is_recording:
            return None, 0

        self.audio_source.stop()
        self.buffer.close()
        self.is_recording = False

        data = self.buffer.data()
        audio_array = np.frombuffer(data.data(), dtype=np.int16)

        return audio_array, self.sample_rate


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()

        # Dataset profiles
        self.profiles_data = load_dataset_profiles()
        self.current_data_dir = None
        self.current_audio_dir = None
        self.current_text_dir = None
        self.current_metadata_dir = None
        self.current_manifest_file = None

        # Current profile categories (will be loaded from active profile)
        self.current_styles = {}
        self.current_content_types = {}
        self.current_formats = {}

        self.recorder = AudioRecorder()
        self.current_text = ""
        self.recording_timer = QTimer()
        self.recording_seconds = 0

        # Audio playback
        self.media_player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.media_player.setAudioOutput(self.audio_output)

        # Track available audio devices
        self.audio_devices = []

        # Text generation worker
        self.text_worker = None

        # HF sync worker
        self.sync_worker = None

        # Load corpus and word list
        self.corpus_sentences = []
        self.custom_words = []
        self.load_corpus()
        self.load_word_list()

        # Session statistics
        self.session_samples = 0
        self.session_start_time = datetime.now()

        # Auto-record countdown timer
        self.auto_record_timer = QTimer()
        self.auto_record_timer.timeout.connect(self.auto_record_tick)
        self.auto_record_countdown = 0
        self.pending_auto_record = False  # Flag to start countdown after text generation

        self.init_ui()

        # Load active profile after UI is initialized
        self.load_active_profile()

    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Whisper Fine-Tuning Data Collector")
        self.setMinimumSize(900, 650)

        # Central widget with tabs
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # Top bar with dataset selector and settings
        top_bar = QHBoxLayout()

        # Dataset selector
        dataset_label = QLabel("Dataset:")
        dataset_label.setStyleSheet("font-weight: bold;")
        top_bar.addWidget(dataset_label)

        self.dataset_combo = QComboBox()
        self.dataset_combo.setMinimumWidth(200)
        self.populate_dataset_combo()
        self.dataset_combo.currentIndexChanged.connect(self.on_dataset_changed)
        top_bar.addWidget(self.dataset_combo)

        manage_datasets_btn = QPushButton("Manage Datasets...")
        manage_datasets_btn.clicked.connect(self.show_dataset_manager)
        top_bar.addWidget(manage_datasets_btn)

        top_bar.addStretch()

        help_btn = QPushButton("❓ Keyboard Shortcuts")
        help_btn.clicked.connect(self.show_shortcuts_help)
        top_bar.addWidget(help_btn)

        settings_btn = QPushButton("⚙ Settings")
        settings_btn.clicked.connect(self.show_settings)
        top_bar.addWidget(settings_btn)
        main_layout.addLayout(top_bar)

        # Tab widget
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # Record tab
        record_tab = QWidget()
        self.setup_record_tab(record_tab)
        self.tabs.addTab(record_tab, "Record")

        # Manage tab
        manage_tab = QWidget()
        self.setup_manage_tab(manage_tab)
        self.tabs.addTab(manage_tab, "Manage")

        # Sync tab
        sync_tab = QWidget()
        self.setup_sync_tab(sync_tab)
        self.tabs.addTab(sync_tab, "Sync to HF")

        # Timer for recording duration
        self.recording_timer.timeout.connect(self.update_timer)

        # Setup keyboard shortcuts
        self.setup_shortcuts()

        # Initial stats
        self.update_statistics()

    def on_mode_changed(self, index):
        """Handle mode change between LLM and Manual."""
        mode = self.mode_combo.currentData()

        if mode == "manual":
            # Show manual controls, hide LLM controls
            self.manual_controls.setVisible(True)
            self.llm_controls.setVisible(False)
            self.llm_info_row.setVisible(False)
            self.skip_btn.setVisible(False)
            self.status_label.setText("Enter text manually and click 'Set Text for Recording'")
        else:
            # Show LLM controls, hide manual controls
            self.manual_controls.setVisible(False)
            self.llm_controls.setVisible(True)
            self.llm_info_row.setVisible(True)
            self.skip_btn.setVisible(True)
            self.status_label.setText("Click 'Generate New Text' to get started")

    def set_manual_text(self):
        """Set manually entered text as the current text to record."""
        text = self.manual_text_input.toPlainText().strip()

        if not text:
            QMessageBox.warning(self, "Warning", "Please enter some text first")
            return

        if len(text) < 5:
            QMessageBox.warning(self, "Warning", "Text is too short. Please enter at least 5 characters.")
            return

        # Estimate speaking time (rough: 150 words per minute = 2.5 words per second)
        word_count = len(text.split())
        estimated_seconds = word_count / 2.5

        if estimated_seconds > 35:
            reply = QMessageBox.question(
                self, "Long Text Warning",
                f"This text has ~{word_count} words which may take ~{estimated_seconds:.0f} seconds to read.\n"
                f"Recordings must be under 35 seconds.\n\nContinue anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        self.current_text = text
        self.text_display.setText(text)
        self.status_label.setText(f"✓ Text set ({word_count} words, ~{estimated_seconds:.0f}s) - Ready to record (Press Space)")

    def setup_shortcuts(self):
        """Setup keyboard shortcuts for efficient workflow."""
        # Record tab shortcuts
        QShortcut(QKeySequence("Space"), self).activated.connect(self.toggle_recording)
        QShortcut(QKeySequence("Ctrl+S"), self).activated.connect(self.save_recording)
        QShortcut(QKeySequence("Ctrl+Return"), self).activated.connect(self.save_and_auto_record)
        QShortcut(QKeySequence("Ctrl+D"), self).activated.connect(self.confirm_discard_recording)
        QShortcut(QKeySequence("Ctrl+N"), self).activated.connect(self.generate_text)
        QShortcut(QKeySequence("Ctrl+G"), self).activated.connect(self.generate_text)
        QShortcut(QKeySequence("Escape"), self).activated.connect(self.cancel_auto_record)

    def setup_record_tab(self, tab):
        """Set up the recording tab."""
        layout = QHBoxLayout(tab)

        # Left panel - recording
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # Advanced settings (collapsible) - moved mic and mode here
        self.advanced_settings_group = QGroupBox("⚙ Advanced Settings")
        self.advanced_settings_group.setCheckable(True)
        self.advanced_settings_group.setChecked(False)  # Collapsed by default
        advanced_layout = QVBoxLayout(self.advanced_settings_group)

        # Microphone selector (moved to advanced)
        mic_row = QHBoxLayout()
        self.mic_combo = QComboBox()
        self.populate_audio_devices()
        self.mic_combo.currentIndexChanged.connect(self.on_mic_changed)
        mic_row.addWidget(QLabel("Microphone:"))
        mic_row.addWidget(self.mic_combo, 1)

        refresh_mic_btn = QPushButton("Refresh")
        refresh_mic_btn.clicked.connect(self.populate_audio_devices)
        mic_row.addWidget(refresh_mic_btn)
        advanced_layout.addLayout(mic_row)

        # Mode selector (LLM vs Manual) - moved to advanced
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("🤖 LLM Generated", "llm")
        self.mode_combo.addItem("✍️ Manual Entry", "manual")
        self.mode_combo.currentIndexChanged.connect(self.on_mode_changed)
        mode_row.addWidget(self.mode_combo)
        mode_row.addStretch()
        advanced_layout.addLayout(mode_row)

        left_layout.addWidget(self.advanced_settings_group)

        # Style selector and generate button (for LLM mode)
        self.llm_controls = QWidget()
        llm_layout = QVBoxLayout(self.llm_controls)
        llm_layout.setContentsMargins(0, 0, 0, 0)

        # Three separate selection rows
        # Row 1: Style (tone/voice)
        style_row = QHBoxLayout()
        style_row.addWidget(QLabel("Style (tone):"))
        self.style_combo = QComboBox()
        # Will be populated when profile is loaded
        style_row.addWidget(self.style_combo, 1)
        llm_layout.addLayout(style_row)

        # Row 2: Content (topic/domain)
        content_row = QHBoxLayout()
        content_row.addWidget(QLabel("Content:"))
        self.content_combo = QComboBox()
        # Will be populated when profile is loaded
        content_row.addWidget(self.content_combo, 1)
        llm_layout.addLayout(content_row)

        # Row 3: Format (delivery type)
        format_row = QHBoxLayout()
        format_row.addWidget(QLabel("Format:"))
        self.format_combo = QComboBox()
        # Will be populated when profile is loaded
        format_row.addWidget(self.format_combo, 1)
        llm_layout.addLayout(format_row)

        # Generate button
        gen_btn_row = QHBoxLayout()
        self.generate_btn = QPushButton("Generate New Text")
        self.generate_btn.clicked.connect(self.generate_text)
        self.generate_btn.setStyleSheet("font-weight: bold; background-color: #2196F3; color: white;")
        gen_btn_row.addWidget(self.generate_btn)
        llm_layout.addLayout(gen_btn_row)

        left_layout.addWidget(self.llm_controls)

        # LLM mode info and progress
        self.llm_info_row = QWidget()
        llm_info_layout = QVBoxLayout(self.llm_info_row)
        llm_info_layout.setContentsMargins(0, 0, 0, 0)

        info_row = QHBoxLayout()
        mode_label = QLabel("✓ Using OpenRouter Cloud API (gpt-4o-mini) - Fast, reliable")
        mode_label.setStyleSheet("color: blue; font-weight: bold;")
        info_row.addWidget(mode_label)
        info_row.addStretch()
        llm_info_layout.addLayout(info_row)

        # Progress bar for text generation
        self.generation_progress = QProgressBar()
        self.generation_progress.setVisible(False)
        self.generation_progress.setRange(0, 0)  # Indeterminate
        self.generation_progress.setTextVisible(False)
        self.generation_progress.setMaximumHeight(3)
        llm_info_layout.addWidget(self.generation_progress)

        # Dictionary words input (support up to 3 words)
        word_row = QHBoxLayout()
        word_row.addWidget(QLabel("Dictionary Words:"))
        self.word_input = QLineEdit()
        self.word_input.setPlaceholderText("Optional: Enter 1-3 words (comma-separated) to include in generated text...")
        word_row.addWidget(self.word_input, 1)

        self.clear_word_btn = QPushButton("Clear")
        self.clear_word_btn.clicked.connect(lambda: self.word_input.clear())
        self.clear_word_btn.setFixedWidth(60)
        word_row.addWidget(self.clear_word_btn)
        llm_info_layout.addLayout(word_row)

        # Disambiguation/context input for technical terms
        disambig_row = QHBoxLayout()
        disambig_row.addWidget(QLabel("Context/Clarification:"))
        self.disambig_input = QLineEdit()
        self.disambig_input.setPlaceholderText("Optional: Specify context or clarify meaning (e.g., 'Docker containerization', 'React framework')...")
        disambig_row.addWidget(self.disambig_input, 1)

        self.clear_disambig_btn = QPushButton("Clear")
        self.clear_disambig_btn.clicked.connect(lambda: self.disambig_input.clear())
        self.clear_disambig_btn.setFixedWidth(60)
        disambig_row.addWidget(self.clear_disambig_btn)
        llm_info_layout.addLayout(disambig_row)

        left_layout.addWidget(self.llm_info_row)

        # Manual entry controls
        self.manual_controls = QWidget()
        manual_layout = QVBoxLayout(self.manual_controls)
        manual_layout.setContentsMargins(0, 0, 0, 0)

        manual_info = QLabel("Enter or paste the text you want to record:")
        manual_info.setStyleSheet("color: #666; font-style: italic;")
        manual_layout.addWidget(manual_info)

        # Manual text input area
        self.manual_text_input = QTextEdit()
        self.manual_text_input.setPlaceholderText("Type or paste your text here...")
        self.manual_text_input.setMaximumHeight(100)
        manual_layout.addWidget(self.manual_text_input)

        # Set button for manual mode
        manual_btn_row = QHBoxLayout()
        self.set_manual_text_btn = QPushButton("✓ Set Text for Recording")
        self.set_manual_text_btn.clicked.connect(self.set_manual_text)
        self.set_manual_text_btn.setStyleSheet("font-weight: bold; background-color: #2196F3; color: white;")
        manual_btn_row.addWidget(self.set_manual_text_btn)

        self.clear_manual_text_btn = QPushButton("Clear")
        self.clear_manual_text_btn.clicked.connect(lambda: self.manual_text_input.clear())
        manual_btn_row.addWidget(self.clear_manual_text_btn)
        manual_layout.addLayout(manual_btn_row)

        left_layout.addWidget(self.manual_controls)
        self.manual_controls.setVisible(False)  # Hidden by default

        # Text display - larger and more readable
        self.text_display = QTextEdit()
        self.text_display.setReadOnly(True)
        self.text_display.setPlaceholderText("Click 'Generate New Text' to get text to read...")
        self.text_display.setMinimumHeight(150)
        self.text_display.setStyleSheet("""
            QTextEdit {
                font-size: 16px;
                line-height: 1.5;
                padding: 15px;
                background-color: #f8f9fa;
                border: 2px solid #dee2e6;
                border-radius: 5px;
            }
        """)
        left_layout.addWidget(self.text_display)

        # Recording controls
        record_group = QGroupBox("Recording")
        record_layout = QVBoxLayout(record_group)

        self.record_btn = QPushButton("🎙️ Start Recording (Space)")
        self.record_btn.clicked.connect(self.toggle_recording)
        self.record_btn.setMinimumHeight(50)
        self.record_btn.setStyleSheet("font-size: 14px; font-weight: bold;")
        record_layout.addWidget(self.record_btn)

        self.time_label = QLabel("0:00")
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.time_label.setStyleSheet("font-size: 24px; font-weight: bold;")
        record_layout.addWidget(self.time_label)

        left_layout.addWidget(record_group)

        # Action buttons
        btn_row = QHBoxLayout()

        self.save_btn = QPushButton("💾 Save Recording (Ctrl+S)")
        self.save_btn.clicked.connect(self.save_recording)
        self.save_btn.setEnabled(False)
        self.save_btn.setStyleSheet("font-weight: bold;")
        btn_row.addWidget(self.save_btn)

        self.save_and_next_btn = QPushButton("💾➡️🎙️ Save & Auto-Record (Ctrl+Enter)")
        self.save_and_next_btn.clicked.connect(self.save_and_auto_record)
        self.save_and_next_btn.setEnabled(False)
        self.save_and_next_btn.setStyleSheet("font-weight: bold; background-color: #4CAF50; color: white;")
        self.save_and_next_btn.setToolTip("Save, generate next text, and auto-start recording after 3s countdown")
        btn_row.addWidget(self.save_and_next_btn)

        self.discard_btn = QPushButton("🗑️ Discard (Ctrl+D)")
        self.discard_btn.clicked.connect(self.confirm_discard_recording)
        self.discard_btn.setEnabled(False)
        btn_row.addWidget(self.discard_btn)

        self.skip_btn = QPushButton("⏭️ Skip (Ctrl+N)")
        self.skip_btn.clicked.connect(self.generate_text)
        btn_row.addWidget(self.skip_btn)

        left_layout.addLayout(btn_row)

        # Status
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("padding: 5px; background: #f0f0f0; border-radius: 3px;")
        left_layout.addWidget(self.status_label)

        layout.addWidget(left_panel, 2)

        # Right panel - statistics
        right_panel = QGroupBox("Dataset Statistics")
        right_layout = QVBoxLayout(right_panel)

        # Session stats (prominent)
        session_group = QGroupBox("Current Session")
        session_layout = QVBoxLayout(session_group)
        self.session_stats_label = QLabel()
        self.session_stats_label.setWordWrap(True)
        self.session_stats_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #2196F3;")
        session_layout.addWidget(self.session_stats_label)
        right_layout.addWidget(session_group)

        # Overall stats
        self.stats_label = QLabel()
        self.stats_label.setWordWrap(True)
        self.stats_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        right_layout.addWidget(self.stats_label)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.update_statistics)
        right_layout.addWidget(refresh_btn)

        right_layout.addStretch()
        layout.addWidget(right_panel, 1)

    def setup_manage_tab(self, tab):
        """Set up the manage/edit tab."""
        layout = QVBoxLayout(tab)

        # Table for samples
        self.samples_table = QTableWidget()
        self.samples_table.setColumnCount(5)
        self.samples_table.setHorizontalHeaderLabels(["#", "Duration", "Style", "Text Preview", "Play"])
        self.samples_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.samples_table.setColumnWidth(4, 60)
        self.samples_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.samples_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        layout.addWidget(self.samples_table)

        # Buttons
        btn_row = QHBoxLayout()

        refresh_table_btn = QPushButton("Refresh List")
        refresh_table_btn.clicked.connect(self.refresh_samples_table)
        btn_row.addWidget(refresh_table_btn)

        delete_selected_btn = QPushButton("Delete Selected")
        delete_selected_btn.clicked.connect(self.delete_selected_samples)
        btn_row.addWidget(delete_selected_btn)

        btn_row.addStretch()
        layout.addLayout(btn_row)

        # Load initial data
        self.refresh_samples_table()

    def setup_sync_tab(self, tab):
        """Set up the Hugging Face sync tab."""
        layout = QVBoxLayout(tab)

        # Header info
        info_group = QGroupBox("Hugging Face Dataset")
        info_layout = QVBoxLayout(info_group)

        self.hf_dataset_label = QLabel("Dataset: [Configure in profile]")
        self.hf_dataset_label.setStyleSheet("font-weight: bold;")
        info_layout.addWidget(self.hf_dataset_label)

        self.hf_url_label = QLabel('<a href="#">View on Hugging Face</a>')
        self.hf_url_label.setOpenExternalLinks(True)
        info_layout.addWidget(self.hf_url_label)

        layout.addWidget(info_group)

        # Status display
        status_group = QGroupBox("Sync Status")
        status_layout = QVBoxLayout(status_group)

        self.sync_status_display = QTextEdit()
        self.sync_status_display.setReadOnly(True)
        self.sync_status_display.setMinimumHeight(200)
        self.sync_status_display.setPlaceholderText("Click 'Check Status' or 'Sync Now' to see sync information...")
        status_layout.addWidget(self.sync_status_display)

        layout.addWidget(status_group)

        # Statistics display
        stats_group = QGroupBox("File Counts")
        stats_layout = QHBoxLayout(stats_group)

        # Local stats
        local_box = QVBoxLayout()
        self.local_count_label = QLabel("Local: --")
        self.local_count_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        local_box.addWidget(self.local_count_label)
        local_box.addWidget(QLabel("Audio files on disk"))
        stats_layout.addLayout(local_box)

        # Arrow
        arrow_label = QLabel("→")
        arrow_label.setStyleSheet("font-size: 24px;")
        arrow_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        stats_layout.addWidget(arrow_label)

        # Remote stats
        remote_box = QVBoxLayout()
        self.remote_count_label = QLabel("Remote: --")
        self.remote_count_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        remote_box.addWidget(self.remote_count_label)
        remote_box.addWidget(QLabel("Files in HF dataset"))
        stats_layout.addLayout(remote_box)

        layout.addWidget(stats_group)

        # Action buttons
        btn_row = QHBoxLayout()

        self.check_status_btn = QPushButton("Check Status")
        self.check_status_btn.clicked.connect(self.check_sync_status)
        btn_row.addWidget(self.check_status_btn)

        self.sync_btn = QPushButton("Sync Now")
        self.sync_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.sync_btn.setMinimumHeight(40)
        self.sync_btn.clicked.connect(self.start_hf_sync)
        btn_row.addWidget(self.sync_btn)

        layout.addLayout(btn_row)

        # Last sync info
        self.last_sync_label = QLabel("Last sync: Unknown")
        self.last_sync_label.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(self.last_sync_label)

        layout.addStretch()

        # Initial status check
        QTimer.singleShot(500, self.check_sync_status)

    def check_sync_status(self):
        """Check current sync status without performing a sync."""
        self.sync_status_display.clear()
        self.sync_status_display.append("Checking status...")

        # Count local files
        audio_dir = self.current_data_dir / "audio"
        local_count = len(list(audio_dir.glob("*.wav"))) if audio_dir.exists() else 0
        self.local_count_label.setText(f"Local: {local_count}")

        # Check git status
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=self.current_data_dir,
                capture_output=True,
                text=True
            )

            status_lines = [l for l in result.stdout.strip().split('\n') if l]
            new_files = len([l for l in status_lines if l.startswith('?') or l.startswith('A')])
            modified_files = len([l for l in status_lines if l.startswith('M')])

            # Get remote count by fetching and checking origin/main
            remote_count = 0
            try:
                # Fetch to update remote refs
                subprocess.run(
                    ["git", "fetch", "origin"],
                    cwd=self.current_data_dir,
                    capture_output=True,
                    timeout=30
                )
                tree_result = subprocess.run(
                    ["git", "ls-tree", "-r", "origin/main", "--name-only", "audio/"],
                    cwd=self.current_data_dir,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                if tree_result.returncode == 0:
                    remote_files = [f for f in tree_result.stdout.strip().split('\n') if f.endswith('.wav')]
                    remote_count = len(remote_files)
                    self.remote_count_label.setText(f"Remote: {remote_count}")
                else:
                    self.remote_count_label.setText("Remote: ?")
            except Exception:
                self.remote_count_label.setText("Remote: ?")

            # Get last commit time
            log_result = subprocess.run(
                ["git", "log", "-1", "--format=%cr"],
                cwd=self.current_data_dir,
                capture_output=True,
                text=True
            )
            if log_result.returncode == 0:
                self.last_sync_label.setText(f"Last commit: {log_result.stdout.strip()}")

            # Display status
            self.sync_status_display.clear()
            if new_files == 0 and modified_files == 0:
                self.sync_status_display.append("✓ All files synced to Hugging Face")
                self.sync_status_display.append(f"\nLocal: {local_count} audio files")
                self.sync_status_display.append(f"Remote: {remote_count} audio files")
            else:
                self.sync_status_display.append(f"⚠ Pending changes to sync:")
                self.sync_status_display.append(f"  • {new_files} new files")
                self.sync_status_display.append(f"  • {modified_files} modified files")
                self.sync_status_display.append(f"\nClick 'Sync Now' to push to Hugging Face")

        except Exception as e:
            self.sync_status_display.append(f"Error checking status: {e}")

    def start_hf_sync(self):
        """Start the Hugging Face sync process."""
        if self.sync_worker and self.sync_worker.isRunning():
            return

        self.sync_btn.setEnabled(False)
        self.check_status_btn.setEnabled(False)
        self.sync_status_display.clear()
        self.sync_status_display.append("Starting sync to Hugging Face...\n")

        self.sync_worker = HFSyncWorker(self.current_data_dir)
        self.sync_worker.progress.connect(self.on_sync_progress)
        self.sync_worker.finished.connect(self.on_sync_finished)
        self.sync_worker.error.connect(self.on_sync_error)
        self.sync_worker.start()

    def on_sync_progress(self, message):
        """Handle sync progress updates."""
        self.sync_status_display.append(f"• {message}")

    def on_sync_finished(self, result):
        """Handle sync completion."""
        self.sync_btn.setEnabled(True)
        self.check_status_btn.setEnabled(True)

        # Update counts
        self.local_count_label.setText(f"Local: {result['local_count']}")
        if result['remote_count'] >= 0:
            self.remote_count_label.setText(f"Remote: {result['remote_count']}")

        # Display result
        self.sync_status_display.append("")
        if result['new_files'] > 0:
            self.sync_status_display.append(f"✓ {result['message']}")
            self.sync_status_display.append(f"\nSummary:")
            self.sync_status_display.append(f"  • Added {result['new_files']} new files")
            if result['modified_files'] > 0:
                self.sync_status_display.append(f"  • Updated {result['modified_files']} files")
            self.sync_status_display.append(f"  • Total in HF: {result['remote_count']} files")
        else:
            self.sync_status_display.append(f"✓ {result['message']}")
            self.sync_status_display.append(f"  • Total in HF: {result['remote_count']} files")

        # Update last sync time
        self.last_sync_label.setText(f"Last sync: just now")

    def on_sync_error(self, error_msg):
        """Handle sync error."""
        self.sync_btn.setEnabled(True)
        self.check_status_btn.setEnabled(True)
        self.sync_status_display.append(f"\n✗ Error: {error_msg}")

    def refresh_samples_table(self):
        """Refresh the samples table with current data."""
        manifest = self.load_manifest()
        samples = manifest.get("samples", [])

        self.samples_table.setRowCount(len(samples))

        for row, sample in enumerate(samples):
            # ID/Number
            num_item = QTableWidgetItem(sample.get("id", str(row + 1)))
            num_item.setData(Qt.ItemDataRole.UserRole, sample.get("id"))
            self.samples_table.setItem(row, 0, num_item)

            # Duration
            duration = sample.get("duration_seconds", 0)
            self.samples_table.setItem(row, 1, QTableWidgetItem(f"{duration:.1f}s"))

            # Style - show content type primarily
            metadata = sample.get("metadata", {})
            if metadata:
                content = metadata.get("content", "unknown")
                style_display = content.replace('_', ' ').title()
            else:
                # Fallback for old format
                style_key = sample.get("style", "")
                style_display = style_key.replace('_', ' ').title()
            self.samples_table.setItem(row, 2, QTableWidgetItem(style_display))

            # Text preview
            text = sample.get("text", "")
            preview = text[:50] + "..." if len(text) > 50 else text
            self.samples_table.setItem(row, 3, QTableWidgetItem(preview))

            # Play button
            play_btn = QPushButton("▶")
            play_btn.setFixedWidth(40)
            audio_file = sample.get("audio_file", "")
            play_btn.clicked.connect(lambda checked, f=audio_file: self.play_sample(f))
            self.samples_table.setCellWidget(row, 4, play_btn)

    def delete_selected_samples(self):
        """Delete selected samples."""
        selected_rows = set(item.row() for item in self.samples_table.selectedItems())

        if not selected_rows:
            QMessageBox.warning(self, "Warning", "No samples selected")
            return

        reply = QMessageBox.question(
            self, "Confirm Delete",
            f"Delete {len(selected_rows)} sample(s)? This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        manifest = self.load_manifest()
        samples = manifest.get("samples", [])

        # Get sample IDs to delete
        ids_to_delete = []
        for row in selected_rows:
            item = self.samples_table.item(row, 0)
            if item:
                sample_id = item.data(Qt.ItemDataRole.UserRole)
                ids_to_delete.append(sample_id)

        # Remove samples and their files
        new_samples = []
        for sample in samples:
            if sample.get("id") in ids_to_delete:
                # Delete files
                audio_file = self.current_audio_dir / sample.get("audio_file", "")
                text_file = self.current_text_dir / sample.get("text_file", "")

                for f in [audio_file, text_file]:
                    if f.exists():
                        f.unlink()
            else:
                new_samples.append(sample)

        manifest["samples"] = new_samples
        manifest["updated"] = datetime.now().isoformat()

        with open(self.current_manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)

        self.refresh_samples_table()
        self.update_statistics()
        self.status_label.setText(f"Deleted {len(ids_to_delete)} sample(s)")

    def populate_audio_devices(self):
        """Populate the microphone dropdown with available audio input devices."""
        self.mic_combo.clear()
        self.audio_devices = QMediaDevices.audioInputs()

        if not self.audio_devices:
            self.mic_combo.addItem("No microphones found")
            return

        default_device = QMediaDevices.defaultAudioInput()
        default_index = 0

        for i, device in enumerate(self.audio_devices):
            name = device.description()
            if device == default_device:
                name += " (Default)"
                default_index = i
            self.mic_combo.addItem(name)

        self.mic_combo.setCurrentIndex(default_index)
        self.on_mic_changed(default_index)

    def on_mic_changed(self, index):
        """Handle microphone selection change."""
        if 0 <= index < len(self.audio_devices):
            self.recorder.set_device(self.audio_devices[index])

    def play_sample(self, audio_filename):
        """Play an audio sample."""
        audio_path = self.current_audio_dir / audio_filename

        if not audio_path.exists():
            QMessageBox.warning(self, "Error", f"Audio file not found: {audio_filename}")
            return

        # Stop any currently playing audio
        self.media_player.stop()

        # Play the audio
        self.media_player.setSource(QUrl.fromLocalFile(str(audio_path)))
        self.media_player.play()

    def load_corpus(self):
        """Load English sentences corpus from file."""
        corpus_path = BASE_DIR / "english-sentences.txt"
        if corpus_path.exists():
            try:
                with open(corpus_path, 'r', encoding='utf-8') as f:
                    self.corpus_sentences = [line.strip() for line in f if line.strip()]
                print(f"Loaded {len(self.corpus_sentences)} sentences from corpus")
            except Exception as e:
                print(f"Error loading corpus: {e}")
        else:
            print(f"Corpus not found at {corpus_path}")

    def load_word_list(self):
        """Load custom words from word-list.txt."""
        word_list_path = BASE_DIR / "word-list.txt"
        if word_list_path.exists():
            try:
                with open(word_list_path, 'r', encoding='utf-8') as f:
                    self.custom_words = [line.strip() for line in f if line.strip()]
                print(f"Loaded {len(self.custom_words)} custom words")
            except Exception as e:
                print(f"Error loading word list: {e}")
        else:
            print(f"Word list not found at {word_list_path}")

    def generate_from_corpus(self):
        """Generate text by randomly selecting from corpus and optionally injecting custom words."""
        import random

        if not self.corpus_sentences:
            return "Corpus not loaded. Please ensure english-sentences.txt exists."

        # Select a random sentence
        sentence = random.choice(self.corpus_sentences)

        # Optionally inject a custom word (50% chance if word list exists)
        if self.custom_words and random.random() < 0.5:
            custom_word = random.choice(self.custom_words)
            words = sentence.split()

            # Try to replace a word naturally (skip first/last, articles, etc)
            replaceable_indices = [
                i for i, w in enumerate(words)
                if i > 0 and i < len(words) - 1
                and w.lower() not in ('the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being')
                and len(w) > 3
            ]

            if replaceable_indices:
                replace_idx = random.choice(replaceable_indices)
                words[replace_idx] = custom_word
                sentence = ' '.join(words)

        # Check if dictionary word should be included
        dict_word = self.word_input.text().strip() if hasattr(self, 'word_input') else ""
        if dict_word and dict_word.lower() not in sentence.lower():
            # Append the dictionary word naturally
            sentence = f"{sentence} {dict_word}."

        return sentence

    def save_and_auto_record(self):
        """Save recording, generate next text, then auto-start recording after countdown."""
        if self.audio_data is None or len(self.audio_data) == 0:
            return

        # Set flag to start auto-record after text generation
        self.pending_auto_record = True

        # Save first
        self.save_recording()
        # save_recording will trigger text generation for LLM mode
        # After text is generated, on_text_generated will check pending_auto_record flag

    def start_auto_record_countdown(self):
        """Start 3-second countdown before auto-recording."""
        current_mode = self.mode_combo.currentData()

        # Only auto-record in LLM mode (manual mode doesn't auto-generate)
        if current_mode != "llm":
            return

        if not self.current_text:
            return

        self.auto_record_countdown = 3
        self.auto_record_timer.start(1000)  # 1 second intervals
        self.status_label.setText(f"🔴 Auto-recording in {self.auto_record_countdown}... (Press Esc to cancel)")

    def auto_record_tick(self):
        """Handle countdown tick for auto-record."""
        self.auto_record_countdown -= 1

        if self.auto_record_countdown > 0:
            self.status_label.setText(f"🔴 Auto-recording in {self.auto_record_countdown}... (Press Esc to cancel)")
        else:
            # Countdown finished - start recording
            self.auto_record_timer.stop()
            self.start_recording()

    def cancel_auto_record(self):
        """Cancel auto-record countdown."""
        if self.auto_record_timer.isActive():
            self.auto_record_timer.stop()
            self.status_label.setText("Auto-record cancelled - Press Space to record manually")
            self.auto_record_countdown = 0

    def confirm_discard_recording(self):
        """Confirm before discarding a recording."""
        if not hasattr(self, 'audio_data') or self.audio_data is None:
            return

        duration = len(self.audio_data) / self.sample_rate if self.audio_data is not None else 0

        if duration > 3:  # Only confirm for recordings longer than 3 seconds
            reply = QMessageBox.question(
                self, "Confirm Discard",
                f"Discard {duration:.1f}s recording? This cannot be undone.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )

            if reply != QMessageBox.StandardButton.Yes:
                return

        self.discard_recording()

    def generate_text(self):
        """Generate new text using cloud API in background thread."""
        # Don't start a new request if one is already running
        if self.text_worker and self.text_worker.isRunning():
            return

        self.status_label.setText("Generating text...")
        self.generate_btn.setEnabled(False)
        self.generation_progress.setVisible(True)

        # Get selections from all three dropdowns
        style = self.style_combo.currentData() or "casual"
        content = self.content_combo.currentData() or "general"
        format_type = self.format_combo.currentData() or "explanation"

        import random
        random_seed = random.randint(10000, 99999)

        # Get dictionary words if provided (up to 3)
        dict_words_text = self.word_input.text().strip() if hasattr(self, 'word_input') else ""
        dict_words = [w.strip() for w in dict_words_text.split(',') if w.strip()][:3]  # Max 3 words

        # Get disambiguation/context if provided
        disambiguation = self.disambig_input.text().strip() if hasattr(self, 'disambig_input') else ""

        dict_word_constraint = ""
        if dict_words:
            if len(dict_words) == 1:
                dict_word_constraint = f"\nCRITICAL: You MUST use the word '{dict_words[0]}' exactly 3 times in the output. Vary the context each time it appears."
            else:
                words_formatted = ', '.join([f"'{w}'" for w in dict_words])
                dict_word_constraint = f"\nCRITICAL: You MUST use each of these words at least 2-3 times: {words_formatted}. Vary the context each time they appear."

        if disambiguation:
            dict_word_constraint += f"\nContext/clarification: {disambiguation}"

        # Build prompt based on selections

        # Style descriptors for prompt
        style_prompts = {
            "casual": "casual, conversational tone with filler words like 'so', 'like', 'you know'",
            "formal": "professional, polished tone without contractions",
            "informal": "very relaxed, chatty tone with lots of informal language",
            "instructional": "clear, step-by-step educational tone"
        }

        # Format descriptors
        format_prompts = {
            "conversation": "one side of a conversation or phone call",
            "email": "written in email style",
            "explanation": "explaining something clearly",
            "note_to_self": "voice memo or note to yourself",
            "story": "telling a short story or anecdote"
        }

        tone_instruction = style_prompts.get(style, style_prompts["casual"])
        format_instruction = format_prompts.get(format_type, format_prompts["explanation"])

        # Handle technical content with specific terminology
        if content == "technical":
            # If dictionary words are provided, build coherent context around them
            if dict_words:
                # Build a coherent technical topic around the dictionary words
                words_list = "', '".join(dict_words)
                context_instruction = f"\nAdditional context: {disambiguation}" if disambiguation else ""

                system_prompt = f"""Generate natural developer speech about technical work.

RULES:
- 40-50 words total
- Use {tone_instruction}
- Format as {format_instruction}
- Create a COHERENT technical narrative that naturally incorporates: '{words_list}'{dict_word_constraint}
- The words should relate to each other in the discussion
- Sound like real developer talk (e.g., "so I was debugging the Docker container and had to check the Kubernetes logs, then updated the Docker config")
- NOT like documentation or a list
- The entire statement must make technical sense as one cohesive thought{context_instruction}

Seed: {random_seed}

Output speech only."""

                user_prompt = f"Create a natural, coherent technical discussion incorporating: {words_list}"

            else:
                # No dictionary words - use improved tech clusters with better coherence
                tech_scenarios = [
                    {
                        "context": "containerization deployment",
                        "terms": ["Docker", "Kubernetes", "deployment"],
                        "scenario": "deploying containerized applications"
                    },
                    {
                        "context": "API development",
                        "terms": ["API", "REST", "authentication", "JWT"],
                        "scenario": "building and securing APIs"
                    },
                    {
                        "context": "frontend development",
                        "terms": ["React", "TypeScript", "component", "hooks"],
                        "scenario": "developing React components"
                    },
                    {
                        "context": "version control",
                        "terms": ["Git", "GitHub", "branch", "merge"],
                        "scenario": "managing code with version control"
                    },
                    {
                        "context": "machine learning",
                        "terms": ["PyTorch", "model", "training", "dataset"],
                        "scenario": "training ML models"
                    },
                    {
                        "context": "database work",
                        "terms": ["database", "PostgreSQL", "query", "schema"],
                        "scenario": "working with databases"
                    },
                    {
                        "context": "server administration",
                        "terms": ["Linux", "SSH", "server", "systemd"],
                        "scenario": "managing Linux servers"
                    },
                    {
                        "context": "CI/CD pipeline",
                        "terms": ["CI/CD", "GitHub Actions", "testing", "deployment"],
                        "scenario": "setting up automated pipelines"
                    }
                ]

                scenario = random.choice(tech_scenarios)
                selected_terms = random.sample(scenario["terms"], min(2, len(scenario["terms"])))

                system_prompt = f"""Generate natural developer speech about technical work.

RULES:
- 40-50 words total
- Use {tone_instruction}
- Format as {format_instruction}
- Context: {scenario["scenario"]}
- Naturally incorporate: {', '.join(selected_terms)}
- Create a coherent narrative, not just a list
- Sound like real developer talk with natural flow
- Example good output: "so I spent the morning setting up the Docker container, had some issues with the Kubernetes config but got it deployed eventually"

Seed: {random_seed}

Output speech only."""

                user_prompt = f"Tech scenario: {scenario['scenario']}"

        else:
            # Other content types - topic based on content selection
            content_topics = {
                "general": ["daily routines", "weekend plans", "interesting observations", "recent experiences", "practical tips"],
                "personal": ["personal reflections", "journal entries", "thoughts and feelings", "life updates", "self-notes"],
                "business": ["meeting summaries", "project updates", "client communications", "work priorities", "professional goals"],
                "creative": ["story ideas", "creative concepts", "artistic inspirations", "imaginative scenarios", "narrative fragments"],
                "educational": ["learning notes", "study summaries", "concept explanations", "teaching points", "educational insights"],
                "communication": ["message drafts", "quick updates", "correspondence notes", "conversation points", "follow-up reminders"],
                "documentation": ["process notes", "how-to steps", "reference information", "setup instructions", "usage guidelines"]
            }

            topics = content_topics.get(content, content_topics["general"])
            topic = random.choice(topics)

            system_prompt = f"""Generate natural spoken language about everyday topics.

RULES:
- 40-50 words total
- Topic area: {topic}
- Use {tone_instruction}
- Format as {format_instruction}
- Sound like real, natural speech with conversational flow{dict_word_constraint}
- Be specific with details

Seed: {random_seed}

Output speech only."""

            user_prompt = f"Topic: {topic}"

        # Run in background thread with cloud API
        api_key = self.get_api_key()
        if not api_key:
            self.status_label.setText("Error: No API key configured. Please add API key in Settings.")
            self.generate_btn.setEnabled(True)
            return

        self.text_worker = TextGeneratorWorker(api_key, system_prompt, user_prompt)
        self.text_worker.finished.connect(self.on_text_generated)
        self.text_worker.error.connect(self.on_text_error)
        self.text_worker.start()

    def on_text_generated(self, text):
        """Handle successful text generation."""
        self.current_text = text
        self.text_display.setText(text)
        self.generate_btn.setEnabled(True)
        self.generation_progress.setVisible(False)

        # Check if we should start auto-record countdown
        if self.pending_auto_record:
            self.pending_auto_record = False
            self.start_auto_record_countdown()
        else:
            self.status_label.setText("Text generated - ready to record (Press Space to start)")

    def on_text_error(self, error_msg):
        """Handle text generation error."""
        self.status_label.setText(f"Error: {error_msg}")
        self.generate_btn.setEnabled(True)
        self.generation_progress.setVisible(False)

    def toggle_recording(self):
        """Start or stop recording."""
        if self.recorder.is_recording:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self):
        """Start audio recording."""
        if not self.current_text:
            QMessageBox.warning(self, "Warning", "Please generate text first")
            return

        try:
            self.recorder.start_recording()
            self.record_btn.setText("⏹️ Stop Recording (Space)")
            self.record_btn.setStyleSheet("background-color: #ff6b6b; font-weight: bold; animation: pulse 1.5s infinite;")
            self.recording_seconds = 0
            self.recording_timer.start(1000)
            self.save_btn.setEnabled(False)
            self.save_and_next_btn.setEnabled(False)
            self.discard_btn.setEnabled(False)
            self.status_label.setText("🔴 Recording in progress...")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start recording: {e}")

    def stop_recording(self):
        """Stop audio recording."""
        self.audio_data, self.sample_rate = self.recorder.stop_recording()
        self.recording_timer.stop()
        self.record_btn.setText("🎙️ Start Recording (Space)")
        self.record_btn.setStyleSheet("")
        self.save_btn.setEnabled(True)
        self.save_and_next_btn.setEnabled(True)
        self.discard_btn.setEnabled(True)

        duration = len(self.audio_data) / self.sample_rate if self.audio_data is not None else 0
        self.status_label.setText(f"✓ Recorded {duration:.1f}s - Save (Ctrl+S) or Save & Next (Ctrl+Enter)")

    def update_timer(self):
        """Update recording timer display."""
        self.recording_seconds += 1
        mins = self.recording_seconds // 60
        secs = self.recording_seconds % 60
        self.time_label.setText(f"{mins}:{secs:02d}")

        if self.recording_seconds > 30:
            self.time_label.setStyleSheet("font-size: 24px; font-weight: bold; color: red;")
        else:
            self.time_label.setStyleSheet("font-size: 24px; font-weight: bold;")

    def save_recording(self):
        """Save the current recording."""
        if self.audio_data is None or len(self.audio_data) == 0:
            QMessageBox.warning(self, "Warning", "No audio to save")
            return

        duration = len(self.audio_data) / self.sample_rate

        if duration < 1:
            QMessageBox.warning(self, "Warning", "Recording too short (< 1 second)")
            return
        if duration > 35:
            QMessageBox.warning(self, "Warning", f"Recording too long ({duration:.1f}s > 35s)")
            return

        try:
            # Trim 100ms from start and end to remove mouse clicks
            trim_samples = int(0.1 * self.sample_rate)  # 100ms
            if len(self.audio_data) > trim_samples * 2:
                trimmed_audio = self.audio_data[trim_samples:-trim_samples]
            else:
                trimmed_audio = self.audio_data

            # Generate filenames with timestamp and UUID
            sample_id = str(uuid.uuid4())[:8]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            audio_filename = f"{timestamp}_{sample_id}.wav"
            text_filename = f"{timestamp}_{sample_id}.txt"

            # Ensure directories exist
            self.current_audio_dir.mkdir(parents=True, exist_ok=True)
            self.current_text_dir.mkdir(parents=True, exist_ok=True)

            # Save audio
            audio_path = self.current_audio_dir / audio_filename
            with wave.open(str(audio_path), 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.sample_rate)
                wf.writeframes(trimmed_audio.tobytes())

            # Save text
            text_path = self.current_text_dir / text_filename
            with open(text_path, 'w') as f:
                f.write(self.current_text)

            # Update manifest with trimmed duration
            trimmed_duration = len(trimmed_audio) / self.sample_rate
            manifest = self.load_manifest()

            # Determine if this was manual or LLM generated
            current_mode = self.mode_combo.currentData()

            # Build style metadata
            if current_mode == "manual":
                style_str = "manual"
                style_metadata = {"source": "manual"}
            else:
                # Combine the three dimensions for LLM mode
                style_val = self.style_combo.currentData() or "casual"
                content_val = self.content_combo.currentData() or "general"
                format_val = self.format_combo.currentData() or "explanation"
                style_str = f"{style_val}_{content_val}_{format_val}"
                style_metadata = {
                    "source": "llm",
                    "style": style_val,
                    "content": content_val,
                    "format": format_val
                }

            sample_entry = {
                "id": sample_id,
                "audio_file": audio_filename,
                "text_file": text_filename,
                "text": self.current_text,
                "duration_seconds": round(trimmed_duration, 2),
                "sample_rate": self.sample_rate,
                "style": style_str,
                "metadata": style_metadata,
                "timestamp": datetime.now().isoformat()
            }
            manifest["samples"].append(sample_entry)
            manifest["updated"] = datetime.now().isoformat()

            with open(self.current_manifest_file, 'w') as f:
                json.dump(manifest, f, indent=2)

            # Update session statistics
            self.session_samples += 1

            self.status_label.setText(f"✓ Saved sample {sample_id} ({trimmed_duration:.1f}s)")
            self.update_statistics()

            # Reset for next recording
            self.discard_recording()

            # Only auto-generate in LLM mode
            if current_mode == "llm":
                self.generate_text()
            else:
                # In manual mode, clear the input and display, ready for next entry
                self.manual_text_input.clear()
                self.text_display.clear()
                self.current_text = ""
                self.status_label.setText("Enter next text to record")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save recording: {str(e)}")
            self.status_label.setText(f"❌ Save failed: {str(e)}")
            print(f"Save error: {e}")  # Debug output

    def discard_recording(self):
        """Discard the current recording."""
        self.audio_data = None
        self.save_btn.setEnabled(False)
        self.save_and_next_btn.setEnabled(False)
        self.discard_btn.setEnabled(False)
        self.time_label.setText("0:00")
        self.time_label.setStyleSheet("font-size: 24px; font-weight: bold;")
        self.status_label.setText("Recording discarded")

    def load_manifest(self):
        """Load the dataset manifest."""
        if self.current_manifest_file and self.current_manifest_file.exists():
            with open(self.current_manifest_file, 'r') as f:
                return json.load(f)
        return {"samples": [], "created": datetime.now().isoformat(), "version": "1.0"}

    def update_statistics(self):
        """Update the statistics display."""
        manifest = self.load_manifest()
        samples = manifest.get("samples", [])

        total = len(samples)
        duration = sum(s.get("duration_seconds", 0) for s in samples)
        hours = int(duration // 3600)
        mins = int((duration % 3600) // 60)
        secs = int(duration % 60)

        # Update session stats
        session_duration = (datetime.now() - self.session_start_time).total_seconds()
        session_mins = int(session_duration // 60)
        session_secs = int(session_duration % 60)

        session_stats = f"""📊 Samples: {self.session_samples}
⏱️ Session time: {session_mins}m {session_secs}s"""

        if self.session_samples > 0:
            avg_per_sample = session_duration / self.session_samples
            session_stats += f"\n⚡ Avg: {avg_per_sample:.0f}s per sample"

        self.session_stats_label.setText(session_stats)

        # Overall statistics
        source_counts = {"llm": 0, "manual": 0}
        content_counts = {}
        style_tone_counts = {}
        format_counts = {}

        for s in samples:
            # Get metadata if available (new format) or fall back to old format
            metadata = s.get("metadata", {})
            source = metadata.get("source", "llm" if s.get("style", "") != "manual" else "manual")
            source_counts[source] = source_counts.get(source, 0) + 1

            # Count by content, style tone, and format (only for LLM-generated)
            if source == "llm":
                content = metadata.get("content", "unknown")
                content_counts[content] = content_counts.get(content, 0) + 1

                style_tone = metadata.get("style", "unknown")
                style_tone_counts[style_tone] = style_tone_counts.get(style_tone, 0) + 1

                format_type = metadata.get("format", "unknown")
                format_counts[format_type] = format_counts.get(format_type, 0) + 1

        stats = f"""<b>Total Samples:</b> {total}
<b>Total Time:</b> {hours}h {mins}m {secs}s

<b>By Source:</b>
• 🤖 LLM Generated: {source_counts['llm']}
• ✍️ Manual Entry: {source_counts['manual']}
"""

        if content_counts:
            stats += "\n<b>Content Types:</b>\n"
            for content, count in sorted(content_counts.items()):
                stats += f"• {content.replace('_', ' ').title()}: {count}\n"

        if total == 0:
            stats += "\n<i>No samples yet</i>"

        self.stats_label.setText(stats)

    def show_shortcuts_help(self):
        """Show keyboard shortcuts help dialog."""
        help_text = """
<h2>Keyboard Shortcuts</h2>

<h3>Recording Workflow</h3>
<table cellpadding="5">
<tr><td><b>Space</b></td><td>Start/Stop recording</td></tr>
<tr><td><b>Ctrl+S</b></td><td>Save recording (then generate next)</td></tr>
<tr><td><b>Ctrl+Enter</b></td><td>Save → Generate → Auto-record after 3s (fastest!)</td></tr>
<tr><td><b>Escape</b></td><td>Cancel auto-record countdown</td></tr>
<tr><td><b>Ctrl+D</b></td><td>Discard recording</td></tr>
<tr><td><b>Ctrl+N</b> or <b>Ctrl+G</b></td><td>Generate new text / Skip (LLM mode)</td></tr>
</table>

<h3>Recording Modes</h3>
<ul>
<li><b>🤖 LLM Generated:</b> AI generates varied text prompts automatically</li>
<li><b>✍️ Manual Entry:</b> Type or paste your own text to record</li>
</ul>

<h3>Tips</h3>
<ul>
<li><b>⚡ Fastest Workflow:</b> Use <b>Ctrl+Enter</b> - saves, generates next, and auto-starts recording after 3 second countdown. Just keep reading!</li>
<li>Press <b>Escape</b> during countdown to cancel auto-record</li>
<li>Press <b>Space</b> to quickly start/stop without reaching for mouse</li>
<li>In LLM mode, dictionary words are automatically included in generated text</li>
<li>In Manual mode, you can paste lists of sentences to record in sequence</li>
<li>Recordings longer than 3 seconds will ask for confirmation before discarding</li>
</ul>
        """

        msg = QMessageBox(self)
        msg.setWindowTitle("Keyboard Shortcuts")
        msg.setTextFormat(Qt.TextFormat.RichText)
        msg.setText(help_text)
        msg.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg.exec()

    def populate_dataset_combo(self):
        """Populate the dataset combo box with available profiles."""
        self.dataset_combo.clear()
        active_profile = self.profiles_data.get("active_profile", "")

        for profile in self.profiles_data.get("profiles", []):
            name = profile.get("name", "Unnamed")
            self.dataset_combo.addItem(name, profile)

            if name == active_profile:
                self.dataset_combo.setCurrentIndex(self.dataset_combo.count() - 1)

    def load_active_profile(self):
        """Load and apply the active dataset profile."""
        active_profile_name = self.profiles_data.get("active_profile", "")

        # Find the active profile
        active_profile = None
        for profile in self.profiles_data.get("profiles", []):
            if profile.get("name") == active_profile_name:
                active_profile = profile
                break

        if not active_profile and self.profiles_data.get("profiles"):
            # Fall back to first profile
            active_profile = self.profiles_data["profiles"][0]

        if active_profile:
            self.switch_to_profile(active_profile)
        else:
            # No profiles - use default DATA_DIR
            self.current_data_dir = DATA_DIR
            self.update_directory_paths()

    def switch_to_profile(self, profile):
        """Switch to a different dataset profile."""
        profile_path = Path(profile.get("path", ""))

        if not profile_path.exists():
            QMessageBox.warning(
                self,
                "Profile Path Not Found",
                f"The dataset path for '{profile.get('name')}' does not exist:\n{profile_path}\n\n"
                "Please update the profile or create the directory."
            )
            return

        # Update current directory paths
        self.current_data_dir = profile_path
        self.update_directory_paths()

        # Ensure subdirectories exist
        for dir_path in [self.current_audio_dir, self.current_text_dir, self.current_metadata_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Load categories from profile
        self.load_profile_categories(profile)

        # Update UI
        self.update_statistics()
        self.refresh_samples_table()

        # Update sync tab display
        if hasattr(self, 'hf_dataset_label'):
            remote_url = profile.get("remote_url", "Not configured")
            if remote_url:
                self.hf_dataset_label.setText(f"Dataset: {profile.get('name')}")
                self.hf_url_label.setText(f'Remote: {remote_url}')
            else:
                self.hf_dataset_label.setText(f"Dataset: {profile.get('name')} (No remote)")
                self.hf_url_label.setText("No remote URL configured")

        self.status_label.setText(f"✓ Switched to dataset: {profile.get('name')}")

    def load_profile_categories(self, profile):
        """Load categories from profile and update UI combo boxes."""
        # Get categories from profile or use defaults
        categories = profile.get("categories", get_default_categories())

        # Update current categories
        self.current_styles = categories.get("styles", get_default_categories()["styles"])
        self.current_content_types = categories.get("content_types", get_default_categories()["content_types"])
        self.current_formats = categories.get("formats", get_default_categories()["formats"])

        # Update combo boxes
        if hasattr(self, 'style_combo'):
            current_style = self.style_combo.currentData()
            self.style_combo.clear()
            for key in sorted(self.current_styles.keys()):
                self.style_combo.addItem(key.replace('_', ' ').title(), key)
            # Try to restore previous selection
            index = self.style_combo.findData(current_style)
            if index >= 0:
                self.style_combo.setCurrentIndex(index)

        if hasattr(self, 'content_combo'):
            current_content = self.content_combo.currentData()
            self.content_combo.clear()
            for key in sorted(self.current_content_types.keys()):
                self.content_combo.addItem(key.replace('_', ' ').title(), key)
            # Try to restore previous selection
            index = self.content_combo.findData(current_content)
            if index >= 0:
                self.content_combo.setCurrentIndex(index)

        if hasattr(self, 'format_combo'):
            current_format = self.format_combo.currentData()
            self.format_combo.clear()
            for key in sorted(self.current_formats.keys()):
                self.format_combo.addItem(key.replace('_', ' ').title(), key)
            # Try to restore previous selection
            index = self.format_combo.findData(current_format)
            if index >= 0:
                self.format_combo.setCurrentIndex(index)

    def update_directory_paths(self):
        """Update all directory paths based on current_data_dir."""
        self.current_audio_dir = self.current_data_dir / "audio"
        self.current_text_dir = self.current_data_dir / "text"
        self.current_metadata_dir = self.current_data_dir / "metadata"
        self.current_manifest_file = self.current_data_dir / "manifest.json"

    def on_dataset_changed(self, index):
        """Handle dataset selection change."""
        if index < 0:
            return

        profile = self.dataset_combo.itemData(index)
        if profile:
            # Update active profile in config
            self.profiles_data["active_profile"] = profile.get("name", "")
            save_dataset_profiles(self.profiles_data)

            # Switch to the selected profile
            self.switch_to_profile(profile)

    def show_dataset_manager(self):
        """Show the dataset profile manager dialog."""
        dialog = DatasetProfileManager(self)
        dialog.exec()
        # Always reload profiles after the dialog closes (covers closing with the window controls)
        self.profiles_data = load_dataset_profiles()
        self.populate_dataset_combo()
        self.load_active_profile()

    def show_settings(self):
        """Show the settings dialog."""
        dialog = SettingsDialog(self)
        dialog.exec()

    def get_api_key(self):
        """Get API key from QSettings or environment."""
        settings = QSettings("WhisperFinetuning", "DataCollector")
        raw_key = settings.value("openrouter_api_key", "")
        api_key = sanitize_api_key(raw_key)
        if api_key:
            return api_key

        if raw_key:
            settings.remove("openrouter_api_key")

        env_key = sanitize_api_key(os.environ.get("OPENROUTER_API_KEY", ""))
        if env_key:
            return env_key

        return DEFAULT_API_KEY


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
