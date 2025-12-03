# 🎵 Sonic Vision Pro: AI Music Workstation

> **"The Ultimate AI Engineer for Musicians"**
> An end-to-end AI workstation exploring Audio Analysis, Intelligent Mixing, and Vocal Forensics using PyTorch & Librosa.

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.9-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c?logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?logo=streamlit" alt="Streamlit">
  <img src="https://img.shields.io/badge/Demucs-Source%20Separation-blueviolet" alt="Demucs">
  <img src="https://img.shields.io/badge/Librosa-DSP-green" alt="Librosa">
</div>

## 1. Project Overview
* **Identity:** Developed by **PARFUMDEWALKER** (R&B Artist & Dev).
* **Evolution:** Evolved from a simple classifier into **Sonic Vision Pro**—a comprehensive dashboard that visualizes the "DNA" of a track using Deep Learning and DSP.
* **Goal:** To bridge the gap between **Music Production** and **AI Engineering** by building tools that solve real-world musician problems.

## 2. Key Features (v2.0 Updates)

### 📊 1. Deep Audio Analysis (Dashboard UI)
* **Genre Classification:** Fine-tuned **ResNet18** on Mel-Spectrograms (Accuracy: **92.09%**).
* **Deep Feature Regression:** Unlike simple rule-based tools, this uses a **Custom Regression Model** to predict abstract musical features:
    * ⚡ **Energy** | 💃 **Danceability** | 🎻 **Acousticness** | 🌞 **Valence (Mood)**
* **Visual AI:** Uses **Grad-CAM** to generate attention heatmaps, visualizing exactly *what* the AI is listening to (e.g., Kick drum vs. Vocals).

### 🎤 2. Vocal Lab (Analysis & Simulation)
* **Timbre Forensics:** A Multi-label CNN analyzes vocal textures (`Warm`, `Bright`, `Breathy`, etc.).
* **Real-time DSP Simulation:** **(New)** Hear your vocals processed in different styles instantly.
    * Uses **44.1kHz High-Quality DSP** logic.
    * Features strictly require separated vocal stems to ensure quality.

### 🎚️ 3. AI Mixing Assistant
* **Genre-Adaptive:** Analyzes the frequency spectrum and compares it against "Ideal Genre Targets" (e.g., Hip-hop vs. Country).
* **Actionable Feedback:** Provides specific EQ advice (e.g., *"Cut Low-Mids -2dB"*, *"Boost Brilliance +1.5dB"*).

### 🔭 4. Discovery Engine
* **Metric Learning:** Implemented a **Siamese Network with Triplet Loss**.
* **Similarity Search:** Finds songs with acoustically similar vibes/grooves rather than just matching genre labels.

### 🎛️ 5. Remix Station
* **Source Separation:** Integrated **Meta's Demucs (htdemucs)** to separate tracks into 4 stems (Vocals, Drums, Bass, Other) with SOTA quality.

## 3. UI/UX Design Philosophy
* **"Black & Bold":** A fully custom Dark Mode interface designed for studio environments.
* **Dashboard Grid:** Key metrics are displayed in a unified grid layout for at-a-glance analysis.
* **Visual Feedback:** All charts and metric cards use the **Spotify Green (`#1DB954`)** accent color for high visibility and consistent branding.

## 4. Tech Stack
| Category | Technology | Description |
|---|---|---|
| **Core** | Python 3.9 | Main programming language |
| **Model** | **PyTorch** | ResNet18 (Cls/Reg), Siamese Network, Custom CNN |
| **Audio DSP** | **Librosa**, Torchaudio | STFT, MFCC, EQ/Filter processing (44.1kHz) |
| **Separation** | **Demucs (v4)** | SOTA Music Source Separation |
| **Deployment** | **Streamlit** | Interactive Web App with Custom CSS |
| **Hardware** | macOS (MPS) | Optimized for Apple Silicon GPU acceleration |

## 5. Installation & Usage

### Prerequisites
- Python 3.9+
- FFmpeg (Required for audio processing)

### Setup
```bash
# 1. Clone the repository
git clone [https://github.com/sanghyeokchowork-ctrl/sonic-vision.git](https://github.com/sanghyeokchowork-ctrl/sonic-vision.git)
cd sonic-vision

# 2. Install dependencies
pip install -r requirements.txt