title: {{Sonic Vision Pro}} 
emoji: {{🎹}} 
colorFrom: {{blue}} 
colorTo: {{purple}} 
sdk: {{streamlit}} 
sdk_version: "{{1.31.0}}" 
app_file: app.py 
pinned: false
---

# 🎵 Sonic Vision: AI Music Analyzer & Deconstructor

> **"How does AI listen to music compared to an artist?"**
> An end-to-end AI project exploring Audio Analysis, Explainable AI (XAI), and Source Separation using PyTorch.

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.9-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c?logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?logo=streamlit" alt="Streamlit">
  <img src="https://img.shields.io/badge/Demucs-Source%20Separation-blueviolet" alt="Demucs">
</div>

## 1. Project Overview
* **Motivation:** As an R&B artist ('PARFUMDEWALKER'), I noticed that standard open-source datasets (like GTZAN) lack modern 'R&B' or 'Melodic Rap' categories. I wanted to investigate **how AI interprets these "unseen" genres** based on limited training data.
* **Goal:** To build a full-stack AI application that classifies genres, visualizes decision-making processes (Heatmaps), recommends similar tracks, and **deconstructs audio into stems (Vocals/Drums/Bass)**.

## 2. Key Features
1.  **Genre Classification:** Fine-tuned **ResNet18** (CNN) on Mel-Spectrograms (Accuracy: **92.09%**).
2.  **Explainable AI (XAI):** Visualized model attention using **Grad-CAM** to reveal *why* a specific genre was predicted (e.g., focusing on Bass vs. Vocals).
3.  **Recommendation Engine:** Implemented a **Vector Search** system using Cosine Similarity on latent feature vectors (512-dim) extracted from the penultimate layer.
4.  **Remix Station (Source Separation):** Integrated **Meta's Demucs (htdemucs)** model to separate tracks into 4 stems (Vocals, Drums, Bass, Other) with SOTA quality.

## 3. Tech Stack
| Category | Technology | Description |
|---|---|---|
| **Core** | Python 3.9 | Main programming language |
| **Model** | **PyTorch**, Torchvision | CNN Architecture & Transfer Learning |
| **Audio** | **Librosa**, Torchaudio | DSP (Spectrogram conversion) & I/O |
| **Separation** | **Demucs (v4)** | **SOTA Music Source Separation** (Hybrid Transformer) |
| **XAI** | **Grad-CAM** | Visualizing model focus areas (Heatmaps) |
| **Deployment** | **Streamlit** | Interactive Web UI implementation |
| **Env** | macOS (MPS/CPU) | Hardware acceleration & Memory optimization |

## 4. Analysis Case Study
I analyzed my own tracks to test the model's adaptability.

### Case 1: JAZZ CLUB (feat. AstralCube) (Melodic Rap) → Predicted: POP
<p align="center">
  <img src="results/heatmap_9624 JAZZ CLUB AR.wav.jpg" width="400" alt="Heatmap Jazz Club">
</p>

* **Context:** The title is inspired by *Maison Margiela's 'Replica Jazz Club'* fragrance. The track is actually **Melodic Rap**.
* **AI Focus:** The heatmap highlights the **bottom area (Low Frequencies/Bass)**.
* **Insight:** The AI was not misled by the text title "Jazz". Instead, it correctly identified the **modern mixing balance (strong low-end & kick)** characteristic of Pop/Hip-Hop, proving it analyzes actual audio textures rather than metadata.

### Case 2: Lawson. (Hip-Hop) → Predicted: HIPHOP
<p align="center">
  <img src="results/heatmap_9624 Lawson. AR.wav.jpg" width="400" alt="Heatmap Lawson">
</p>

* **AI Focus:** The red zones are concentrated in **dense blocks at the bottom (Sub-bass)**.
* **Insight:** The model successfully identified the **808 sub-bass and rhythmic kick patterns**, which are signature elements of Hip-Hop.

### Case 3: Serenade, Pt. II (R&B) → Predicted: COUNTRY
<p align="center">
  <img src="results/heatmap_9624 SERENADE PT. II AR.wav.jpg" width="400" alt="Heatmap Serenade">
</p>

* **AI Focus:** The heatmap is spread across the **Mid-to-High frequencies (Vocals & Acoustics)**.
* **Insight:** Instead of the beat, the model focused on the **vocal melody and acoustic textures**. Since 'R&B' was not a label option, the AI mapped the **vocal-centric nature** of the track to 'COUNTRY', which shares similar acoustic characteristics in the training set.

## 5. Installation & Usage

### Prerequisites
- Python 3.9+
- FFmpeg (for audio processing)

### Setup
```bash
# 1. Clone the repository
git clone [https://github.com/sanghyeokchowork-ctrl/sonic-vision.git](https://github.com/sanghyeokchowork-ctrl/sonic-vision.git)
cd sonic-vision

# 2. Install dependencies
pip install -r requirements.txt

Run Web App
Bash

# Start the Streamlit application
streamlit run src/app.py
The app allows you to upload MP3/WAV files, view analysis results, and perform source separation.

6. Engineering Challenges & Solutions
OOM (Out of Memory) on Mac:

Issue: High-quality source separation (Demucs) caused memory overflow on macOS GPU (MPS).

Solution: Implemented a hybrid inference pipeline where lightweight classification runs on MPS (GPU) for speed, while heavy separation tasks are offloaded to CPU to ensure stability and prevent crashing.

Dataset Bias:

Issue: GTZAN dataset is outdated (1950s-90s style).

Solution: Used Feature Vector Similarity search to bridge the gap between vintage training data and modern production styles.

Developed by PARFUMDEWALKER