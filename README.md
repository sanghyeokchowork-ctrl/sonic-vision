# 🎵 Sonic Vision Pro: AI Music Workstation

> **"From Analysis to Creation: An AI Engineer for Musicians"**
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
* **Evolution:** Started as a simple genre classifier, this project evolved into **Sonic Vision Pro**—a comprehensive tool that not only "listens" but also "assists" in the creative process.
* **Goal:** To bridge the gap between **Music Production** and **AI Engineering** by building tools that solve real-world musician problems:
    * *"Is my mix balanced for Hip-hop?"*
    * *"What is the texture of my vocal?"*
    * *"Which songs are acoustically similar to mine?"*

## 2. Key Features & Architecture

### 🧠 1. Deep Audio Analysis (Vision & Metric Learning)
* **Genre Classification:** Fine-tuned **ResNet18** on Mel-Spectrograms (Accuracy: **92.09%**).
* **Explainable AI (XAI):** Visualized model attention using **Grad-CAM** to reveal *why* a specific genre was predicted (e.g., focusing on Bass vs. Vocals).
* **Acoustic Similarity (Siamese Network):** Implemented a **Siamese Network with Triplet Loss** to learn a metric space where acoustically similar songs are clustered together, surpassing simple class-based matching.

### 🎚️ 2. AI Mixing Assistant
* **Algorithm:** A hybrid Signal Processing engine (FFT-based) that analyzes the frequency balance of a track.
* **Genre-Adaptive:** Automatically compares the track's spectrum against "Ideal Genre Targets" (e.g., Hip-hop requires boosted Sub-bass, Pop requires present Mids).
* **Actionable Feedback:** Provides specific engineering advice (e.g., *"Cut Low-Mids -2dB"*, *"Boost Brilliance +1.5dB"*).

### 🎤 3. Vocal Lab (Timbre Forensics)
* **Pipeline:** Automates the extraction of **Vocals Only** using Demucs.
* **Model:** A custom **Multi-label CNN** trained on MFCC features to classify abstract vocal textures.
* **Tags:** Detects nuances like `Warm`, `Bright`, `Breathy`, `Rough`, and `Clean`.

### 🎛️ 4. Remix Station
* **Source Separation:** Integrated **Meta's Demucs (htdemucs)** model to separate tracks into 4 stems (Vocals, Drums, Bass, Other) with SOTA quality.

## 3. Tech Stack
| Category | Technology | Description |
|---|---|---|
| **Core** | Python 3.9 | Main programming language |
| **Model** | **PyTorch** | CNN (ResNet), Siamese Network, Custom Timbre CNN |
| **Audio DSP** | **Librosa**, Torchaudio | STFT, MFCC, Spectrogram conversion, Frequency Analysis |
| **Separation** | **Demucs (v4)** | SOTA Music Source Separation (Hybrid Transformer) |
| **Deployment** | **Streamlit** | Interactive Dashboard (4-Tab Layout) |
| **Data** | Pandas | Handling dataset annotations and labeling |
| **Hardware** | macOS (MPS) | Optimized for Apple Silicon GPU acceleration |

## 4. Analysis Case Study
I analyzed my own tracks to test the model's adaptability.

### Case 1: JAZZ CLUB (feat. AstralCube) (Melodic Rap) → Predicted: POP
<p align="center">
  <img src="results/heatmap_9624 JAZZ CLUB AR.wav.jpg" width="400" alt="Heatmap Jazz Club">
</p>

* **Insight:** The AI focused on the **low-end & kick**, correctly identifying the modern mixing balance typical of Pop/Hip-Hop, ignoring the misleading title "Jazz".

### Case 2: Lawson. (Trapsoul) → Predicted: HIPHOP
<p align="center">
  <img src="results/heatmap_9624 Lawson. AR.wav.jpg" width="400" alt="Heatmap Lawson">
</p>

* **Insight:** The red zones are dense in the **Sub-bass** region, proving the model detects 808 patterns effectively.

### Case 3: Serenade, Pt. II (R&B) → Predicted: COUNTRY
<p align="center">
  <img src="results/heatmap_9624 SERENADE PT. II AR.wav.jpg" width="400" alt="Heatmap Serenade">
</p>

* **Insight:** The model focused on **Vocals & Acoustics**. Since 'R&B' was not in the training set, it mapped the "Acoustic/Vocal-centric" texture to the closest learned representation: Country.

## 5. Installation & Usage

### Prerequisites
- Python 3.9+
- FFmpeg

### Setup
```bash
# 1. Clone the repository
git clone [https://github.com/sanghyeokchowork-ctrl/sonic-vision.git](https://github.com/sanghyeokchowork-ctrl/sonic-vision.git)
cd sonic-vision

# 2. Install dependencies
pip install -r requirements.txt

Running the App
Bash

streamlit run src/app.py
Tab 1 (Analysis): Check Genre & Heatmaps.

Tab 2 (Remix): Separate stems using Demucs.

Tab 3 (Mixing): Get AI EQ suggestions.

Tab 4 (Vocal Lab): Transcribe lyrics & Analyze vocal timbre.

Tab 5 (Discovery): Find acoustically similar songs.

Training Modules (Optional)
If you want to train your own Vocal Timbre model:

# 1. Auto-extract vocals from your songs
python src/prepare_vocals.py

# 2. Generate CSV & Train
python src/vocal_timbre_train.py

## 6. Engineering Challenges & Solutions
1. Data Scarcity for "Vocal Timbre"
Issue: There are no public datasets labeled with abstract tags like "Breathy" or "Warm".

Solution: Built an Auto-Labelling Pipeline.

Script iterates through my discography -> Uses Demucs to extract vocals -> Saves to a training folder.

Created a custom PyTorch Dataset class that handles CSV encoding issues (CP949/UTF-8) robustly.

2. Subjective Similarity
Issue: Determining if two songs are "similar" is subjective and hard to define with classification accuracy.

Solution: Switched from Classification to Metric Learning. Implemented a Siamese Network with Triplet Loss, teaching the AI to understand "distance" between tracks rather than just labels.

3. OOM (Out of Memory) on Mac
Issue: Running Demucs + ResNet + Grad-CAM simultaneously crashed the GPU (MPS).

Solution: Implemented Lazy Loading (@st.cache_resource) and strategic CPU offloading for heavy separation tasks, ensuring a smooth UX on local machines.