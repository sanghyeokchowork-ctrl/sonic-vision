# 🎵 Sonic Vision: Explainable AI for Music Analysis
> **"How does AI hear music compared to an artist?"**
> An AI project exploring the gap between human auditory perception and machine vision using PyTorch & Grad-CAM.

## 1. Project Overview
* **Objective:** To build a Music Genre Classification model using CNN (ResNet18) and visualize the model's decision-making process using Explainable AI (XAI).
* **Motivation:** As an R&B artist, I noticed that most open-source datasets (like GTZAN) lack an 'R&B' category. I was curious: **"How would an AI interpret my R&B tracks based on limited training data?"**
* **Key Result:** Achieved **92.09% accuracy** on the test set and successfully visualized the model's focus points (Bass vs. Vocals) using Grad-CAM heatmaps.

## 2. Tech Stack
| Category | Technology | Purpose |
|---|---|---|
| **Language** | Python 3.9 | Core Logic |
| **Deep Learning** | **PyTorch**, Torchvision | ResNet18 (Transfer Learning) |
| **Audio Proc** | **Librosa** | Audio-to-Spectrogram Transformation |
| **XAI** | **Grad-CAM** | Model Attention Visualization (Heatmap) |
| **Environment** | macOS (MPS) | GPU Acceleration Optimization |

## 3. Methodology
1.  **Data Preprocessing:** - Converted audio waveforms into **Mel-Spectrograms** (visual representations of sound).
    - Applied **Data Augmentation** by slicing tracks into 3-second segments to overcome data scarcity.
2.  **Model Training:** - Fine-tuned a pre-trained **ResNet18** model.
    - Optimized for 10 genre classes (Blues, Classical, Country, Disco, Hiphop, Jazz, Metal, Pop, Reggae, Rock).
3.  **Inference & Visualization:** - Analyzed personal R&B tracks.
    - Used **Grad-CAM** to generate heatmaps, revealing which frequency bands (Low/Mid/High) the model focused on.

## 4. Analysis Results (Case Study)
I tested the model with my own songs to analyze how it interprets "unseen" genres like R&B.

### Case 1: JAZZ CLUB (feat. AstralCube) (Melodic Rap) → Predicted: POP
<p align="center">
  <img src="results/heatmap_9624 JAZZ CLUB AR.wav.jpg" width="400" alt="Heatmap Jazz Club">
</p>

* **Context:** The title is inspired by **Maison Margiela's 'Replica Jazz Club' fragrance**, not the music genre. The track is actually **Singing Rap** (Melodic Rap).
* **AI Focus:** The heatmap highlights the **bottom area (Low Frequencies/Bass)**.
* **Insight:** The AI was not misled by the semantic title "Jazz". Instead, it correctly identified the acoustic features of **Singing Rap**—which blends rhythmic rapping with pop-like melodies and modern bass mixing—and classified it as **'POP'**, showing its ability to analyze the actual audio texture over metadata.

### Case 2: Lawson (R&B/HipHop) → Predicted: HIPHOP
<p align="center">
  <img src="results/heatmap_9624 Lawson. AR.wav.jpg" width="400" alt="Heatmap Lawson">
</p>

* **AI Focus:** The red zones are concentrated in **dense blocks at the bottom (Sub-bass)**.
* **Insight:** The model successfully identified the **808 sub-bass and kick drum patterns**, which are signature elements of Hip-Hop. This demonstrates the model's ability to recognize rhythmic textures.

### Case 3: Serenade, Pt. II (R&B) → Predicted: COUNTRY
<p align="center">
  <img src="results/heatmap_9624 SERENADE PT. II AR.wav.jpg" width="400" alt="Heatmap Serenade">
</p>

* **AI Focus:** The heatmap is spread across the **Mid-to-High frequencies (Vocals & Acoustics)**.
* **Insight:** Instead of the beat, the model focused on the **vocal melody and acoustic textures**. Since 'R&B' was not a label options, the AI mapped the **vocal-centric nature** of the track to 'COUNTRY', which shares similar acoustic characteristics in the training set.

## 5. Conclusion
This project demonstrated that CNNs interpret music not just by "listening" but by recognizing visual patterns in frequency distributions. Through Grad-CAM, I proved that **mixing balance (Bass vs. Treble)** and **instrumentation (Beat vs. Vocals)** are the key factors driving the AI's classification. This highlights the importance of **dataset diversity** and the value of **XAI tools** in debugging machine learning models.

---
*Developed by PARFUMDEWALKER*