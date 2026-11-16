# Audio Studio Pro & 8D Spatial Audio Suite

A comprehensive audio manipulation suite featuring both a client-side Web Audio Studio, a Real-Time Live Audio Mixer, and a high-fidelity Python DSP engine powered by Gradio, Spotify's Pedalboard, and Librosa.

---

## 🚀 Features

### 1. Python DSP Studio (`studio_app.py`)
Powered by **Spotify Pedalboard**, **Librosa**, **SoundFile**, and **Gradio**:
- **🎧 8D Spatial Audio**: Sine LFO binaural orbital panner with dynamic room reverberation.
- **🌌 Slowed + Reverb**: True time-stretch tempo slowdown preserving pitch without chipmunk distortion + studio convolution reverb.
- **🎵 Phase-Vocoder Pitch Shifter**: True semitone pitch shifter (-12 to +12 st) without tempo alteration.
- **🎤 Center-Channel Vocal Remover**: Differential stereo phase cancellation with custom vocal-frequency notch filter.
- **🔊 Bass & Volume Booster**: Low-shelf biquad filtering with dynamic master gain normalization.
- **🥁 BPM & Musical Key Detector**: Librosa onset-strength autocorrelation tempo estimation and Krumhansl–Schmuckler chroma tonal profiling.

### 2. Client-Side Web Audio Suite (`index.html`)
- 100% client-side Web Audio API processing.
- Persistent Light/Dark theme switcher.
- Dual visualizers: Real-time 360° Orbit Radar & Live Spectrum Analyzer.
- Universal seekable playback scrubbers.
- Lossless 16-bit PCM and 32-bit Float WAV exporter.

### 3. Live Audio Mixer Pro (`mixer.html`)
- Real-time live audio capture from system audio (Apple Music, Spotify, YouTube) via `getDisplayMedia` or live microphone input via `getUserMedia`.
- Live real-time audio graph DSP with instant effects chain (Spatial LFO, Delay Reverb, Vocal Cut, Bass Boost).
- Live frequency spectrum bars and oscilloscope waveform canvas.

---

## 🛠️ Quick Start

### Python DSP Studio
```bash
# 1. Install dependencies
pip install gradio pedalboard librosa soundfile numpy scipy pydub

# 2. Run the application
python studio_app.py
```
Access the web interface at `http://127.0.0.1:7860`.

### Client-Side Suite & Live Mixer
Simply open `index.html` or `mixer.html` in any modern web browser (Google Chrome or Microsoft Edge recommended for system audio capture).

---

## 📄 License
MIT License
