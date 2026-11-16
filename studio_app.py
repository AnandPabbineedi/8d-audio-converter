"""
Audio Studio Pro — Python Edition
Gradio-based multi-tab audio processing suite.
Powered by: pedalboard (Spotify), librosa, soundfile, numpy, scipy
"""

import numpy as np
import soundfile as sf
import librosa
import tempfile
import os
import gradio as gr

from pedalboard import (
    Pedalboard,
    Reverb,
    PitchShift,
    LowShelfFilter,
    HighpassFilter,
    Compressor,
)
from pedalboard.io import AudioFile
from scipy.signal import butter, sosfilt


# ─────────────────────────────────────────────
# UTILITY
# ─────────────────────────────────────────────

def load_audio(path: str, target_sr: int = 44100):
    """Load audio file, always return float32 stereo (2, N)."""
    audio, sr = librosa.load(path, sr=target_sr, mono=False)
    if audio.ndim == 1:
        audio = np.stack([audio, audio])   # mono → stereo
    return audio.astype(np.float32), sr


def save_wav(audio: np.ndarray, sr: int, suffix="_out") -> str:
    """Save float32 stereo (2, N) to a temp WAV, return path."""
    tmp = tempfile.NamedTemporaryFile(suffix=f"{suffix}.wav", delete=False)
    # soundfile wants (N, channels)
    sf.write(tmp.name, audio.T, sr, subtype="PCM_16")
    return tmp.name


# ─────────────────────────────────────────────
# 1. 8D SPATIAL AUDIO
# ─────────────────────────────────────────────

def process_8d(audio_path, speed_hz, width_pct, reverb_room, reverb_wet):
    """
    8D effect: LFO-based stereo panning + reverb.
    - Generates a sine LFO at speed_hz that pans between -width and +width.
    - Applies a Pedalboard Reverb to simulate room space.
    """
    audio, sr = load_audio(audio_path)
    n = audio.shape[1]
    t = np.arange(n) / sr

    # Sine LFO for panning [-1, 1] × width
    width = width_pct / 100.0
    lfo = np.sin(2 * np.pi * speed_hz * t) * width   # shape (N,)

    # Apply panning: left = cos((pan+1)*π/4), right = sin((pan+1)*π/4)
    pan_angle = (lfo + 1) * np.pi / 4          # 0..π/2
    left_gain  = np.cos(pan_angle)
    right_gain = np.sin(pan_angle)

    out = np.stack([
        audio[0] * left_gain  + audio[1] * left_gain,
        audio[0] * right_gain + audio[1] * right_gain,
    ]).astype(np.float32)

    # Reverb via pedalboard
    board = Pedalboard([
        Reverb(
            room_size=float(reverb_room) / 100.0,
            wet_level=float(reverb_wet) / 100.0,
            dry_level=1.0 - float(reverb_wet) / 200.0,
            damping=0.5,
        )
    ])
    out = board(out, sr)

    return save_wav(out, sr, "_8d")


# ─────────────────────────────────────────────
# 2. SLOWED + REVERB
# ─────────────────────────────────────────────

def process_slowed(audio_path, speed_rate, reverb_wet, room_size):
    """
    Slowed: time-stretch to slow down (without pitch change) using librosa.
    Reverb: pedalboard Reverb.
    """
    audio, sr = load_audio(audio_path)

    # Time-stretch each channel independently (preserves pitch)
    stretched = np.stack([
        librosa.effects.time_stretch(audio[0], rate=float(speed_rate)),
        librosa.effects.time_stretch(audio[1], rate=float(speed_rate)),
    ]).astype(np.float32)

    board = Pedalboard([
        Reverb(
            room_size=float(room_size) / 100.0,
            wet_level=float(reverb_wet) / 100.0,
            dry_level=1.0 - float(reverb_wet) / 200.0,
            damping=0.6,
            width=0.8,
        )
    ])
    out = board(stretched, sr)
    return save_wav(out, sr, "_slowed")


# ─────────────────────────────────────────────
# 3. PITCH SHIFTER (true — no tempo change)
# ─────────────────────────────────────────────

def process_pitch(audio_path, semitones):
    """True pitch shift via pedalboard PitchShift (phase vocoder)."""
    audio, sr = load_audio(audio_path)
    board = Pedalboard([PitchShift(semitones=float(semitones))])
    out = board(audio, sr)
    return save_wav(out, sr, "_pitch")


# ─────────────────────────────────────────────
# 4. VOCAL REMOVER (Center-Channel Extraction)
# ─────────────────────────────────────────────

def process_vocal(audio_path, cut_depth, notch_freq):
    """
    Vocal removal via stereo differential subtraction:
    Instrumental = Left - (depth × Right) → removes center-panned content.
    Then apply a notch filter around the vocal frequency band.
    """
    audio, sr = load_audio(audio_path)
    depth = float(cut_depth) / 100.0

    # Center cancel: L - depth*R  (reduces shared center content)
    side_l = audio[0] - depth * audio[1]
    side_r = audio[1] - depth * audio[0]

    # Notch filter around vocal frequencies
    def notch(sig, freq, q=10.0):
        bw = freq / q
        low  = (freq - bw / 2) / (sr / 2)
        high = (freq + bw / 2) / (sr / 2)
        low  = max(0.001, min(low, 0.999))
        high = max(0.001, min(high, 0.999))
        if low >= high:
            return sig
        sos = butter(4, [low, high], btype='bandstop', output='sos')
        return sosfilt(sos, sig)

    freq = float(notch_freq)
    out = np.stack([
        notch(side_l, freq).astype(np.float32),
        notch(side_r, freq).astype(np.float32),
    ])

    # Normalize to avoid clipping
    peak = np.max(np.abs(out))
    if peak > 0.98:
        out = out / peak * 0.98

    return save_wav(out, sr, "_instrumental")


# ─────────────────────────────────────────────
# 5. BASS & VOLUME BOOSTER
# ─────────────────────────────────────────────

def process_bass(audio_path, bass_gain_db, cutoff_hz, volume_x):
    """
    Low-shelf boost via pedalboard LowShelfFilter + master gain.
    """
    audio, sr = load_audio(audio_path)
    board = Pedalboard([
        LowShelfFilter(
            cutoff_frequency_hz=float(cutoff_hz),
            gain_db=float(bass_gain_db),
        )
    ])
    out = board(audio, sr) * float(volume_x)

    # Anti-clip
    peak = np.max(np.abs(out))
    if peak > 0.98:
        out = out / peak * 0.98

    return save_wav(out, sr, "_bass")


# ─────────────────────────────────────────────
# 6. BPM & KEY DETECTOR
# ─────────────────────────────────────────────

MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def detect_bpm_key(audio_path):
    """
    BPM: librosa.beat.beat_track (onset-strength autocorrelation).
    Key: Chroma energy profile correlated against Krumhansl-Schmuckler key profiles.
    """
    audio, sr = load_audio(audio_path, target_sr=22050)
    mono = librosa.to_mono(audio)

    # BPM
    tempo, _ = librosa.beat.beat_track(y=mono, sr=sr)
    bpm = round(float(np.atleast_1d(tempo)[0]), 1)

    # Chroma
    chroma = librosa.feature.chroma_cqt(y=mono, sr=sr)
    chroma_mean = chroma.mean(axis=1)   # shape (12,)

    best_score = -np.inf
    best_key = "C Major"

    for i in range(12):
        rolled_major = np.roll(MAJOR_PROFILE, i)
        rolled_minor = np.roll(MINOR_PROFILE, i)

        score_major = np.corrcoef(chroma_mean, rolled_major)[0, 1]
        score_minor = np.corrcoef(chroma_mean, rolled_minor)[0, 1]

        if score_major > best_score:
            best_score = score_major
            best_key = f"{NOTES[i]} Major"
        if score_minor > best_score:
            best_score = score_minor
            best_key = f"{NOTES[i]} Minor"

    return f"🥁 {bpm} BPM", f"🎼 {best_key}"


# ─────────────────────────────────────────────
# GRADIO UI
# ─────────────────────────────────────────────

THEME = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="green",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
).set(
    button_primary_background_fill="#0b57d0",
    button_primary_background_fill_hover="#0842a0",
    button_primary_text_color="#ffffff",
)

with gr.Blocks(theme=THEME, title="Audio Studio Pro") as app:

    gr.HTML("""
    <div style="display:flex;align-items:center;gap:14px;padding:20px 0 8px;">
      <div style="display:flex;gap:6px;">
        <span style="width:10px;height:10px;border-radius:50%;background:#4285F4;display:inline-block;"></span>
        <span style="width:10px;height:10px;border-radius:50%;background:#EA4335;display:inline-block;"></span>
        <span style="width:10px;height:10px;border-radius:50%;background:#FBBC05;display:inline-block;"></span>
        <span style="width:10px;height:10px;border-radius:50%;background:#34A853;display:inline-block;"></span>
      </div>
      <span style="font-size:1.5rem;font-weight:800;letter-spacing:-0.02em;">Audio Studio Pro</span>
      <span style="background:#e6f4ea;color:#146c2e;border:1px solid #a8d5b5;padding:3px 10px;border-radius:999px;font-size:0.78rem;font-weight:700;">Python Edition</span>
    </div>
    <p style="color:#444746;font-size:0.95rem;margin-bottom:8px;">Professional audio processing suite — 100% client-side, studio-grade quality.</p>
    <hr style="border:none;border-top:1px solid #e0e0e0;margin:8px 0 0;">
    """)

    # ── TAB 1: 8D SPATIAL ──────────────────────
    with gr.Tab("🎧 8D Spatial Audio"):
        gr.Markdown("### 8D Spatial Audio Studio\nSine LFO binaural panning with real convolution reverb.")
        with gr.Row():
            with gr.Column(scale=1):
                in_8d = gr.Audio(label="Upload Track", type="filepath")
                gr.Markdown("**Presets**")
                with gr.Row():
                    p8d_classic = gr.Button("🎧 Classic 8D",   size="sm")
                    p8d_hypno   = gr.Button("🌌 Deep Hypnotic", size="sm")
                    p8d_cyber   = gr.Button("🚀 Cyberpunk",     size="sm")
                    p8d_hall    = gr.Button("🏛️ Concert Hall", size="sm")
                speed_8d  = gr.Slider(0.05, 0.60, value=0.15, step=0.01, label="Panning Speed (Hz)")
                width_8d  = gr.Slider(0, 100,  value=90,   step=1,    label="Orbit Width (%)")
                room_8d   = gr.Slider(0, 100,  value=40,   step=1,    label="Room Size (%)")
                wet_8d    = gr.Slider(0, 100,  value=35,   step=1,    label="Reverb Wet (%)")
                btn_8d    = gr.Button("⚡ Process & Export 8D WAV", variant="primary")
            with gr.Column(scale=1):
                out_8d = gr.Audio(label="8D Output", type="filepath")

        # Preset buttons
        p8d_classic.click(lambda: (0.15, 90, 40, 35), outputs=[speed_8d, width_8d, room_8d, wet_8d])
        p8d_hypno.click(  lambda: (0.08, 100, 55, 55), outputs=[speed_8d, width_8d, room_8d, wet_8d])
        p8d_cyber.click(  lambda: (0.35, 95, 25, 20),  outputs=[speed_8d, width_8d, room_8d, wet_8d])
        p8d_hall.click(   lambda: (0.12, 85, 75, 60),  outputs=[speed_8d, width_8d, room_8d, wet_8d])
        btn_8d.click(process_8d, inputs=[in_8d, speed_8d, width_8d, room_8d, wet_8d], outputs=out_8d)

    # ── TAB 2: SLOWED + REVERB ─────────────────
    with gr.Tab("🌌 Slowed + Reverb"):
        gr.Markdown("### Slowed + Reverb Studio\nTrue time-stretch (pitch preserved) + studio reverb.")
        with gr.Row():
            with gr.Column(scale=1):
                in_sl = gr.Audio(label="Upload Track", type="filepath")
                gr.Markdown("**Presets**")
                with gr.Row():
                    psl_mid  = gr.Button("🌌 Midnight Drive", size="sm")
                    psl_eth  = gr.Button("✨ Ethereal Abyss", size="sm")
                    psl_chill= gr.Button("🛋️ Subtle Chill",  size="sm")
                    psl_deep = gr.Button("🌀 Ultra Deep",     size="sm")
                speed_sl = gr.Slider(0.50, 1.00, value=0.85, step=0.01, label="Tempo Rate (0.5 = half speed)")
                wet_sl   = gr.Slider(0, 100,    value=55,   step=1,    label="Reverb Wet (%)")
                room_sl  = gr.Slider(0, 100,    value=60,   step=1,    label="Room Size (%)")
                btn_sl   = gr.Button("⚡ Process & Export Slowed WAV", variant="primary")
            with gr.Column(scale=1):
                out_sl = gr.Audio(label="Slowed Output", type="filepath")

        psl_mid.click(  lambda: (0.85, 55, 60), outputs=[speed_sl, wet_sl, room_sl])
        psl_eth.click(  lambda: (0.75, 80, 75), outputs=[speed_sl, wet_sl, room_sl])
        psl_chill.click(lambda: (0.90, 35, 40), outputs=[speed_sl, wet_sl, room_sl])
        psl_deep.click( lambda: (0.65, 90, 85), outputs=[speed_sl, wet_sl, room_sl])
        btn_sl.click(process_slowed, inputs=[in_sl, speed_sl, wet_sl, room_sl], outputs=out_sl)

    # ── TAB 3: PITCH SHIFTER ───────────────────
    with gr.Tab("🎵 Pitch Shifter"):
        gr.Markdown("### Pitch Shifter\nTrue pitch shift via phase vocoder — no tempo change.")
        with gr.Row():
            with gr.Column(scale=1):
                in_pt = gr.Audio(label="Upload Track", type="filepath")
                gr.Markdown("**Presets**")
                with gr.Row():
                    pp_orig  = gr.Button("🎵 Original",    size="sm")
                    pp_nc    = gr.Button("⚡ Nightcore",   size="sm")
                    pp_slow  = gr.Button("🌌 Slowed",      size="sm")
                    pp_chip  = gr.Button("🐿️ Chipmunk",   size="sm")
                    pp_deep  = gr.Button("🎙️ Deep Voice",  size="sm")
                    pp_up    = gr.Button("🎼 Octave Up",   size="sm")
                    pp_down  = gr.Button("🎸 Octave Down", size="sm")
                pitch_st = gr.Slider(-12, 12, value=0, step=1, label="Pitch (Semitones)")
                btn_pt   = gr.Button("⚡ Process & Export Pitch WAV", variant="primary")
            with gr.Column(scale=1):
                out_pt = gr.Audio(label="Pitch Shifted Output", type="filepath")

        pp_orig.click( lambda:  0,  outputs=pitch_st)
        pp_nc.click(   lambda:  3,  outputs=pitch_st)
        pp_slow.click( lambda: -3,  outputs=pitch_st)
        pp_chip.click( lambda:  6,  outputs=pitch_st)
        pp_deep.click( lambda: -6,  outputs=pitch_st)
        pp_up.click(   lambda:  12, outputs=pitch_st)
        pp_down.click( lambda: -12, outputs=pitch_st)
        btn_pt.click(process_pitch, inputs=[in_pt, pitch_st], outputs=out_pt)

    # ── TAB 4: VOCAL REMOVER ───────────────────
    with gr.Tab("🎤 Vocal Remover"):
        gr.Markdown("### Vocal Remover & Instrumental Isolator\nSpectral center-channel subtraction + bandstop notch filter.")
        with gr.Row():
            with gr.Column(scale=1):
                in_vc = gr.Audio(label="Upload Stereo Track", type="filepath")
                gr.Markdown("**Presets**")
                with gr.Row():
                    pvc_agg   = gr.Button("🎤 Aggressive",    size="sm")
                    pvc_mod   = gr.Button("🎙️ Moderate",      size="sm")
                    pvc_kara  = gr.Button("🎹 Karaoke Mode",  size="sm")
                    pvc_bypass= gr.Button("🎵 Bypass",        size="sm")
                cut_vc  = gr.Slider(0, 100, value=100, step=5,  label="Vocal Cut Depth (%)")
                freq_vc = gr.Slider(300, 3400, value=1000, step=50, label="Notch Center Frequency (Hz)")
                btn_vc  = gr.Button("⚡ Process & Export Instrumental WAV", variant="primary")
            with gr.Column(scale=1):
                out_vc = gr.Audio(label="Instrumental Output", type="filepath")
                gr.Markdown("""
> ⚠️ **Note:** Client-side vocal removal works best on stereo tracks where vocals are panned to center.
> For AI-grade stem separation, consider [Demucs](https://github.com/facebookresearch/demucs).
                """)

        pvc_agg.click(   lambda: (100, 1000), outputs=[cut_vc, freq_vc])
        pvc_mod.click(   lambda: (70, 1500),  outputs=[cut_vc, freq_vc])
        pvc_kara.click(  lambda: (50, 2200),  outputs=[cut_vc, freq_vc])
        pvc_bypass.click(lambda: (0, 1000),   outputs=[cut_vc, freq_vc])
        btn_vc.click(process_vocal, inputs=[in_vc, cut_vc, freq_vc], outputs=out_vc)

    # ── TAB 5: BASS BOOSTER ────────────────────
    with gr.Tab("🔊 Bass & Volume"):
        gr.Markdown("### Bass & Volume Booster\nLowshelf EQ boost via pedalboard + master volume gain.")
        with gr.Row():
            with gr.Column(scale=1):
                in_bs = gr.Audio(label="Upload Track", type="filepath")
                gr.Markdown("**Presets**")
                with gr.Row():
                    pbs_clean = gr.Button("🔊 Clean (+6dB)",  size="sm")
                    pbs_club  = gr.Button("💥 Club (+12dB)",  size="sm")
                    pbs_max   = gr.Button("🔥 Max (+18dB)",   size="sm")
                    pbs_flat  = gr.Button("🎵 Flat",          size="sm")
                bass_db  = gr.Slider(0, 20,  value=6,   step=1,   label="Bass Gain (dB, Lowshelf)")
                bass_cut = gr.Slider(60, 500, value=200, step=10,  label="Bass Cutoff Frequency (Hz)")
                vol_x    = gr.Slider(0.5, 2.5, value=1.2, step=0.1, label="Master Volume (×)")
                btn_bs   = gr.Button("⚡ Process & Export Bass WAV", variant="primary")
            with gr.Column(scale=1):
                out_bs = gr.Audio(label="Bass Boosted Output", type="filepath")

        pbs_clean.click(lambda: (6,  200, 1.2), outputs=[bass_db, bass_cut, vol_x])
        pbs_club.click( lambda: (12, 200, 1.5), outputs=[bass_db, bass_cut, vol_x])
        pbs_max.click(  lambda: (18, 200, 2.0), outputs=[bass_db, bass_cut, vol_x])
        pbs_flat.click( lambda: (0,  200, 1.0), outputs=[bass_db, bass_cut, vol_x])
        btn_bs.click(process_bass, inputs=[in_bs, bass_db, bass_cut, vol_x], outputs=out_bs)

    # ── TAB 6: BPM & KEY ───────────────────────
    with gr.Tab("🥁 BPM & Key Detector"):
        gr.Markdown("### BPM & Musical Key Detector\nLibrosa beat tracking + Krumhansl–Schmuckler chroma key profiling.")
        with gr.Row():
            with gr.Column(scale=1):
                in_bpm = gr.Audio(label="Upload Track", type="filepath")
                btn_bpm = gr.Button("🥁 Analyze Tempo & Key", variant="primary")
            with gr.Column(scale=1):
                bpm_out = gr.Textbox(label="Detected BPM", interactive=False)
                key_out = gr.Textbox(label="Detected Key",  interactive=False)
                gr.Markdown("""
**How it works:**
- **BPM** — `librosa.beat.beat_track` uses onset-strength autocorrelation for accurate tempo estimation.
- **Key** — Chroma energy vectors are correlated against the Krumhansl-Schmuckler key profiles for all 24 major/minor keys.
                """)

        btn_bpm.click(detect_bpm_key, inputs=in_bpm, outputs=[bpm_out, key_out])


if __name__ == "__main__":
    app.launch(inbrowser=True, share=False)
