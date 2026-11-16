"""
Audio Studio Pro — Python Edition
High-performance audio processing suite powered by Spotify Pedalboard, Librosa, and Gradio.
"""

import os
import uuid
import numpy as np
import soundfile as sf
import librosa
import gradio as gr

from pedalboard import (
    Pedalboard,
    Reverb,
    PitchShift,
    LowShelfFilter,
)
from scipy.signal import butter, sosfilt

# Dedicated exports directory
EXPORTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exports")
os.makedirs(EXPORTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# AUDIO IO UTILITIES
# ─────────────────────────────────────────────

def load_audio(path: str, target_sr: int = 44100):
    """Load audio file safely, return float32 stereo (2, N)."""
    audio, sr = librosa.load(path, sr=target_sr, mono=False)
    if audio.ndim == 1:
        audio = np.stack([audio, audio])  # convert mono to stereo
    return audio.astype(np.float32), sr


def save_wav(audio: np.ndarray, sr: int, suffix="_out") -> str:
    """Save float32 stereo (2, N) to an uncorrupted WAV file in exports/."""
    peak = np.max(np.abs(audio))
    if peak > 1.0:
        audio = (audio / peak) * 0.95

    filename = f"{uuid.uuid4().hex[:8]}{suffix}.wav"
    filepath = os.path.join(EXPORTS_DIR, filename)
    # Write cleanly and close file handle immediately
    sf.write(filepath, audio.T, sr, subtype="PCM_16")
    return filepath


# ─────────────────────────────────────────────
# 1. 8D SPATIAL AUDIO
# ─────────────────────────────────────────────

def process_8d(audio_path, speed_hz, width_pct, reverb_room, reverb_wet):
    """
    True 360-degree binaural spatial rotation panner + room acoustics.
    """
    if not audio_path:
        raise gr.Error("Please upload an audio track first!")

    audio, sr = load_audio(audio_path)
    n = audio.shape[1]
    t = np.arange(n, dtype=np.float32) / float(sr)

    # 360-degree continuous circular panning
    theta = 2.0 * np.pi * float(speed_hz) * t
    width = float(width_pct) / 100.0

    # Constant-power panning law
    x = np.sin(theta) * width  # [-1, 1]
    pan_angle = (x + 1.0) * (np.pi / 4.0)

    left_gain = np.cos(pan_angle)
    right_gain = np.sin(pan_angle)

    # Apply stereo panning preserving stereo field
    out_l = audio[0] * left_gain + audio[1] * (1.0 - right_gain) * 0.3
    out_r = audio[1] * right_gain + audio[0] * (1.0 - left_gain) * 0.3
    out = np.stack([out_l, out_r]).astype(np.float32)

    # Studio reverb
    if reverb_wet > 0:
        board = Pedalboard([
            Reverb(
                room_size=float(reverb_room) / 100.0,
                wet_level=float(reverb_wet) / 100.0,
                dry_level=1.0 - (float(reverb_wet) / 200.0),
                damping=0.5,
            )
        ])
        out = board(out, sr)

    # Anti-clipping normalization
    peak = np.max(np.abs(out))
    if peak > 0:
        out = (out / peak) * 0.95

    path = save_wav(out, sr, "_8d")
    return path, path


# ─────────────────────────────────────────────
# 2. SLOWED + REVERB
# ─────────────────────────────────────────────

def process_slowed(audio_path, speed_rate, reverb_wet, room_size):
    """
    Time-stretch tempo slowdown (pitch preserved) + atmospheric hall reverb.
    """
    if not audio_path:
        raise gr.Error("Please upload an audio track first!")

    audio, sr = load_audio(audio_path)
    rate = float(speed_rate)

    stretched = np.stack([
        librosa.effects.time_stretch(audio[0], rate=rate),
        librosa.effects.time_stretch(audio[1], rate=rate),
    ]).astype(np.float32)

    if reverb_wet > 0:
        board = Pedalboard([
            Reverb(
                room_size=float(room_size) / 100.0,
                wet_level=float(reverb_wet) / 100.0,
                dry_level=1.0 - (float(reverb_wet) / 200.0),
                damping=0.6,
                width=0.9,
            )
        ])
        stretched = board(stretched, sr)

    peak = np.max(np.abs(stretched))
    if peak > 0:
        stretched = (stretched / peak) * 0.95

    path = save_wav(stretched, sr, "_slowed")
    return path, path


# ─────────────────────────────────────────────
# 3. PITCH SHIFTER (phase-vocoder — tempo unchanged)
# ─────────────────────────────────────────────

def process_pitch(audio_path, semitones):
    """
    True pitch-shifting without changing tempo.
    """
    if not audio_path:
        raise gr.Error("Please upload an audio track first!")

    audio, sr = load_audio(audio_path)
    board = Pedalboard([PitchShift(semitones=float(semitones))])
    out = board(audio, sr)

    peak = np.max(np.abs(out))
    if peak > 0:
        out = (out / peak) * 0.95

    path = save_wav(out, sr, "_pitch")
    return path, path


# ─────────────────────────────────────────────
# 4. VOCAL REMOVER (Center Channel Extraction)
# ─────────────────────────────────────────────

def process_vocal(audio_path, cut_depth, notch_freq):
    """
    Vocal suppression via stereo phase subtraction + vocal notch filtering.
    """
    if not audio_path:
        raise gr.Error("Please upload an audio track first!")

    audio, sr = load_audio(audio_path)
    depth = float(cut_depth) / 100.0

    side_l = audio[0] - depth * audio[1]
    side_r = audio[1] - depth * audio[0]

    def notch(sig, freq, q=5.0):
        bw = freq / q
        low = (freq - bw / 2.0) / (sr / 2.0)
        high = (freq + bw / 2.0) / (sr / 2.0)
        low = max(0.001, min(low, 0.999))
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

    peak = np.max(np.abs(out))
    if peak > 0:
        out = (out / peak) * 0.95

    path = save_wav(out, sr, "_instrumental")
    return path, path


# ─────────────────────────────────────────────
# 5. BASS & VOLUME BOOSTER
# ─────────────────────────────────────────────

def process_bass(audio_path, bass_gain_db, cutoff_hz, volume_x):
    """
    Lowshelf filter boost + master volume gain.
    """
    if not audio_path:
        raise gr.Error("Please upload an audio track first!")

    audio, sr = load_audio(audio_path)
    board = Pedalboard([
        LowShelfFilter(
            cutoff_frequency_hz=float(cutoff_hz),
            gain_db=float(bass_gain_db),
        )
    ])
    out = board(audio, sr) * float(volume_x)

    peak = np.max(np.abs(out))
    if peak > 0:
        out = (out / peak) * 0.95

    path = save_wav(out, sr, "_bass")
    return path, path


# ─────────────────────────────────────────────
# 6. BPM & KEY DETECTOR
# ─────────────────────────────────────────────

MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def detect_bpm_key(audio_path):
    """
    Librosa beat tracking + Krumhansl–Schmuckler chroma key profiling.
    """
    if not audio_path:
        raise gr.Error("Please upload an audio track first!")

    audio, sr = load_audio(audio_path, target_sr=22050)
    mono = librosa.to_mono(audio)

    # BPM
    tempo, _ = librosa.beat.beat_track(y=mono, sr=sr)
    bpm = round(float(np.atleast_1d(tempo)[0]), 1)

    # Key profiling
    chroma = librosa.feature.chroma_cqt(y=mono, sr=sr)
    chroma_mean = chroma.mean(axis=1)

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
# GRADIO APPLICATION
# ─────────────────────────────────────────────

with gr.Blocks(title="Audio Studio Pro") as app:

    gr.HTML("""
    <div style="display:flex;align-items:center;gap:14px;padding:16px 0 6px;">
      <div style="display:flex;gap:6px;">
        <span style="width:10px;height:10px;border-radius:50%;background:#4285F4;display:inline-block;"></span>
        <span style="width:10px;height:10px;border-radius:50%;background:#EA4335;display:inline-block;"></span>
        <span style="width:10px;height:10px;border-radius:50%;background:#FBBC05;display:inline-block;"></span>
        <span style="width:10px;height:10px;border-radius:50%;background:#34A853;display:inline-block;"></span>
      </div>
      <span style="font-size:1.6rem;font-weight:800;letter-spacing:-0.02em;">Audio Studio Pro</span>
      <span style="background:#e6f4ea;color:#146c2e;border:1px solid #a8d5b5;padding:3px 10px;border-radius:999px;font-size:0.78rem;font-weight:700;">Python Edition</span>
    </div>
    <p style="color:#5f6368;font-size:0.95rem;margin-bottom:8px;">Studio-grade DSP audio processing suite powered by Spotify's Pedalboard and Librosa.</p>
    <hr style="border:none;border-top:1px solid #dadce0;margin:6px 0 16px;">
    """)

    # ── TAB 1: 8D SPATIAL ──────────────────────
    with gr.Tab("🎧 8D Spatial Audio"):
        gr.Markdown("### 8D Spatial Audio Studio\nTrue 360° binaural rotation LFO panner with early reflection room reverb.")
        with gr.Row():
            with gr.Column(scale=1):
                in_8d = gr.Audio(label="Upload Audio Track", type="filepath")
                gr.Markdown("**Quick Presets**")
                with gr.Row():
                    p8d_classic = gr.Button("🎧 Classic 8D",   size="sm")
                    p8d_hypno   = gr.Button("🌌 Deep Hypnotic", size="sm")
                    p8d_cyber   = gr.Button("🚀 Cyberpunk",     size="sm")
                    p8d_hall    = gr.Button("🏛️ Concert Hall", size="sm")
                speed_8d = gr.Slider(0.05, 0.60, value=0.15, step=0.01, label="Panning Speed (Hz)")
                width_8d = gr.Slider(0, 100,  value=90,   step=1,    label="Orbit Width (%)")
                room_8d  = gr.Slider(0, 100,  value=40,   step=1,    label="Room Size (%)")
                wet_8d   = gr.Slider(0, 100,  value=35,   step=1,    label="Reverb Wet (%)")
                btn_8d   = gr.Button("⚡ Process & Export 8D WAV", variant="primary")
            with gr.Column(scale=1):
                out_8d_audio = gr.Audio(label="8D Audio Player")
                out_8d_file  = gr.File(label="⬇ Download 8D WAV File")

        p8d_classic.click(lambda: (0.15, 90, 40, 35), outputs=[speed_8d, width_8d, room_8d, wet_8d])
        p8d_hypno.click(  lambda: (0.08, 100, 55, 55), outputs=[speed_8d, width_8d, room_8d, wet_8d])
        p8d_cyber.click(  lambda: (0.35, 95, 25, 20),  outputs=[speed_8d, width_8d, room_8d, wet_8d])
        p8d_hall.click(   lambda: (0.12, 85, 75, 60),  outputs=[speed_8d, width_8d, room_8d, wet_8d])
        btn_8d.click(process_8d, inputs=[in_8d, speed_8d, width_8d, room_8d, wet_8d], outputs=[out_8d_audio, out_8d_file])

    # ── TAB 2: SLOWED + REVERB ─────────────────
    with gr.Tab("🌌 Slowed + Reverb"):
        gr.Markdown("### Slowed + Reverb Studio\nTrue time-stretch (pitch preserved) + studio convolution reverb.")
        with gr.Row():
            with gr.Column(scale=1):
                in_sl = gr.Audio(label="Upload Audio Track", type="filepath")
                gr.Markdown("**Quick Presets**")
                with gr.Row():
                    psl_mid  = gr.Button("🌌 Midnight Drive", size="sm")
                    psl_eth  = gr.Button("✨ Ethereal Abyss", size="sm")
                    psl_chill= gr.Button("🛋️ Subtle Chill",  size="sm")
                    psl_deep = gr.Button("🌀 Ultra Deep",     size="sm")
                speed_sl = gr.Slider(0.50, 1.00, value=0.85, step=0.01, label="Tempo Rate (0.85 = slowed)")
                wet_sl   = gr.Slider(0, 100,    value=55,   step=1,    label="Reverb Wet (%)")
                room_sl  = gr.Slider(0, 100,    value=60,   step=1,    label="Room Size (%)")
                btn_sl   = gr.Button("⚡ Process & Export Slowed WAV", variant="primary")
            with gr.Column(scale=1):
                out_sl_audio = gr.Audio(label="Slowed Audio Player")
                out_sl_file  = gr.File(label="⬇ Download Slowed WAV File")

        psl_mid.click(  lambda: (0.85, 55, 60), outputs=[speed_sl, wet_sl, room_sl])
        psl_eth.click(  lambda: (0.75, 80, 75), outputs=[speed_sl, wet_sl, room_sl])
        psl_chill.click(lambda: (0.90, 35, 40), outputs=[speed_sl, wet_sl, room_sl])
        psl_deep.click( lambda: (0.65, 90, 85), outputs=[speed_sl, wet_sl, room_sl])
        btn_sl.click(process_slowed, inputs=[in_sl, speed_sl, wet_sl, room_sl], outputs=[out_sl_audio, out_sl_file])

    # ── TAB 3: PITCH SHIFTER ───────────────────
    with gr.Tab("🎵 Pitch Shifter"):
        gr.Markdown("### Pitch Shifter\nTrue phase-vocoder pitch shift (-12 to +12 semitones) without tempo changes.")
        with gr.Row():
            with gr.Column(scale=1):
                in_pt = gr.Audio(label="Upload Audio Track", type="filepath")
                gr.Markdown("**Quick Presets**")
                with gr.Row():
                    pp_orig  = gr.Button("🎵 Original (0)", size="sm")
                    pp_nc    = gr.Button("⚡ Nightcore (+3)", size="sm")
                    pp_slow  = gr.Button("🌌 Slowed (-3)", size="sm")
                    pp_chip  = gr.Button("🐿️ Chipmunk (+6)", size="sm")
                    pp_deep  = gr.Button("🎙️ Deep (-6)", size="sm")
                    pp_up    = gr.Button("🎼 Octave Up (+12)", size="sm")
                    pp_down  = gr.Button("🎸 Octave Down (-12)", size="sm")
                pitch_st = gr.Slider(-12, 12, value=0, step=1, label="Pitch (Semitones)")
                btn_pt   = gr.Button("⚡ Process & Export Pitch WAV", variant="primary")
            with gr.Column(scale=1):
                out_pt_audio = gr.Audio(label="Pitch Shifted Player")
                out_pt_file  = gr.File(label="⬇ Download Pitch WAV File")

        pp_orig.click( lambda:  0,  outputs=pitch_st)
        pp_nc.click(   lambda:  3,  outputs=pitch_st)
        pp_slow.click( lambda: -3,  outputs=pitch_st)
        pp_chip.click( lambda:  6,  outputs=pitch_st)
        pp_deep.click( lambda: -6,  outputs=pitch_st)
        pp_up.click(   lambda:  12, outputs=pitch_st)
        pp_down.click( lambda: -12, outputs=pitch_st)
        btn_pt.click(process_pitch, inputs=[in_pt, pitch_st], outputs=[out_pt_audio, out_pt_file])

    # ── TAB 4: VOCAL REMOVER ───────────────────
    with gr.Tab("🎤 Vocal Remover"):
        gr.Markdown("### Vocal Remover & Instrumental Isolator\nCenter-channel stereo cancellation + frequency notch filtering.")
        with gr.Row():
            with gr.Column(scale=1):
                in_vc = gr.Audio(label="Upload Stereo Track", type="filepath")
                gr.Markdown("**Quick Presets**")
                with gr.Row():
                    pvc_agg   = gr.Button("🎤 Aggressive Cut", size="sm")
                    pvc_mod   = gr.Button("🎙️ Moderate Cut",   size="sm")
                    pvc_kara  = gr.Button("🎹 Karaoke Mode",   size="sm")
                    pvc_bypass= gr.Button("🎵 Bypass",         size="sm")
                cut_vc  = gr.Slider(0, 100, value=100, step=5, label="Vocal Cut Depth (%)")
                freq_vc = gr.Slider(300, 3400, value=1000, step=50, label="Notch Center Frequency (Hz)")
                btn_vc  = gr.Button("⚡ Process & Export Instrumental WAV", variant="primary")
            with gr.Column(scale=1):
                out_vc_audio = gr.Audio(label="Instrumental Player")
                out_vc_file  = gr.File(label="⬇ Download Instrumental WAV File")

        pvc_agg.click(   lambda: (100, 1000), outputs=[cut_vc, freq_vc])
        pvc_mod.click(   lambda: (70, 1500),  outputs=[cut_vc, freq_vc])
        pvc_kara.click(  lambda: (50, 2200),  outputs=[cut_vc, freq_vc])
        pvc_bypass.click(lambda: (0, 1000),   outputs=[cut_vc, freq_vc])
        btn_vc.click(process_vocal, inputs=[in_vc, cut_vc, freq_vc], outputs=[out_vc_audio, out_vc_file])

    # ── TAB 5: BASS BOOSTER ────────────────────
    with gr.Tab("🔊 Bass & Volume"):
        gr.Markdown("### Bass & Volume Booster\nLowshelf filter boost with dynamic volume gain.")
        with gr.Row():
            with gr.Column(scale=1):
                in_bs = gr.Audio(label="Upload Audio Track", type="filepath")
                gr.Markdown("**Quick Presets**")
                with gr.Row():
                    pbs_clean = gr.Button("🔊 Clean (+6dB)",  size="sm")
                    pbs_club  = gr.Button("💥 Club (+12dB)",  size="sm")
                    pbs_max   = gr.Button("🔥 Max (+18dB)",   size="sm")
                    pbs_flat  = gr.Button("🎵 Flat (0dB)",    size="sm")
                bass_db  = gr.Slider(0, 20,  value=6,   step=1,   label="Bass Gain (dB, Lowshelf)")
                bass_cut = gr.Slider(60, 500, value=200, step=10,  label="Bass Cutoff Frequency (Hz)")
                vol_x    = gr.Slider(0.5, 2.5, value=1.2, step=0.1, label="Master Volume Gain (×)")
                btn_bs   = gr.Button("⚡ Process & Export Bass WAV", variant="primary")
            with gr.Column(scale=1):
                out_bs_audio = gr.Audio(label="Bass Boosted Player")
                out_bs_file  = gr.File(label="⬇ Download Bass WAV File")

        pbs_clean.click(lambda: (6,  200, 1.2), outputs=[bass_db, bass_cut, vol_x])
        pbs_club.click( lambda: (12, 200, 1.5), outputs=[bass_db, bass_cut, vol_x])
        pbs_max.click(  lambda: (18, 200, 2.0), outputs=[bass_db, bass_cut, vol_x])
        pbs_flat.click( lambda: (0,  200, 1.0), outputs=[bass_db, bass_cut, vol_x])
        btn_bs.click(process_bass, inputs=[in_bs, bass_db, bass_cut, vol_x], outputs=[out_bs_audio, out_bs_file])

    # ── TAB 6: BPM & KEY ───────────────────────
    with gr.Tab("🥁 BPM & Key Detector"):
        gr.Markdown("### BPM & Musical Key Detector\nLibrosa beat tracking + Krumhansl–Schmuckler chroma key profiling.")
        with gr.Row():
            with gr.Column(scale=1):
                in_bpm = gr.Audio(label="Upload Audio Track", type="filepath")
                btn_bpm = gr.Button("🥁 Analyze Tempo & Key", variant="primary")
            with gr.Column(scale=1):
                bpm_out = gr.Textbox(label="Detected BPM", interactive=False)
                key_out = gr.Textbox(label="Detected Key",  interactive=False)

        btn_bpm.click(detect_bpm_key, inputs=in_bpm, outputs=[bpm_out, key_out])


if __name__ == "__main__":
    app.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)
