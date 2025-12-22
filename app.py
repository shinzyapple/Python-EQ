import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import json
import os

# ===== パラメータ =====
TARGET_SR = 44100
N_FFT = 1024
HOP_LENGTH = 256
MAX_GAIN = 30
STEP = 0.05
PRESET_DIR = "presets"

FREQS = [
    20, 25, 31.5, 40, 50, 63, 80, 100,
    125, 160, 200, 250, 315, 400, 500,
    630, 800, 1000, 1250, 1600, 2000,
    2500, 3150, 4000, 5000, 6300, 8000,
    10000, 12500, 16000, 20000
]

os.makedirs(PRESET_DIR, exist_ok=True)

# ===== 共通 =====
def quantize(val):
    return round(val / STEP) * STEP

def normalize(val, mn, mx):
    val = np.clip((val - mn) / (mx - mn), 0, 1)
    return quantize(val * 2 - 1)

# ===== 空間解析 =====
def analyze_spatial(y, sr):
    rms = librosa.feature.rms(y=y)[0]
    t = librosa.frames_to_time(range(len(rms)), sr=sr)

    direct = np.mean(rms[t < 0.1])
    reverb = np.mean(rms[t >= 0.1])

    decay = np.polyfit(t, np.log(rms + 1e-6), 1)[0]

    max_e = np.max(rms)
    rt60 = (
        t[np.where(rms < max_e * 1e-3)[0][0]]
        if np.any(rms < max_e * 1e-3)
        else t[-1]
    )

    return {
        "direct": normalize(direct, 0, 0.1),
        "reverb": normalize(reverb, 0, 0.1),
        "room_size": normalize(rt60, 0, 3.0),
        "decay": normalize(decay, -10, 0),
    }

# ===== EQ解析 =====
def compute_gain(target, ref):
    S_t = np.abs(librosa.stft(target, n_fft=N_FFT, hop_length=HOP_LENGTH))
    S_r = np.abs(librosa.stft(ref, n_fft=N_FFT, hop_length=HOP_LENGTH))
    freqs = np.linspace(0, TARGET_SR / 2, S_t.shape[0])

    gain = []
    for f in FREQS:
        low, high = f / (2 ** (1 / 6)), f * (2 ** (1 / 6))
        idx = np.where((freqs >= low) & (freqs <= high))[0]
        if len(idx) == 0:
            gain.append(0.0)
            continue

        g = np.mean(S_t[idx]) / (np.mean(S_r[idx]) + 1e-8)
        g_db = 20 * np.log10(g + 1e-8)
        gain.append(float(np.clip(g_db, -MAX_GAIN, MAX_GAIN)))

    return np.array(gain, dtype=float)

# ===== UI =====
st.set_page_config(page_title="音響プリセット生成ツール", layout="centered")
st.title("🎧 録音に近づける音響プリセット生成")

st.markdown("### 音源アップロード")
rec = st.file_uploader("🎤 録音音源（基準）", type=["wav", "mp3"])
ref = st.file_uploader("🎛 調整対象音源", type=["wav", "mp3"])

if rec and ref and st.button("解析してプリセット生成"):
    y_rec, _ = librosa.load(rec, sr=TARGET_SR, mono=True)
    y_ref, _ = librosa.load(ref, sr=TARGET_SR, mono=True)

    spatial_rec = analyze_spatial(y_rec, TARGET_SR)
    spatial_ref = analyze_spatial(y_ref, TARGET_SR)

    spatial_diff = {
        k: quantize(spatial_rec[k] - spatial_ref[k])
        for k in spatial_rec
    }

    eq_gain = compute_gain(y_rec, y_ref)

    st.markdown("## 🎚 EQ 推奨設定")
    for f, g in zip(FREQS, eq_gain):
        # ★ ここが修正ポイント（安全なfloat表示）
        st.text(f"{f:>6} Hz : {g:+.1f} dB")

    st.markdown("## 🌌 リバーブ 推奨設定")
    st.text(f"原音        : {spatial_diff['direct']:+}")
    st.text(f"残響        : {spatial_diff['reverb']:+}")
    st.text(f"部屋の広さ  : {spatial_diff['room_size']:+}")
    st.text(f"減衰        : {spatial_diff['decay']:+}")

    preset_name = st.text_input("💾 プリセット名", "my_room_preset")

    if st.button("プリセット保存"):
        preset = {
            "name": preset_name,
            "eq_gain_db": eq_gain.tolist(),
            "spatial": spatial_diff
        }

        path = os.path.join(PRESET_DIR, preset_name + ".json")
        with open(path, "w") as f:
            json.dump(preset, f, indent=2)

        st.success(f"プリセット保存完了：{path}")
