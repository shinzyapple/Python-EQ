import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import tempfile
import json

# =====================
# 基本設定
# =====================
TARGET_SR = 44100
N_FFT = 1024
HOP_LENGTH = 256
MAX_GAIN = 30
STEP = 0.01

FREQS = [
    20, 25, 31.5, 40, 50, 63, 80, 100,
    125, 160, 200, 250, 315, 400, 500,
    630, 800, 1000, 1250, 1600, 2000,
    2500, 3150, 4000, 5000, 6300, 8000,
    10000, 12500, 16000, 20000
]

# =====================
# 共通ユーティリティ
# =====================
def q01(x):
    return round(np.clip(x, 0.0, 1.0) / STEP) * STEP

# =====================
# EQ解析
# =====================
def compute_gain(audio_target, audio_ref):
    S_target = np.abs(librosa.stft(audio_target, n_fft=N_FFT, hop_length=HOP_LENGTH))
    S_ref = np.abs(librosa.stft(audio_ref, n_fft=N_FFT, hop_length=HOP_LENGTH))

    freqs_stft = np.linspace(0, TARGET_SR / 2, S_target.shape[0])
    gain_db = []

    for f in FREQS:
        f_low = f / (2 ** (1/6))
        f_high = f * (2 ** (1/6))
        idx = np.where((freqs_stft >= f_low) & (freqs_stft <= f_high))[0]

        if len(idx) == 0:
            gain_db.append(0.0)
            continue

        t = np.mean(S_target[idx, :])
        r = np.mean(S_ref[idx, :])
        g = 20 * np.log10((t + 1e-8) / (r + 1e-8))
        gain_db.append(float(np.clip(g, -MAX_GAIN, MAX_GAIN)))

    return np.array(gain_db)

# =====================
# リバーブ解析
# =====================
def analyze_reverb(y, sr):
    rms = librosa.feature.rms(y=y)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr)

    # ---- 原音 / 残響 ----
    early = rms[times < 0.08]
    late = rms[times >= 0.08]

    e = np.mean(early)
    l = np.mean(late)
    total = e + l + 1e-9

    direct = q01(e / total)
    reverb = q01(l / total)

    # ---- 部屋の広さ（RT60）----
    rms_db = 20 * np.log10(rms + 1e-6)
    peak = np.max(rms_db)

    try:
        rt60 = times[np.where(rms_db < peak - 60)[0][0]]
    except IndexError:
        rt60 = times[-1]

    room_size = q01(rt60 / 3.0)

    # ---- 減衰 ----
    valid = rms_db > peak - 60
    slope, _ = np.polyfit(times[valid], rms_db[valid], 1)
    decay = q01((-slope - 5) / 35)

    return {
        "direct": direct,
        "reverb": reverb,
        "room_size": room_size,
        "decay": decay
    }

# =====================
# Streamlit UI
# =====================
st.title("🎧 録音 → 音響プリセット抽出アプリ")

st.markdown("### 音声ファイルをアップロード")

col1, col2 = st.columns(2)
with col1:
    rec_file = st.file_uploader("🎙 録音ファイル", type=["wav", "mp3"])
with col2:
    ref_file = st.file_uploader("🎼 音源ファイル", type=["wav", "mp3"])

if st.button("解析する"):
    if rec_file is None or ref_file is None:
        st.error("両方のファイルをアップロードしてね")
    else:
        with tempfile.NamedTemporaryFile(delete=False) as f1:
            f1.write(rec_file.read())
            rec_path = f1.name
        with tempfile.NamedTemporaryFile(delete=False) as f2:
            f2.write(ref_file.read())
            ref_path = f2.name

        y_rec, _ = librosa.load(rec_path, sr=TARGET_SR, mono=True)
        y_ref, _ = librosa.load(ref_path, sr=TARGET_SR, mono=True)

        # ---- 解析 ----
        eq_gain = compute_gain(y_rec, y_ref)
        reverb_params = analyze_reverb(y_rec, TARGET_SR)

        # =====================
        # 表示
        # =====================
        st.markdown("## 🎚 EQ補正値")
        for f, g in zip(FREQS, eq_gain):
            st.text(f"{f:>6} Hz : {g:+.1f} dB")

        st.markdown("## 🏠 リバーブ特性（0.01刻み）")
        st.text(f"原音        : {reverb_params['direct']:.2f}")
        st.text(f"残響        : {reverb_params['reverb']:.2f}")
        st.text(f"部屋の広さ  : {reverb_params['room_size']:.2f}")
        st.text(f"減衰        : {reverb_params['decay']:.2f}")

        # =====================
        # プリセット保存
        # =====================
        preset = {
            "eq_gain_db": eq_gain.tolist(),
            "reverb": reverb_params
        }

        preset_json = json.dumps(preset, indent=2, ensure_ascii=False)

        st.download_button(
            "📥 プリセットとして保存",
            data=preset_json,
            file_name="audio_preset.json",
            mime="application/json"
        )
