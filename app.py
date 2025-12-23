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
PRESET_DIR = "presets"

if not os.path.exists(PRESET_DIR):
    os.makedirs(PRESET_DIR)

# ===== 共通関数 =====
def quantize_01(x):
    """
    0.00〜1.00 にクリップして 0.01 刻みに量子化
    """
    x = float(x)
    x = np.clip(x, 0.0, 1.0)
    return round(x / 0.01) * 0.01


# ===== UI =====
st.title("🎧 リバーブ解析ツール")

uploaded = st.file_uploader("音声ファイルをアップロード", type=["wav", "mp3", "flac"])

if uploaded is not None:
    # ===== 音声読み込み =====
    y, sr = librosa.load(uploaded, sr=TARGET_SR, mono=True)

    # ===== スペクトル解析 =====
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))
    S_db = librosa.amplitude_to_db(S, ref=np.max)

    # ===== 原音・残響の推定 =====
    # 前半を原音、後半を残響として扱う（簡易モデル）
    mid = S_db.shape[1] // 2
    dry_energy = np.mean(np.abs(S_db[:, :mid]))
    wet_energy = np.mean(np.abs(S_db[:, mid:]))

    # 正規化（0〜1）
    max_energy = max(dry_energy, wet_energy, 1e-9)
    dry = quantize_01(dry_energy / max_energy)
    wet = quantize_01(wet_energy / max_energy)

    # ===== 空間差分 =====
    spatial_diff = np.diff(S_db, axis=1)

    # ===== 部屋の広さ =====
    # 差分の平均エネルギー → 空間の広がり
    room_size_raw = np.mean(np.abs(spatial_diff))
    room_size = quantize_01(room_size_raw / np.max(np.abs(S_db)))

    # ===== 減衰 =====
    # フレーム間変化量 → 残響の減り方
    decay_raw = np.mean(np.abs(np.diff(spatial_diff)))
    decay = quantize_01(decay_raw / np.max(np.abs(S_db)))

    # ===== 表示 =====
    st.subheader("📊 解析結果")
    st.text(f"原音        : {dry:.2f}")
    st.text(f"残響        : {wet:.2f}")
    st.text(f"部屋の広さ  : {room_size:.2f}")
    st.text(f"減衰        : {decay:.2f}")

    # ===== プリセット保存 =====
    st.divider()
    preset_name = st.text_input("💾 プリセット名", "my_room_preset")

    if st.button("プリセット保存"):
        preset = {
            "name": preset_name,
            "dry": dry,
            "wet": wet,
            "room_size": room_size,
            "decay": decay
        }

        path = os.path.join(PRESET_DIR, preset_name + ".json")
        with open(path, "w") as f:
            json.dump(preset, f, indent=2, ensure_ascii=False)

        st.success(f"プリセット保存完了：{path}")
