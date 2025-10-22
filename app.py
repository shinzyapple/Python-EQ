import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import tempfile
import os

# ===== パラメータ =====
TARGET_SR = 44100
N_FFT = 1024
HOP_LENGTH = 256
MAX_GAIN = 30
FREQS = [
    20, 25, 31.5, 40, 50, 63, 80, 100,
    125, 160, 200, 250, 315, 400, 500,
    630, 800, 1000, 1250, 1600, 2000,
    2500, 3150, 4000, 5000, 6300, 8000,
    10000, 12500, 16000, 20000
]

# ===== 音声処理関数 =====
def compute_gain(audio_target, audio_ref):
    S_target = np.abs(librosa.stft(audio_target, n_fft=N_FFT, hop_length=HOP_LENGTH))
    S_ref = np.abs(librosa.stft(audio_ref, n_fft=N_FFT, hop_length=HOP_LENGTH))
    freqs_stft = np.linspace(0, TARGET_SR/2, S_target.shape[0])
    gain_db = []
    for f in FREQS:
        f_low = f / (2 ** (1/6))
        f_high = f * (2 ** (1/6))
        idx = np.where((freqs_stft >= f_low) & (freqs_stft <= f_high))[0]
        if len(idx) == 0:
            gain_db.append(0.0)
            continue
        mean_target = np.mean(S_target[idx, :])
        mean_ref = np.mean(S_ref[idx, :])
        g = mean_target / (mean_ref + 1e-8)
        g_db = 20 * np.log10(g + 1e-8)
        gain_db.append(np.clip(g_db, -MAX_GAIN, MAX_GAIN))
    return np.array(gain_db)

def apply_gain(audio, gain_db):
    S_complex = librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH)
    magnitude = np.abs(S_complex)
    phase = np.angle(S_complex)
    freqs_stft = np.linspace(0, TARGET_SR/2, S_complex.shape[0])
    gain_per_bin = np.ones_like(freqs_stft)
    for i, f in enumerate(FREQS):
        f_low = f / (2 ** (1/6))
        f_high = f * (2 ** (1/6))
        idx = np.where((freqs_stft >= f_low) & (freqs_stft <= f_high))[0]
        gain_per_bin[idx] = 10 ** (gain_db[i] / 20.0)
    S_adjusted = magnitude * gain_per_bin[:, np.newaxis] * np.exp(1j * phase)
    return librosa.istft(S_adjusted, hop_length=HOP_LENGTH)

# ===== Streamlit GUI =====
st.title("イコライザー補正 Webアプリ")

tab = st.tabs(["EQ作成 & 再生", "EQ適用 & 再生"])

with tab[0]:
    st.header("EQ作成 & 再生")
    target_file = st.file_uploader("録音ファイル", type=["wav", "mp3"])
    ref_file = st.file_uploader("音源ファイル", type=["wav", "mp3"])
    
    if st.button("EQ作成"):
        if target_file is None or ref_file is None:
            st.error("ファイルをアップロードしてください")
        else:
            # 一時ファイルとして保存
            with tempfile.NamedTemporaryFile(delete=False) as tf_target:
                tf_target.write(target_file.read())
                path_target = tf_target.name
            with tempfile.NamedTemporaryFile(delete=False) as tf_ref:
                tf_ref.write(ref_file.read())
                path_ref = tf_ref.name
            
            target, _ = librosa.load(path_target, sr=TARGET_SR, mono=True)
            ref, _ = librosa.load(path_ref, sr=TARGET_SR, mono=True)
            gain_db = compute_gain(target, ref)
            
            adjusted = apply_gain(ref, gain_db)
            out_path = "adjusted.wav"
            sf.write(out_path, adjusted, TARGET_SR)
            
            st.success("EQ作成完了！")
            st.audio(out_path)
            
            # EQバンド表示
            for i, g in enumerate(gain_db):
                st.text(f"{FREQS[i]} Hz: {int(round(g)):+d} dB")
            
            # EQファイルをバイナリで渡す
            with tempfile.NamedTemporaryFile(delete=False, suffix=".npy") as tf_eq:
                np.save(tf_eq.name, gain_db)
                with open(tf_eq.name, "rb") as f:
                    eq_bytes = f.read()
            st.download_button(
                label="EQファイルをダウンロード",
                data=eq_bytes,
                file_name="eq_gain.npy",
                mime="application/octet-stream"
            )

with tab[1]:
    st.header("EQ適用 & 再生")
    input_file = st.file_uploader("音声ファイル", type=["wav", "mp3"], key="input")
    gain_file = st.file_uploader("EQファイル", type=["npy"], key="gain")
    
    if st.button("EQ適用"):
        if input_file is None or gain_file is None:
            st.error("ファイルをアップロードしてください")
        else:
            # 一時ファイルとして保存
            with tempfile.NamedTemporaryFile(delete=False) as tf_input:
                tf_input.write(input_file.read())
                path_input = tf_input.name
            with tempfile.NamedTemporaryFile(delete=False, suffix=".npy") as tf_gain:
                tf_gain.write(gain_file.read())
                path_gain = tf_gain.name
            
            audio, _ = librosa.load(path_input, sr=TARGET_SR, mono=True)
            gain_db = np.load(path_gain)
            adjusted = apply_gain(audio, gain_db)
            
            out_path = "adjusted_apply.wav"
            sf.write(out_path, adjusted, TARGET_SR)
            st.success("EQ適用完了！")
            st.audio(out_path)
            st.download_button(
                label="補正音声をダウンロード",
                data=open(out_path, "rb").read(),
                file_name="adjusted_apply.wav",
                mime="audio/wav"
            )
