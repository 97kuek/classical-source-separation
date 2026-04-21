import wave as wave                 #wave形式の音声波形を読み込むためのモジュール(wave)をインポート
import numpy as np                  #numpyをインポート（波形データを2byteの数値列に変換するために使用）
import scipy.signal as sp           #scipy.signalをインポート（短時間フーリエ変換と逆変換のために使用）
import sounddevice as sd            #sounddeviceをインポート（音声の再生のために使用）
import matplotlib.pyplot as plt     #matplotlib.pyplotをインポート（スペクトログラムの描画のために使用）

#ファイル・データ読み込み
sample_wave_file="./CMU_ARCTIC/cmu_us_aew_arctic/wav/arctic_a0001.wav"  #読み込むサンプルファイル
wav=wave.open(sample_wave_file)                                         #ファイルを読み込む
data=wav.readframes(wav.getnframes())                                   #PCM形式の波形データを読み込み

#dataを2バイトの数値列に変換
data=np.frombuffer(data, dtype=np.int16)

#短時間フーリエ変換を行う
f,t,stft_data=sp.stft(data,fs=wav.getframerate(),window="hann",nperseg=512,noverlap=256)

#フィルタリング前のスペクトログラムを保存
stft_data_pre = stft_data.copy()

#特定の周波数成分を消す(100番目の周波数よりも高い周波数成分を全て消す)
stft_data[100:,:]=0


#時間領域の波形に戻す
_,data_post=sp.istft(stft_data,fs=wav.getframerate(),window="hann",nperseg=512,noverlap=256)

#2バイトのデータに変換
data_post=data_post.astype(np.int16)

#スペクトログラムを描画して保存
fig, axes = plt.subplots(2, 1, figsize=(10, 6))

spec_pre = 20 * np.log10(np.abs(stft_data_pre) + 1e-10)
spec_post = 20 * np.log10(np.abs(stft_data) + 1e-10)
vmin, vmax = -100, spec_pre.max()

axes[0].set_title("Before filtering")
im0 = axes[0].pcolormesh(t, f, spec_pre, shading="auto", cmap="gray_r", vmin=vmin, vmax=vmax)
axes[0].set_ylabel("Frequency [Hz]")
axes[0].set_xlabel("Time [sec]")
fig.colorbar(im0, ax=axes[0], label="Intensity [dB]")

axes[1].set_title("After filtering (100th bin and above removed)")
im1 = axes[1].pcolormesh(t, f, spec_post, shading="auto", cmap="gray_r", vmin=vmin, vmax=vmax)
axes[1].set_ylabel("Frequency [Hz]")
axes[1].set_xlabel("Time [sec]")
fig.colorbar(im1, ax=axes[1], label="Intensity [dB]")

plt.tight_layout()
plt.savefig("./2-11_spectrogram.png")
plt.show()
print("スペクトログラムを 2-11_spectrogram.png に保存しました")

#wavファイルとして保存
output_wav_file = "./2-11_output.wav"
with wave.open(output_wav_file, "w") as out_wav:
    out_wav.setnchannels(wav.getnchannels())
    out_wav.setsampwidth(wav.getsampwidth())
    out_wav.setframerate(wav.getframerate())
    out_wav.writeframes(data_post.tobytes())
print(f"wavファイルを {output_wav_file} に保存しました")

#dataを再生する
sd.play(data_post,wav.getframerate())

print("再生開始")

#再生が終わるまで待つ
status = sd.wait()

#waveファイルを閉じる
wav.close()
