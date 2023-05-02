import librosa
import matplotlib.pyplot as plt

# sr表示采样率
# None使用原频率
y, sr = librosa.load("./data/qy1.wav", sr=None, mono=False)
print(y.shape, sr)
ax = plt.subplot()
ax.set(xlim=[0.23, 0.35], title="time")
librosa.display.waveshow(y, sr=sr, ax=ax)
# librosa.display.waveshow(y_harm, sr=sr, alpha=0.5, ax=ax2, label='Harmonic')
# librosa.display.waveshow(y_perc, sr=sr, color='r', alpha=0.5, ax=ax2, label='Percussive')
# ax.label_outer()
# ax.legend()
z = y[int(sr * 0.23) : int(sr * 0.35)]
S = librosa.stft(y, n_fft=len(z))
print(S)
print(S.shape)
# librosa.display.waveshow(z, sr=sr, axis="time")
# plt.show()
