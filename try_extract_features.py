import librosa
import librosa.display
from matplotlib import pyplot

genre = "pop"
num = "00008"

audio_filepath = "./genres/" + genre + "/" + genre + "." + num + ".wav"
x, sr = librosa.load(audio_filepath, sr = 44100) #'sr' is the sample rate

print(x)
print(len(x))

wave_plot = pyplot.figure(figsize =  (13, 5))
librosa.display.waveshow(x, sr)
wave_plot.savefig("waveplot_" + genre + "_" + num + ".png")
pyplot.close()

stft_data = librosa.stft(x)
stft_data_db = librosa.amplitude_to_db(abs(stft_data))
spectrogram = pyplot.figure(figsize = (13, 5))
librosa.display.specshow(stft_data_db, sr = sr, x_axis = "time", y_axis = "hz")
pyplot.color()
spectrogram.savefig("waveplot_" + genre + "_" + num + ".png")
pyplot.close()

hop_length = 44100
librosa.feature.chroma_stft(x, sr = sr, hop_length = hop_length)