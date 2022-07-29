import librosa
import pandas
import glob
import numpy
import os

dataset = pandas.DataFrame()

genres = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
column_names = ["file name", "spectral_centroids", "spectral_rolloff", "bandwith", "zero_crossing", "chromagram", "mfcc 0", "mfcc 1", "mfcc 2", "mfcc 3", "mfcc 4","mfcc 5", "mfcc 6", "mfcc 7","mfcc 8", "mfcc 9","mfcc 10", "mfcc 11" ,"mfcc 12", "mfcc 13", "mfcc 14", "mfcc 15", "mfcc 16", "mfcc 17", "mfcc 18", "mfcc 19", "rms"]

dataset = pandas.DataFrame(columns = column_names)

old_data = []
if os.path.exists("dataset.csv"):
	old_data = pandas.read_csv("dataset.csv")["file name"].values
else:
	dataset.to_csv("dataset.csv", index = None)

for genre in genres:
	for file_name in glob.glob("./genres/" + genre + "/*.wav"):
		if (file_name in old_data):
			print(file_name, "exists.")
		else:
			print(file_name, "processing.")
			x, sr = librosa.load(file_name, sr = 44100)
			chromagram = librosa.feature.chroma_stft(x, sr = sr)
			spectral_centroids = librosa.feature.spectral_centroid(x, sr = sr)[0]
			spectral_rolloff = librosa.feature.spectral_rolloff(x + 0.01, sr = sr)[0]
			bandwith = librosa.feature.spectral_bandwidth(x, sr = sr)
			zero_crossing = librosa.feature.zero_crossing_rate(x)
			mfcc = librosa.feature.mfcc(x, sr = sr)
			rms = librosa.feature.rms(x)

			dataset = pandas.DataFrame(columns = column_names)

			dataset = dataset.append({
				"file name": file_name,
				"spectral_centroids" : numpy.mean(spectral_centroids),
				"spectral_rolloff" : numpy.mean(spectral_rolloff),
				"bandwith" : numpy.mean(bandwith),
				"zero_crossing" : numpy.mean(zero_crossing),
				"chromagram": numpy.mean(chromagram),
				"mfcc 0" : numpy.mean(mfcc[0]),
				"mfcc 1" : numpy.mean(mfcc[1]),
				"mfcc 2" : numpy.mean(mfcc[2]),
				"mfcc 3" : numpy.mean(mfcc[3]),
				"mfcc 4" : numpy.mean(mfcc[4]),
				"mfcc 5" : numpy.mean(mfcc[5]),
				"mfcc 6" : numpy.mean(mfcc[6]),
				"mfcc 7" : numpy.mean(mfcc[7]),
				"mfcc 8" : numpy.mean(mfcc[8]),
				"mfcc 9" : numpy.mean(mfcc[9]),
				"mfcc 10" : numpy.mean(mfcc[10]),
				"mfcc 11" : numpy.mean(mfcc[11]),
				"mfcc 12" : numpy.mean(mfcc[12]),
				"mfcc 13" : numpy.mean(mfcc[13]),
				"mfcc 14" : numpy.mean(mfcc[14]),
				"mfcc 15" : numpy.mean(mfcc[15]),
				"mfcc 16" : numpy.mean(mfcc[16]),
				"mfcc 17" : numpy.mean(mfcc[17]),
				"mfcc 18" : numpy.mean(mfcc[18]),
				"mfcc 19" : numpy.mean(mfcc[19]),
				"rms" : numpy.mean(rms)
				}, ignore_index =  True)

	#setting 'mode' to 'a' appends data through each iteration, but how the program knows what was the last file that needs to be appended to (i.e., useful if there's an electric outage)?
	dataset.to_csv("dataset.csv", index = None, header = False, mode = "a")

