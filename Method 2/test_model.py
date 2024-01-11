# load the trained model and test individual file for fun :)

# import packages

import joblib
import librosa
import numpy as np

# preprocess the chosen file

audio_file_path = 'audio_clips/ganesh15.wav'
y, sr = librosa.load(audio_file_path)
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128 * 2,)
S_db_mel = librosa.amplitude_to_db(S, ref=np.max)
desired_shape = (1, 33280)
S_db_mel_flat_reshaped = S_db_mel.flatten()[:desired_shape[1]].reshape(desired_shape)
gaus_model = joblib.load('model.joblib')
print(gaus_model.predict(S_db_mel_flat_reshaped))
