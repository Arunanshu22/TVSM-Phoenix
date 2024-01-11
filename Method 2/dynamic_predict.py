# Audio file collected from user and is placed in "buffer" folder from which the audio file is picked and tested
# on the trained model

# import packages

import sounddevice as sd
from scipy.io.wavfile import write
import wavio as wv
import librosa
import joblib
import numpy as np

# collect audio clip
freq = 44100
duration = 3
for i in range(1):
    file_name = "buffer/test_this.wav"
    print("Recording file " + file_name)
    recording = sd.rec(int(duration * freq), samplerate=freq, channels=2)
    sd.wait()
    write(file_name, freq, recording)
    print("Recorded file " + file_name)

# preprocess the audio file

audio_file_path = 'buffer/test_this.wav'
y, sr = librosa.load(audio_file_path)
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128 * 2,)
S_db_mel = librosa.amplitude_to_db(S, ref=np.max)
desired_shape = (1, 33280)
S_db_mel_flat_reshaped = S_db_mel.flatten()[:desired_shape[1]].reshape(desired_shape)
gaus_model = joblib.load('model.joblib')
print(gaus_model.predict(S_db_mel_flat_reshaped))
