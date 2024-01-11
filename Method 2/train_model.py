# this file is to split the audio files collected and then train a naive bayes model using the data
# the naive bayes model is saved as "model.joblib"


# import packages

import numpy as np
import pandas as pd
import librosa
import librosa.display

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
import os

import joblib

# create dataframe

df = pd.DataFrame(columns=['name', 'audio'])

files = [f for f in os.listdir("audio_clips/") if os.path.isfile(os.path.join("audio_clips/", f))]
for file in files:
    y, sr = librosa.load("audio_clips/" + file)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128 * 2,)
    S_db_mel = librosa.amplitude_to_db(S, ref=np.max)
    if "ganesh" in file:
        new_row = {"name": "ganesh", "audio": S_db_mel}
    elif "sagar" in file:
        new_row = {"name": "sagar", "audio": S_db_mel}
    elif "kothi" in file:
        new_row = {"name": "kothi", "audio": S_db_mel}
    df = df._append(new_row, ignore_index=True)

# split into train and test

X = df['audio']
X = np.stack(X.values)
y = df['name']
y = np.stack(y.values)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train model

# Flatten the 3D arrays into 2D arrays
X_train_flattened = X_train.reshape(X_train.shape[0], -1)
X_test_flattened = X_test.reshape(X_test.shape[0], -1)

gaus_model = GaussianNB()
gaus_model.fit(X_train_flattened, y_train)
gaus_pred = gaus_model.predict(X_test_flattened)
gaus_accuracy = accuracy_score(y_test, gaus_pred)

# store the model

joblib.dump(gaus_model, 'model.joblib')