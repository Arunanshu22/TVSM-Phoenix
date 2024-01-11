import sounddevice as sd
from scipy.io.wavfile import write
import wavio as wv

name = input("Enter your name: ")
freq = 44100
duration = 3
for i in range(20):
    file_name = "audio_clips/" + name + str(i+1) + '.wav'
    print("Recording file " + file_name)
    recording = sd.rec(int(duration * freq), samplerate=freq, channels=2)
    sd.wait()
    write(file_name, freq, recording)
    print("Recorded file " + file_name)
