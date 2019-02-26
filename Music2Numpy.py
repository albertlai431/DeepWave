import numpy as np
from scipy.io.wavfile import read 
from scipy.io.wavfile import write

file = 'D:\SampleNoiseRecording.wav'

def Conv2Numpy(file):
    a = read(file)
    #convert it to an array
    sample = np.array(a[1],dtype=float)
    return sample

data = Conv2Numpy(file)

def Conv2Music(data):
    scaled = np.int16(data/np.max(np.abs(data)) * 32767)
    #Put In Root Directory 
    write('CleanAudio.wav', 44100, scaled)
    print("Done Converting!")

Conv2Music(data)