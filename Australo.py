# brew install portaudio
# pip install pyaudio
import whisper
import time
import torch
import sys
import pyaudio
import wave
import audioop
import math
from collections import deque
import numpy as np
import threading
import gtts
from playsound import playsound
# Silence limit in
# only silence is recorded. When this time passes the
# recording finishes and the file is delivered.

# The silence threshold intensity that defines silence
# and noise signal (an int. lower than THRESHOLD is silence).

# Previous audio (in seconds) to prepend. When noise
# is detected, how much of previously recorded audio is
# prepended. This helps to prevent chopping the beggining
# of the phrase.

print(torch.cuda.is_available())
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model=whisper.load_model("medium",device=DEVICE)
##
def record_on_detect(file_name, silence_limit=1, silence_threshold=2500, chunk=1024, rate=44100, prev_audio=1):
  CHANNELS = 1
  FORMAT = pyaudio.paInt16

  p = pyaudio.PyAudio()
  stream = p.open(format=p.get_format_from_width(2),
                  channels=CHANNELS,
                  rate=rate,
                  input=True,
                  output=False,
                  frames_per_buffer=chunk)

  listen = True
  started = False
  rel = rate/chunk
  frames = []

  prev_audio = deque(maxlen=int(prev_audio * rel))
  slid_window = deque(maxlen=int(silence_limit * rel))

  while listen:
    data = stream.read(chunk)
    slid_window.append(math.sqrt(abs(audioop.avg(data, 4))))

    if(sum([x > silence_threshold for x in slid_window]) > 0):
      if(not started):
        print("Starting record of phrase")
        started = True
    elif (started is True):
      started = False
      listen = False
      prev_audio = deque(maxlen=int(0.5 * rel))

    if (started is True):
      frames.append(data)
    else:
      prev_audio.append(data)

  stream.stop_stream()
  stream.close()

  p.terminate()


  wf = wave.open('C:/Users/Mathis/Desktop/CODEV/'f'{file_name}.wav', 'wb')
  wf.setnchannels(CHANNELS)
  wf.setsampwidth(p.get_sample_size(FORMAT))
  wf.setframerate(rate)

  wf.writeframes(b''.join(list(prev_audio)))
  wf.writeframes(b''.join(frames))
  wf.close()

def whisper():
    result1=model.transcribe('C:/Users/Mathis/Desktop/CODEV/example.wav')
    return result1['text']

def lucie():
    while True:
        record_on_detect('example')
        txt=whisper()
        print(txt)
        if levenshtein(txt,'lucieblitz')<=2:
            t1 = threading.Timer(0,t2s,['Blitz enclenché',1])
            t1.start()
            temps_demarrage=time.time()
            t2 = threading.Timer(150,bip,[1])
            t2.start()
            t3 = threading.Timer(180,t2s,['fin du temps',2])
            t3.start()
            t4 = threading.Timer(210,longbip,[1])
            t4.start()
        if levenshtein(txt,'luciestop')<=2:
            temps_ecoule=temps_demarrage-time.time()
            t2.cancel()
            t3.cancel()
            t4.cancel()
        if levenshtein(txt,'luciestart')<=2:
            t5 = threading.Timer(0,t2s,['Blitz enclenché',3])
            t5.start()
            t6 = threading.Timer(150-round(temps_ecoule),bip[2])
            t6.start()
            t7 = threading.Timer(180-round(temps_ecoule),t2s,['fin du temps',4])
            t7.start()
            t8 = threading.Timer(210-round(temps_ecoule),longbip,[2])
            t8.start()
        if levenshtein(txt,'lucieminutes')<=3:
            tps=''
            for elt in txt:
                if elt in ['1','2','3','4','5','6','7','8','9']:
                    tps+=elt
            t9 = threading.Timer(tps.int()*60,t2s,['Temps écoulé',5])
            t9.start()




def levenshtein(chaine1i, chaine2i):
    chaine1 = ''.join(char for char in chaine1i if char.isalnum()).lower()
    chaine2 = ''.join(char for char in chaine2i if char.isalnum()).lower()
    taille_chaine1 = len(chaine1) + 1
    taille_chaine2 = len(chaine2) + 1
    levenshtein_matrix = np.zeros ((taille_chaine1, taille_chaine2))
    for x in range(taille_chaine1):
        levenshtein_matrix [x, 0] = x
    for y in range(taille_chaine2):
        levenshtein_matrix [0, y] = y
    for x in range(1, taille_chaine1):
        for y in range(1, taille_chaine2):
            if chaine1[x-1] == chaine2[y-1]:
                levenshtein_matrix [x,y] = min(
                    levenshtein_matrix[x-1, y] + 1,
                    levenshtein_matrix[x-1, y-1],
                    levenshtein_matrix[x, y-1] + 1
                )
            else:
                levenshtein_matrix [x,y] = min(
                    levenshtein_matrix[x-1,y] + 1,
                    levenshtein_matrix[x-1,y-1] + 1,
                    levenshtein_matrix[x,y-1] + 1
                )
    return (levenshtein_matrix[taille_chaine1 - 1, taille_chaine2 - 1])

def t2s(txt,num):
    tts = gtts.gTTS(txt,lang='fr')
    tts.save('C:/Users/Mathis/Desktop/CODEV/textetospeech'+str(num)+'.mp3')
    playsound('C:/Users/Mathis/Desktop/CODEV/textetospeech'+str(num)+'.mp3')

def bip(num):
    playsound('C:/Users//Mathis//Desktop/CODEV/petitbip'+str(num)+'.mp3')

def longbip(num):
    playsound('C:/Users/Mathis/Desktop/CODEV/longbip'+str(num)+'.mp3')