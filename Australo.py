#Import des différents modules nécesaires

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
import soundfile
import torchaudio
from pydub import AudioSegment
from pydub.playback import play
from speechbrain.pretrained import EncoderDecoderASR

asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-crdnn-commonvoice-fr", savedir="pretrained_models/asr-crdnn-commonvoice-fr", run_opts={"device":"cuda"} )

# Détecte la parole et enregiste un fichier .wav de 1 sec
# silence limit : temps de silence qui force l'arrêt en seconde, silence_threshold: seuil de detection de la parole, prev_audio: temps d'enregistrement avant la detection du seuil (pour ne pas couper le premier mot)
def record_on_detect(file_name, silence_limit=0.5 , silence_threshold=2500, chunk=1024, rate=44100, prev_audio=0.8):
    CHANNELS = 1
    FORMAT = pyaudio.paInt16
    end=1
    current=0
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
        if(sum([x > silence_threshold for x in slid_window]) > 0) and current<=end:
            if(not started):
                print("sound detected")
                current=time.time()
                #Temps du fichier modifiable
                end=time.time()+ 1
                started = True
        elif (started is True):
            started = False
            listen = False
            prev_audio = deque(maxlen=int(0.5 * rel))
        if (started is True):
            current=time.time()
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


# Fonction à exécuter pour tester le script
def lucie():
    while True:
        record_on_detect('test')
        #Affichage de la prédiction du moidèle
        print(asr_model.transcribe_file('C:/Users/Mathis/Desktop/CODEV/test.wav'))
        if 2 >= levenshtein(asr_model.transcribe_file('C:/Users/Mathis/Desktop/CODEV/test.wav'),'lucie'):
            #Exécute la fonction t2s après 0 sec
            t1 = threading.Timer(0,t2s,['Blitz enclenché',1])
            t1.start()
            temps_demarrage=time.time()
            #Exécute la fonction bip après 150 sec
            t2 = threading.Timer(150,bip,[1])
            t2.start()
            t3 = threading.Timer(180,t2s,['fin du temps',2])
            t3.start()
            t4 = threading.Timer(210,longbip,[1])
            t4.start()


# Calcul de distance entre chaîne de caractères
def levenshtein(chaine1i, chaine2i):
    chaine1 = ''.join(char for char in chaine1i if char.isalnum()).lower()
    chaine2 = ''.join(char for char in chaine2i if char.isalnum()).lower()
    # print(chaine1,chaine2) affichage possible pour voir ce que le modèle déduit
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

# Utilise le google text to speech pour renvoyer du son
def t2s(txt,num):
    tts = gtts.gTTS(txt,lang='fr')
    tts.save('C://Users//Mathis/Desktop//CODEV/textetospeech'+str(num)+'.mp3')
    playsound('C://Users//Mathis//Desktop//CODEV/textetospeech'+str(num)+'.mp3')

def bip(num):
    playsound('C://Users//Mathis//Desktop//CODEV/petitbip'+str(num)+'.mp3')

def longbip(num):
    playsound('C://Users//Mathis//Desktop//CODEV/longbip'+str(num)+'.mp3')