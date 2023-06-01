#Import des différents modules nécesaires
import os
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
import psutil
import whisper
import openai
from argparse import ArgumentParser
from pathvalidate.argparse import validate_filename_arg, validate_filepath_arg

parser = ArgumentParser()
parser.add_argument("--filepath", type=validate_filepath_arg)

## à modifier 
openai.api_key = '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'
chemin= parser.parse_args().filepath
print(chemin)
## Load des modèles
whisper_model = whisper.load_model("medium")
asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-crdnn-commonvoice-fr", savedir="pretrained_models/asr-crdnn-commonvoice-fr", run_opts={"device":"cuda"} )


# Détecte la parole et enregiste un fichier .wav de 1 sec
# silence limit : temps de silence qui force l'arrêt en seconde, silence_threshold: seuil de detection de la parole, prev_audio: temps d'enregistrement avant la detection du seuil (pour ne pas couper le premier mot)
def record_on_detect(file_name, silence_limit=0.5 , silence_threshold=2000, chunk=1024, rate=44100, prev_audio=0.8,duree=0.5):
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
                end=time.time()+ duree
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


    wf = wave.open(f'{chemin}'f'{file_name}.wav','wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(rate)

    wf.writeframes(b''.join(list(prev_audio)))
    wf.writeframes(b''.join(frames))
    wf.close()

#Bot qui gère les interactions
def lucie(interactions=0):
    while True:
        record_on_detect('test')
        #Affichage de la prédiction du modèle
        print(asr_model.transcribe_file(chemin+'test.wav'))
        if 2 >= levenshtein(asr_model.transcribe_file(chemin+'test.wav'),'lucie'):
            interactions+=1
            t1 = threading.Timer(0,t2s,['Lucie est à vôtre écoute',interactions])
            t1.start()
            t1.cancel()
            while True:
                record_on_detect('test')
                print(asr_model.transcribe_file(chemin+'test.wav'))
                if 2 >= levenshtein(asr_model.transcribe_file(chemin+'test.wav'),'eclair'):
                    interactions+=1
                    t2 = threading.Timer(0,t2s,['Blitz enclenché',interactions])
                    t2.start()
                    t2.cancel()
                    t3 = threading.Timer(150,bip)
                    t3.start()
                    interactions+=1
                    t4 = threading.Timer(180,t2s,['Temps écoulé',interactions])
                    t4.start()
                    t5 = threading.Timer(210,longbip)
                    t5.start()
                    break
             
                if 2 >= levenshtein(asr_model.transcribe_file(chemin+'test.wav'),'annuler'):
                    interactions+=1
                    t6 = threading.Timer(0,t2s,['Que voulez vous annuler',interactions])
                    t6.start()
                    t6.cancel()
                    while True:
                        record_on_detect('test')
                        print(asr_model.transcribe_file(chemin+'test.wav'))
                        if 3 >= levenshtein(asr_model.transcribe_file(chemin+'test.wav'),'eclair'):
                            interactions+=1
                            t7 = threading.Timer(0,t2s,['Blitz annulé',interactions])
                            t7.start()
                            t7.cancel()
                            t3.cancel()
                            t4.cancel()
                            t5.cancel()
                            break
                   
                        if 3 >= levenshtein(asr_model.transcribe_file(chemin+'test.wav'),'question'):
                            interactions+=1
                            t7 = threading.Timer(0,t2s,['Question annulé',interactions])
                            t7.start()
                            t7.cancel()
                            t9.cancel()
                            break
                    break
              
                if 2 >= levenshtein(asr_model.transcribe_file(chemin+'test.wav'),'question'):
                    interactions+=1
                    t8 = threading.Timer(0,t2s,['Posez votre question',interactions])
                    t8.start()
                    t8.cancel()
                    while True:
                        record_on_detect('question',prev_audio=2, duree=10,silence_limit=1,silence_threshold=2500 )
                        question=whisper_model.transcribe(chemin+'question.wav')
                        print(question['text'])
                        result=question_chatgpt(question['text'])
                        interactions+=1
                        t9 = threading.Timer(0,t2s,[result,interactions])
                        t9.start()
                        t9.cancel()
                        break
                    break
               
                if 2 >= levenshtein(asr_model.transcribe_file(chemin+'test.wav'),'blague'):
                    result=question_chatgpt('Donne moi une excellente blague')
                    interactions+=1
                    t9 = threading.Timer(0,t2s,[result,interactions])
                    t9.start()
                    t9.cancel()
                    break
                 
                 
# Utilisation de l'api de chatgpt               
def question_chatgpt(question):
    response = openai.Completion.create(
        engine='text-davinci-003',
        prompt=question,
        max_tokens=50,
        n=1,
        stop=None,
        temperature=0.7
    )
    return response.choices[0].text.strip()

                 
# Calcul de distance entre chaîne de caractères
def levenshtein(chaine1, chaine2):
    chaine1 = ''.join(char for char in chaine1 if char.isalnum()).lower()
    chaine2 = ''.join(char for char in chaine2 if char.isalnum()).lower()
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

# Supprime les fichiers audios créé lors de la dernière utilisation 
def delete_audio_files():
    base_path = '/home/someone/Desktop/lucie/'
    num = 1  # Numéro initial
    #os.remove('C:/Users/Mathis/Desktop/CODEV/question.wav')
    while True:
        file_path = base_path + 'textetospeech' + str(num) + '.mp3'
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Deleted file: {file_path}")
        else:
            break  # Sort de la boucle si le fichier n'existe pas
        
        num += 1
        
        
# Utilise le google text to speech pour renvoyer du son
def t2s(txt,num):
    tts = gtts.gTTS(txt,lang='fr')
    tts.save('/home/someone/Desktop/lucie/textetospeech'+str(num)+'.mp3')
    playsound('/home/someone/Desktop/lucie/textetospeech'+str(num)+'.mp3')

def bip():
    playsound('/home/someone/Desktop/lucie/petitbip.mp3')

def longbip():
    playsound('/home/someone/Desktop/lucie/longbip.mp3')

# Fonction à exécuter pour tester le script    
def main():
    delete_audio_files()
    lucie()
    

main()
