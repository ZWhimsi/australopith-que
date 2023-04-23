from speechbrain.pretrained import EncoderDecoderASR
from speechbrain.dataio.preprocess import AudioNormalizer
from speechbrain.processing.features import InputNormalization
import torch
import soundfile
from scipy.spatial.distance import cityblock
chemin='C:/Users/Mathis/Desktop'
### crdnn FR ###
asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-crdnn-commonvoice-fr", savedir="pretrained_models/asr-crdnn-commonvoice-fr")

### crdnn_librispeech ###
#asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-crdnn-rnnlm-librispeech", savedir="pretrained_models/asr-crdnn-rnnlm-librispeech")
##
import torchaudio
##
wav_normalizer = AudioNormalizer(sample_rate=16000, mix='avg-to-mono')
compute_fourier = asr_model.mods.encoder.compute_features
fourier_normalizer = asr_model.mods.encoder.normalize
crdnn = asr_model.mods.encoder.model
avg_pool = torch.nn.AdaptiveAvgPool2d((1,1))
sim_fn = torch.nn.CosineSimilarity()
process1 = torch.nn.Sequential(compute_fourier, cnn) # regler le normalizer de waveform et de STFT
##
example_file1 = 'C:/Users/Mathis/Downloads/baleine.wav'
signal1 ,sr1 = torchaudio.load(example_file1, normalize=True)
r_32to16 = torchaudio.transforms.Resample(32000, 16000)
signal=r_32to16(signal1)
signal1 = signal1.swapaxes(-2,-1)
normalized1= wav_normalizer(signal1,16000)
norm = InputNormalization()
features1 = crdnn(norm(compute_fourier(normalized1.unsqueeze(0)),torch.ones([1])))
print(features1.size())
##
example_file2 = 'C:/Users/Mathis/Downloads/jag1.wav'
signal2 ,sr2 = torchaudio.load(example_file2, normalize=True)
r_32to16 = torchaudio.transforms.Resample(32000, 16000)
signal2=r_32to16(signal2)
signal2 = signal2.swapaxes(-2,-1)
normalized2= wav_normalizer(signal2,16000)
norm = InputNormalization()
features2 = crdnn(norm(compute_fourier(normalized2.unsqueeze(0)),torch.ones([1])))
print(features2.size())
##
example_file3 = 'C:/Users/Mathis/Desktop/CODEV/jaguar.wav'
signal3 ,sr3 = torchaudio.load(example_file3, normalize=True)
r_32to16 = torchaudio.transforms.Resample(32000, 16000)
signal3=r_32to16(signal3)
signal3 = signal3.swapaxes(-2,-1)
normalized3= wav_normalizer(signal3,16000)
norm = InputNormalization()
features3 = crdnn(norm(compute_fourier(normalized3.unsqueeze(0)),torch.ones([1])))
print(features3.size())
##
print(avg_pool(features1))
print(avg_pool(features2))
print(avg_pool(features3))

print(sim_fn(avg_pool(features1),avg_pool(features2)))
print(sim_fn(avg_pool(features3),avg_pool(features2)))
print(sim_fn(avg_pool(features1),avg_pool(features3)))