

import torch
import os
from torch import nn
from torch.utils.data import Dataset, DataLoader
from scipy.io import wavfile
import numpy as np
import simpleaudio as sa


# DataSet

class AudioDataset(Dataset):

    def __init__(self, dataRoot, sampleRate = 16000, nMix = 2, soundLen = 5, dataType = 'tr', mixType='max'):
        """
        Args:
            dataRoot: the root directory to wsj mixture, the directory setup follows matlab scrip used by Isik, Y., Le Roux, J., Chen, Z., Watanabe, S., & Hershey, J. R. (2016). Single-channel multi-speaker separation using deep clustering. Proceedings of the Annual Conference of the International Speech Communication Association, INTERSPEECH, 08-12-Sept, 545–549. https://doi.org/10.21437/Interspeech.2016-1176. In short, the mixtures and sources are in dataRoot/{wav16k, wav8k}/{max, min}/{tr, cv, tt}/{mix, s1, s2, ..}. The configuration file generated by the matlab file is put under dataRoot/{wav16k, wav8k}/{max, min}/

            sampleRate: 16000 or 8000
            nMix: the number of source speakers
            soundLen: the sound length in seconds, used to make all sound the same length
            dataType: 'tr', 'cv', or 'tt'
            mixType: 'max' or 'min', see matlab script used to generate the mixture

        """

        self.sampleRate = sampleRate
        self.nMix = nMix
        self.soundLen = soundLen
        self.dataType = dataType
        self.mixType = mixType
        self.indexEnd = int(self.sampleRate * self.soundLen)


        # Construct data directory
        if sampleRate == 16000:
            self.dataPath = os.path.join(dataRoot, 'wav16k')
        elif sampleRate == 8000:
            self.dataPath = os.path.join(dataRoot, 'wav8k')
        else:
            raise(ValueError("Sample rate can only be 16k or 8k"))

        if mixType == 'max':
            self.dataPath = os.path.join(self.dataPath, 'max')
        elif mixType == 'min':
            self.dataPath = os.path.join(self.dataPath, 'max')
        else:
            raise(ValueError("Mix type can only be max or min"))

        self.mixInventoryFile = os.path.join(self.dataPath, 'mix_{:d}_spk_{}_{}_mix'.format(nMix, mixType, dataType))

        if dataType == 'tr':
            self.dataPath = os.path.join(self.dataPath, 'tr')
        elif dataType == 'cv':
            self.dataPath = os.path.join(self.dataPath, 'cv')
        elif dataType == 'tt':
            self.dataPath = os.path.join(self.dataPath, 'tt')
        else:
            raise(ValueError("Data type can only be tr, cv or tt"))


        with open(self.mixInventoryFile, 'r') as inventoryFile:
            self.inventory = inventoryFile.read().splitlines()

        self.mixDir = os.path.join(self.dataPath, 'mix')
        self.sDir = []
        for s in range(nMix):
            self.sDir.append(os.path.join(self.dataPath, 's{:d}'.format(s+1)))

        return


    def __len__(self):
        return len(self.inventory)


    def __getitem__(self, idx):
        mixName = self.inventory[idx]+'.wav'
        sr, mixture = wavfile.read(os.path.join(self.dataPath, 'mix', mixName))
        assert sr == self.sampleRate
        mixture = self._trimOrPadAudio(mixture)
        sources = np.zeros((mixture.shape[0], self.nMix))
        for s in range(self.nMix):
            sr , tempSource = wavfile.read(os.path.join(self.dataPath, 's{:d}'.format(s+1), mixName))
            assert sr == self.sampleRate
            sources[:, s] = self._trimOrPadAudio(tempSource)

        return mixture, sources

    def _trimOrPadAudio(self, sound):
        currentLength = sound.shape[0]
        if currentLength > self.indexEnd:
            soundAdjusted = np.copy(sound[0:self.indexEnd])
        else:
            soundAdjusted = np.concatenate((sound, np.zeros(self.indexEnd-currentLength)))

        return soundAdjusted

    # Test the __getitem__ method and play sounds
    def playIdx(self, idx):
        mixture, sources = self.__getitem__(idx)
        sound = np.ascontiguousarray(mixture, dtype=np.int16)
        play_obj = sa.play_buffer(sound, 1, 16//8, self.sampleRate)
        play_obj.wait_done()
        for s in range(self.nMix):
            sound = np.ascontiguousarray(sources[:, s], dtype=np.int16)
            play_obj = sa.play_buffer(sound, 1, 16//8, self.sampleRate)
            play_obj.wait_done()
        return


if __name__ == "__main__":

    sampleRate = 8000
    dataRoot = '/home/yangye/Lab/SpeechSeparation/WSJ0-2mix/'
    trDataset = AudioDataset(dataRoot, sampleRate=sampleRate, nMix=2, soundLen=5,  dataType='tr', mixType='max')

    # test __getitem__
    trDataset.playIdx(1000)

    # test dataloader
    train_dataloader = DataLoader(trDataset, batch_size=64, shuffle=True)
    mixture, sources = next(iter(train_dataloader))
    print(mixture.shape)
    print(sources.shape)

