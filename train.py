from solver import Solver
from TasNet import TasNet
from data import AudioDataset
from torch.utils.data import DataLoader
import torch


if __name__ == '__main__':

    sampleRate = 8000
    dataRoot = '/home/yangye/Lab/SpeechSeparation/WSJ0_2mix_test/'
    trDataset = AudioDataset(dataRoot, sampleRate=sampleRate, nMix=2, soundLen=5,  dataType='tr', mixType='max')
    tr_dataloader = DataLoader(trDataset, batch_size=2, shuffle=True)

    cvDataset = AudioDataset(dataRoot, sampleRate=sampleRate, nMix=2, soundLen=5,  dataType='cv', mixType='max')

    cv_dataloader = DataLoader(cvDataset, batch_size=2, shuffle=True)

    N = 512
    L = int(sampleRate*5/1000)
    stride = L//2
    num_spk = 2
    hidden_size = 1000
    num_layers = 4
    bidirection = False
    tasnet = TasNet(N, L, stride, num_spk, hidden_size, num_layers, bidirection  )

    solver = Solver(tasnet, num_epoches = 2)
    solver.run(tr_dataloader, cv_dataloader)






