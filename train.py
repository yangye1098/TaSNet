from solver import Solver
from TasNet import TasNet
from data import AudioDataset
from torch.utils.data import DataLoader
import torch
import logging

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler('tasnet.log')
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter(
                "%(asctime)s [%(pathname)s:%(lineno)s - %(levelname)s ] %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
   
    sampleRate = 8000
    batch_size = 32
    num_spk = 2
    dataRoot = '/pub/yey27/SpeechSeparation/WSJ0-2mix/'
    trDataset = AudioDataset(dataRoot, sampleRate=sampleRate, nMix=num_spk, soundLen=5,  dataType='tr', mixType='max')
    tr_dataloader = DataLoader(trDataset, batch_size=batch_size, shuffle=True)

    cvDataset = AudioDataset(dataRoot, sampleRate=sampleRate, nMix=num_spk, soundLen=5,  dataType='cv', mixType='max')
    cv_dataloader = DataLoader(cvDataset, batch_size=batch_size, shuffle=True)

    N = 500
    L = int(sampleRate*5/1000)
    stride = L//2
    hidden_size = 1000
    num_layers = 4
    bidirection = False
    tasnet = TasNet(N, L, stride, num_spk, hidden_size, num_layers, bidirection )

    logger.info(f"Start TasNet optimization. Parameters: Batch Size: {batch_size}, "
            f"Number of Speaker:{num_spk}, Sample Rate:{sampleRate}, N:{N}, L:{L}, "
            f"Stride:{stride}, Hidden Size:{hidden_size}, Number of LSTM Layers:{num_layers}, "
            f"Bidirectional: {bidirection}")
    if torch.cuda.is_available():
        tasnet.to(device)
        logger.info("Train on gpu")
    solver = Solver(tasnet, num_epoches = 20, logger = logger)
    solver.run(tr_dataloader, cv_dataloader)






