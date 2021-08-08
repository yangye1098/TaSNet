import time
import torch
import logging
from TasNet import TasNet
import itertools
from pathlib import Path
from torch.autograd import Variable
from utils import prepare_signal

device = torch.device("cpu")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
fh = logging.FileHandler('tasnet.log')
fh.setLevel(logging.INFO)
formatter = logging.Formatter(
            "%(asctime)s [%(pathname)s:%(lineno)s - %(levelname)s ] %(message)s")
fh.setFormatter(formatter)
logger.addHandler(fh)




class Solver(object):
    def __init__(self,
                 tasnet,
                 num_spk = 2,
                 num_epoches = 20,
                 lr = 3e-4,
                 optimizer = 'adam',
                 save_folder = './savedModel'):
        self.tasnet = tasnet
        self.num_spk = num_spk
        #self.currentEpoch = 0
        self.num_epoches = num_epoches

        self.lr = lr
        if optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.tasnet.parameters(),
                                              lr = self.lr)
        else:
            raise(NotImplementedError)

        # halve the learning rate after 3 non-improving epoch
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                            self.optimizer, 'min', factor=0.5, patience=3)

        self.save_folder = Path(save_folder)
        self.save_folder.mkdir(parents=True, exist_ok=True)

        # Record the loss
        self.tr_loss = torch.Tensor(self.num_epoches)
        self.cv_loss = torch.Tensor(self.num_epoches)




        return

    def SISNR (self, s_hat, s):
        """
        Calculate SISNR for one source
        Args:
            s_hat: [B, *, T]
            s: [B, *, T] the true sources
        Returns:
            SI-SNR: [B, *, 1]

        """
        # normalize to zero mean
        s_hat = s_hat - torch.mean(s_hat, 2, keepdim=True) #[B, 1, T]
        s = s - torch.mean(s, 2, keepdim=True)  #[B, 1, T]
        # <s, s_hat>s/||s||^2
        s_shat = torch.sum(s_hat * s, dim=2, keepdim=True) #[B, 1, 1]
        s_2 =  torch.sum(s**2, dim=2, keepdim=True) # [B, 1, T]
        s_target = s_shat * s / s_2 # [B, 1, T]

        # e_noise = s_hat - s_target
        e_noise = s_hat - s_target # [B, 1, T]
        return 10*torch.log10(torch.sum(s_target**2, dim=2, keepdim = True)\
                              /torch.sum(e_noise**2, dim=2, keepdim = True))

    def sumSISNR(self, s_hat, s):
        """
        Calculate summed SISNR for all sources
        Args:
            s_hat: [B, *, T]
            s: [B, *, T] the true sources
        Returns:
            summed SI-SNR across targets: [B]

        """

        return torch.squeeze(torch.sum(self.SISNR(s_hat, s), dim=1))

    def loss(self, output, targets):
        """
        PIT loss taking care of permutation problem. Loss is negative SI-SNR
        Args:
            output: [B, C, T]
            targets: [B, C, T]
        returns:
            loss of the current batch
        """

        # sum negative SISNR of all sources, and then choose the minimum
        # one as the loss of this batch batch_size = output.size(0)

        all_perms = list(itertools.permutations(range(self.num_spk)))
        temp_loss = torch.zeros((len(all_perms), output.size(0)))
        for i, permutation in enumerate(all_perms):
            temp_loss[i, :] = -self.sumSISNR(output[:, permutation, :], targets)

        loss, indics = torch.min(temp_loss, dim=0)
        return torch.mean(loss)

    def train(self, dataloader):
        """
        Train one epoch

        Return:
            Average training loss
        """
        self.tasnet.train()

        num_batches = len(dataloader)
        total_loss = 0

        for batch_idx, (mixture, targets) in enumerate(dataloader):
            self.optimizer.zero_grad()

            if torch.cuda.is_available():
                mixture = mixture.to_cuda()
                targets = targets.to_cuda()

            mixture = mixture.type(torch.float)
            targets = targets.type(torch.float)
            # prepare signal

            mixture = prepare_signal(mixture, self.tasnet.L, self.tasnet.stride )
            targets = prepare_signal(targets, self.tasnet.L, self.tasnet.stride )


            # get output
            output = self.tasnet(mixture)
            # calculate average loss of current batch
            batch_loss = self.loss(output, targets)

            total_loss += batch_loss.item()
            # update
            batch_loss.backward()
            self.optimizer.step()

            if batch_idx % 50 == 0:
                # log progress
                logger.info(f"loss: {batch_loss.item():>7f}  [{batch_idx:>5d}/{num_batches:>5d}]")
        return total_loss/num_batches

    def validate(self, dataloader):
        """
        Validate current model using validation dataset

        """
        self.tasnet.eval()

        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        validation_loss = 0

        with torch.no_grad():
            for mixture, targets in dataloader:

                if torch.cuda.is_available():
                    mixture = mixture.to_cuda()
                    targets = targets.to_cuda()

                mixture = mixture.type(torch.float)
                targets = targets.type(torch.float)
                # prepare signal

                mixture = prepare_signal(mixture, self.tasnet.L, self.tasnet.stride )
                targets = prepare_signal(targets, self.tasnet.L, self.tasnet.stride )
                output = self.tasnet(mixture)
                validation_loss += self.loss(output, targets).item()

        validation_loss /= num_batches

        return validation_loss / num_batches

    def run(self, trainLoader, cvLoader):
        """
        Run train and validate loop for num_epoches time
        log the progress
        save the model at checkpoints in case of system fault
        """
        no_improvement_counter = 0
        for epoch in range(self.num_epoches):
            logger.info(f'Epoch {epoch+1}')
            start = time.time()
            logger.info('Start training')
            avg_train_loss = self.train(trainLoader)
            self.tr_loss[epoch] = avg_train_loss
            logger.info(f"Finished. Taining Time: {time.time()-start:.1f}")
            logger.info(f"Average Train Loss: {avg_train_loss:>.3f}")
            logger.info('Start Validating')
            avg_cv_loss = self.validate(cvLoader)
            self.cv_loss[epoch] = avg_cv_loss
            logger.info(f"Finished. Average Validation Loss: {avg_cv_loss:>.3f}")

            # save model
            file_path = self.save_folder / f'epoch{epoch+1}'
            torch.save(self.tasnet.state_dict(), file_path)
            logger.info(f'Saving model to {file_path}')

            # Halve the learning rate if no improving by updating scheduler
            self.scheduler.step(avg_cv_loss)

            # Check early stopping last 10 epoch doesn't have improvement
            if epoch > 0:
                if self.cv_loss[epoch] > self.cv_loss(epoch-1):
                    no_improvement_counter += 1
                else:
                    no_improvement_counter = 0

            if no_improvement_counter > 10:
                logger.info(f'Early Stop at Epoch {epoch+1}')
                break

        torch.save(self.tr_loss, self.save_folder/'tr_loss.pt')
        torch.save(self.cv_loss, self.save_folder/'cv_loss.pt')
        return

    def resume():
        """
        Start the training from certain epoch again
        log the progress
        save the model at checkpoints in case of system fault
        """
        return



if __name__ == '__main__':
    N = int(512)
    L = int(32)
    stride = L//2
    num_spk = 2
    hidden_size = 1000
    num_layers = 4

    tasnet = TasNet(N, L, stride, num_spk, hidden_size, num_layers)
    solver = Solver(tasnet)


    a = torch.rand([2,2,4])
    b = torch.rand([2,2,4])
    print(a)
    print(b)
    print(solver.loss(a, b))

