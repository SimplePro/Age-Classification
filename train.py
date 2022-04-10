from torch.utils.data import DataLoader
from torch.nn import MSELoss
from torch.optim import Adam
import torch
from tqdm import tqdm

import matplotlib.pyplot as plt

from.models import AgeClassifier
from utils import AgeDataset


class Trainer:

    def __init__(self, lr=0.002, betas=(0.5, 0.999), batch_size=64):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AgeClassifier().to(self.device)

        self.optim = Adam(self.model.parameters(), lr=lr, betas=betas)
        self.criterion = MSELoss()

        self.x = torch.load("./dataset/x.pt")
        self.y = torch.load("./dataset/y.pt").unsqueeze(1) # shape: (N) -> (N, 1)

        trainset = AgeDataset(self.x[:-3000], self.y[:-3000])
        validset = AgeDataset(self.x[-3000:], self.y[-3000:])

        self.trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        self.validloader = DataLoader(validset, batch_size=batch_size, shuffle=True)

        self.best_state_dict = None
        self.min_valid_loss = 1


    def train(self):
        self.model.train()

        train_loss = []

        for x, y in tqdm(self.trainloader):

            x, y = x.to(self.device), y.to(self.device)

            pred = self.model(x)
            loss = self.criterion(pred, y)

            self.optim.zero_grad()
            # loss1.backward(retain_graph=True) # pytorch는 requires_grad=True인 텐서들의 모든 연산을 그래프 형태로 저장한다.
            # loss2.backward() # 하지만 backward를 하면 그 그래프 정보를 제거하는데, retain_graph=True는 이를 제거하지 않는다.
            loss.backward()
            self.optim.step()

            train_loss.append(loss.item())

        return train_loss


    def valid(self):
        self.model.eval()

        valid_loss = []

        for x, y in tqdm(self.validloader):
            x, y = x.to(self.device), y.to(self.device)

            pred = self.model(x)
            loss = self.criterion(pred, y)

            valid_loss.append(loss.item())

        return valid_loss

      
    def save_history(self, train_history=[], valid_history=[]):
        fig, axs = plt.subplots(2)
        axs[0].plot(range(len(train_history)), train_history, label="train_loss")
        axs[0].legend()
        
        axs[1].plot(range(len(valid_history)), valid_history, label="valid_loss")
        axs[1].legend()

        plt.savefig("./loss_history.png")


    def run(self, epochs=30, save_path=None, save_loss_history=True):
        train_history = []
        valid_history = []

        for epoch in range(epochs):
            print("-" * 100, f"\nEPOCH: {epoch+1}/{epochs}\n")

            print("TRAIN", end=" ")
            train_loss = self.train()
            train_history.extend(train_loss)
            mean_train_loss = sum(train_loss) / len(train_loss)
            print(f"- loss: {mean_train_loss}\n")

            print("VALID", end=" ")
            valid_loss = self.valid()
            valid_history.extend(valid_loss)
            mean_valid_loss = sum(valid_loss) / len(valid_loss)
            print(f"- loss: {mean_valid_loss}\n")

            if mean_valid_loss < self.min_valid_loss:
                self.min_valid_loss = mean_valid_loss
                self.best_state_dict = self.model.state_dict()

            if save_loss_history: self.save_history(train_history, valid_history)


        if save_path != None: torch.save(self.best_state_dict, save_path)

        return train_history, valid_history


if __name__ == '__main__':

    trainer = Trainer(lr=0.002, betas=(0.5, 0.999), batch_size=64)

    train_history, valid_history = trainer.run(epochs=30, save_path="./model.pt", save_loss_history=True)
