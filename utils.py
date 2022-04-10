from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import os
from tqdm import tqdm

import time

def path2pair(path: str, res: tuple) -> tuple:
    image = Image.open(path).convert("RGB").resize(res)
    x = transforms.ToTensor()(image)

    y = float(path.split("_")[0].split("/")[-1])

    return (x, y)


def dir2dataset(dir, res) -> tuple:

    paths = os.listdir(dir)
    paths = [f"{dir}/{i}" for i in paths]

    X = torch.zeros((len(paths), 3, *res)).type(torch.float16)
    Y = torch.zeros((len(paths))).type(torch.float16)

    for i in tqdm(range(len(paths))):
        pair = path2pair(paths[i], res)
        X[i], Y[i] = pair

    Y /= torch.max(Y)

    return (X, Y)



class AgeDataset(Dataset):
  
    def __init__(self, x, y):
        self.x = x
        self.y = y
      
    
    def __len__(self):
      return len(self.x)


    def __getitem__(self, idx):
      return (self.x[idx].type(torch.float32), self.y[idx].type(torch.float32))


if __name__ == '__main__':
    dir = "./dataset"

    x, y = dir2dataset(dir, (128, 128))
    torch.save(x, f"./dataset/x.pt")
    torch.save(y, f"./dataset/y.pt")

    # x = torch.load("./dataset/x.pt")
    # y = torch.load("./dataset/y.pt")

    # for i in range(0, len(x), 1000):
    #     img = transforms.ToPILImage()(x[i])
    #     label = y[i]

    #     img.show()
    #     print(116 * label)

    #     time.sleep(2)
