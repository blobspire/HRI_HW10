import torch
import pickle
from models import Autoencoder
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# import dataset for training
class MyData(Dataset):

    def __init__(self, loadname):
        self.data = pickle.load(open(loadname, "rb"))
        self.data = torch.FloatTensor(self.data)
        print("imported dataset of length:", len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        return self.data[idx]


# train model
def train_model(loadname):

    # training parameters
    print("[-] training autoencoder")
    EPOCH = 10000
    LR = 0.0001

    # initialize model and optimizer
    model = Autoencoder(state_dim=15, hidden_dim=256, action_dim=9, latent_dim=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # initialize dataset
    print("[-] loading data: " + loadname)
    train_data = MyData(loadname)
    BATCH_SIZE = int(len(train_data) / 10.)
    print("my batch size is:", BATCH_SIZE)
    train_set = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

    # main training loop
    for epoch in tqdm(range(EPOCH+1)):
        for batch, x in enumerate(train_set):
        
            # collect the demonstrated states and actions
            states = x[:, 0:15]
            actions = x[:, 15:24] # Actions are 9 dims. 7 joints + 2 pybullet things
            actions_hat = model(states, actions) # Add bottleneck in middle

            # compute the loss between actual and predicted
            loss = model.mse_loss(actions, actions_hat)
                 
            # update model parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        if epoch % 10 == 0:
            # print(epoch, loss.item())
            torch.save(model.state_dict(), "model_weights_bidirectional_simple")

# train models
if __name__ == "__main__":
    train_model("dataset_bidirectional_simple.pkl")