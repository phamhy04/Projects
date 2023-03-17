
import numpy as np
import torch
import os
import sys
import time
import matplotlib.pyplot as plt
from model_exp import ConvAngular
from tqdm import tqdm
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

#   Define Computation device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device is not None:
    print("\nComputation device:", torch.cuda.get_device_name(torch.cuda.current_device()))


class My_Dataset(Dataset):
    def __init__(self, root_dir, file_name, transform = None):
        my_data = np.load(os.path.join(root_dir, file_name))
        self.X_data = my_data['x']
        self.Y_data = my_data['y']
        self.transform = transform
        
    def __len__(self):
        return len(self.X_data)
    
    def __getitem__(self, index):
        face = self.X_data[index]
        target = torch.tensor(self.Y_data[index])
        if self.transform:
            face = self.transform(face)
        return face, target



def get_embedding(model, test_loader):
    model = model.to(device).eval()
    full_embeds = []
    full_targets = []
    with torch.no_grad():
        for data, targets in test_loader:
            data = data.to(device)
            full_targets.append(targets.cpu().numpy())
            embeds = model(data, return_embedding = True)
            full_embeds.append(F.normalize(embeds).cpu().numpy())
    return np.concatenate(full_embeds), np.concatenate(full_targets)



def load_data(batch_size):
    root_dir = 'Datasets\\face-recognition-data'
    train_data = My_Dataset(root_dir = root_dir, file_name = 'Face_224x224.npz', transform = transforms.ToTensor()) 
    train_loader = DataLoader(dataset = train_data, batch_size = batch_size, shuffle = False)
    return train_loader



def train(train_loader, no_epochs, batch_size, learning_rate, loss_type):
    since = time.time()
    
    if loss_type == 'cosface':
        print('\n=>=>=> Training CosFace...')
    else:
        print('\n=>=>=> Training ArcFace...')
        
    model = ConvAngular(loss_type).to(device)
    optim = torch.optim.Adam(model.parameters(), lr = learning_rate)
    model_his = {'acc': [], 'loss': []}
    
    for epoch in range(no_epochs):
        loop = tqdm(train_loader, total = len(train_loader))
        for (data, targets) in loop:
            no_correct = 0
            data = data.to(device)
            targets = targets.to(device)
            
            _loss, y_pred = model(data, labels = targets)
            model_his['loss'].append(_loss)
            
            #   Calculate accuracy
            max_idx = torch.argmax(y_pred, dim = 1)
            no_correct += torch.sum(targets == max_idx) 
            _acc = no_correct / targets.shape[0]
            model_his['acc'].append(_acc)
            
            optim.zero_grad()
            _loss.backward()
            optim.step()
            
            loop.set_description(f"Epoch [{epoch+1}/{no_epochs}]")
            loop.set_postfix(loss = _loss.item(), acc = _acc.item())
            
    time_elapsed = time.time() - since
    print(f"\nTraining complete in: {time_elapsed // 60:.0f}m{time_elapsed % 60:.0f}s")
    return model.to(device = 'cpu'), model_his



def plot_result(embeddings, labels, loss_type):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(projection='3d')
    # Create a sphere
    r = 1
    phi, theta = np.mgrid[0.0 : np.pi : 100j, 0.0 : 2.0*np.pi : 100j]
    x = r*np.sin(phi) * np.cos(theta)
    y = r*np.sin(phi) * np.sin(theta)
    z = r*np.cos(phi)
    ax.plot_surface(x, y, z,  rstride=1, cstride=1, color='w', alpha=0.3, linewidth=0)
    ax.scatter(embeddings[:,0], embeddings[:,1], embeddings[:,2], c = labels, s = 20)

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    plt.title(loss_type, fontsize = 15, c = 'r');
    
    
def plot_loss_acc(cos_acc, cos_loss, arc_acc, arc_loss):
    fig = plt.figure(figsize = (15, 7))
    ax1, ax2  = fig.subplots(1, 2)
    
    ax1.plot(cos_acc, c = 'r', label = 'cos_acc')
    ax1.plot(arc_acc, c = 'g', label = 'arc_acc')
    ax1.set_title("Accuracy of CosFace and ArcFace", fontsize = 15)
    ax1.legend(borderpad = 2, shadow = True, fontsize = 12);
    
    ax2.plot(cos_loss, c = 'r', label = 'cos_loss')
    ax2.plot(arc_loss, c = 'g', label = 'arc_loss')
    ax2.set_title("Loss of CosFace and ArcFace", fontsize = 15)
    ax2.legend(borderpad = 2, shadow = True, fontsize = 12);
    


def main():
    #   Define some hyperparameters
    batch_size = 16
    learning_rate = 1e-3
    no_epochs = 10
    
    
    train_loader = load_data(batch_size)
    model, model_his = train(train_loader, no_epochs, batch_size, learning_rate, loss_type = 'arcface')
    
    root_dir = 'models'
    filename = 'model_vgg19.pth'
    #   Save model
    torch.save(model.state_dict(), os.path.join(root_dir, filename))
    #   Save model_history
    np.save('arcface_his.npy', model_his)
    
    # #   Load model
    # model = ConvAngular('arcface').to(device)
    # model.load_state_dict(torch.load(filename))
    # model.eval()
    
    # embeddings, labels = get_embedding(model, train_loader)
    
    # plot_result(embeddings, labels)
    
    
if __name__ == "__main__":
    main() 
