import torch 
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class SparseAutoencoder(nn.Module):
    def __init__(self,input_dim,hidden_dim,sparsity_weight=0.01):
        super(SparseAutoencoder,self).__init__()
        self.encoder=nn.Sequential(nn.Linear(input_dim,hidden_dim),nn.ReLU())
        self.decoder=nn.Sequential(nn.Linear(hidden_dim,input_dim),nn.Sigmoid())
        self.sparsity_weight=sparsity_weight
    def forward(self,x):
        encoded=self.encoder(x)
        decoded=self.decoder(encoded)
        return encoded,decoded
    def sparsity_loss(self,encoded):
        return self.sparsity_weight*torch.mean(torch.abs(encoded)) #L1 regulariztaion for sparsity
    
transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])
train_dataset=datasets.MNIST(root="./data",train=True,download=True,transform=transform)
test_dataset=datasets.MNIST(root="./data",train=False,download=True,transform=transform)
train_loader=DataLoader(train_dataset,batch_size=128,shuffle=True)
test_loader=DataLoader(test_dataset,batch_size=128,shuffle=True)

def train(model,train_loader,optimizer,criterion,device,epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss=0
        for data,_ in train_loader:
            data=data.view(data.size(0),-1).to(device)
            #Forward
            encoded,decoded=model(data)
            reconstruction_loss=criterion(decoded,data)
            sparsity_loss=model.sparsity_loss(encoded)
            loss=reconstruction_loss+sparsity_loss
            #Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss+=loss.item()
        print(f"Epoch [{epoch+1}/{epochs}],Loss:{total_loss/len(train_loader):.4f}")

def evaluate(model,test_loader,criterion,device):
    model.eval()
    total_loss=0
    with torch.no_grad():
        for data,_ in test_loader:
            data=data.view(data.size(0),-1).to(device)
            encoded,decoded=model(data)
            reconstruction_loss=criterion(decoded,data)
            sparsity_loss=model.sparsity_loss(encoded)
            loss=reconstruction_loss+sparsity_loss
            total_loss+=loss.item()
    print(f"Test loss:{total_loss/len(test_loader):.4f}")

input_dim=28*28
hidden_dim=128
sparsity_weight=0.01
learning_rate=0.001
epochs=10

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=SparseAutoencoder(input_dim=input_dim,hidden_dim=hidden_dim,sparsity_weight=sparsity_weight).to(device)
criterion=nn.MSELoss()
optimizer=optim.Adam(model.parameters(),lr=learning_rate)
train(model=model,train_loader=train_loader,optimizer=optimizer,criterion=criterion,device=device,epochs=epochs)
evaluate(model=model,test_loader=test_loader,criterion=criterion,device=device)

def visualize_reconstruction(model,test_loader,device,num_images=5):
    model.eval()
    with torch.no_grad():
        for data,_ in test_loader:
            data = data.view(data.size(0), -1).to(device)
            _, decoded = model(data)
            data = data.cpu().numpy()
            decoded = decoded.cpu().numpy()
            break
    plt.figure(figsize=(10, 4))
    for i in range(num_images):
        plt.subplot(2, num_images, i+1)
        plt.imshow(data[i].reshape(28, 28), cmap='gray')
        plt.title("Original")
        plt.axis('off')
        plt.subplot(2, num_images, i+1+num_images)
        plt.imshow(decoded[i].reshape(28, 28), cmap='gray')
        plt.title("Reconstructed")
        plt.axis('off')
    plt.show()
visualize_reconstruction(model=model,test_loader=test_loader,device=device)