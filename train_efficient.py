from torchvision.models import efficientnet_v2_s, efficientnet_b0, EfficientNet_B0_Weights
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from data.dataset_maker import CustomImageDataset
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import wandb
import argparse


# Training function.
def train(model, trainloader, optimizer, criterion):
    model.train()
    print('Training')
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        counter += 1
        image, labels = data
        image = image.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # Forward pass.
        outputs = model(image)
        # Calculate the loss.
        loss = criterion(outputs, labels)
        train_running_loss += loss.item()
        # Calculate the accuracy.
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == labels).sum().item()
        # Backpropagation
        loss.backward()
        # Update the weights.
        optimizer.step()
        wandb.log({"train_loss": loss.item(), "train_accuracy": float((preds == labels).sum().item()) / len(preds)})
    
    # Loss and accuracy for the complete epoch.
    epoch_loss = train_running_loss / counter
    epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))
    return epoch_loss, epoch_acc

# Validation function.
def validate(model, testloader, criterion):
    model.eval()
    print('Validation')
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1
            
            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            # Forward pass.
            outputs = model(image)
            # Calculate the loss.
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()
            # Calculate the accuracy.
            _, preds = torch.max(outputs.data, 1)
            valid_running_correct += (preds == labels).sum().item()
        
    # Loss and accuracy for the complete epoch.
    epoch_loss = valid_running_loss / counter
    epoch_acc = 100. * (valid_running_correct / len(testloader.dataset))
    return epoch_loss, epoch_acc


def run_exp(FOLD):
        WANDB_START_METHOD = "thread"
        wandb.init(project="CVSA_FINAL", entity="tandl", name=f"EFFICIENT_SPLIT_{FOLD}", save_code=True)
        # Load the training and validation datasets.
        # model = efficientnet_v2_s(weights='DEFAULT')
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(1280, 18, bias=True)

        dataset_train = CustomImageDataset(f'fold_indexes/fold{FOLD}_train.csv', n_samples=48000)
        dataset_valid = CustomImageDataset(f'fold_indexes/fold{FOLD}_val.csv', n_samples=10000)

        train_loader = DataLoader(dataset_train, batch_size=16, shuffle=True)
        valid_loader = DataLoader(dataset_valid, batch_size=16, shuffle=False)

        print(f"[INFO]: Number of training images: {len(dataset_train)}")
        print(f"[INFO]: Number of validation images: {len(dataset_valid)}")
        # Learning_parameters. 
        model = model.to(device)
        
        # Optimizer.
        optimizer = optim.Adam(model.parameters(), lr=lr)
        # Loss function.
        criterion = nn.CrossEntropyLoss()
        # Lists to keep track of losses and accuracies.
        train_loss, valid_loss = [], []
        train_acc, valid_acc = [], []
        # Start the training.
        best_val_accuracy = 0
        for epoch in range(epochs):
            print(f"[INFO]: Epoch {epoch+1} of {epochs}")
            train_epoch_loss, train_epoch_acc = train(model, train_loader, 
                                                    optimizer, criterion)
            valid_epoch_loss, valid_epoch_acc = validate(model, valid_loader,  
                                                        criterion)
            train_loss.append(train_epoch_loss)
            valid_loss.append(valid_epoch_loss)
            train_acc.append(train_epoch_acc)
            valid_acc.append(valid_epoch_acc)
            wandb.log({"Training Loss": train_epoch_loss, 
                        "Training Accur.": train_epoch_acc, 
                        "Valid. Loss": valid_epoch_loss, 
                        "Valid. Accur.": valid_epoch_acc})

            print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
            print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
            print('-'*50)
            if valid_epoch_acc > best_val_accuracy:
                best_val_accuracy = valid_epoch_acc
                # Save the trained model weights.
                torch.save(model.state_dict(), f'efficient_models/fold{FOLD}_model.pkl')
                torch.save(optimizer.state_dict(), f'efficient_models/fold{FOLD}_optimizer.pkl')
                # Save the loss and accuracy plots.
        print('TRAINING COMPLETE')


lr = 0.001
epochs = 1
device = ('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Computation device: {device}")
print(f"Learning rate: {lr}")
print(f"Epochs to train for: {epochs}\n")
parser = argparse.ArgumentParser()
parser.add_argument('--FOLD', default='0')

args = parser.parse_args()
run_exp(args.FOLD)