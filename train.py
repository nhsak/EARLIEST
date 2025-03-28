import numpy as np
import argparse
import torch
from model import EARLIEST
from dataset import SyntheticTimeSeries
from bugsense_data import BugSenseData
from torch.utils.data.sampler import SubsetRandomSampler
import utils
from sklearn.metrics import accuracy_score
import os
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm  # Import tqdm for the progress bar

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Dataset hyperparameters
parser.add_argument("--dataset", type=str, default="bugsense", help="Dataset to load. Available: Synthetic")
parser.add_argument("--ntimesteps", type=int, default=80, help="Synthetic dataset can control the number of timesteps")
parser.add_argument("--nseries", type=int, default=180, help="Synthetic dataset can control the number of time series")

# Model hyperparameters
parser.add_argument("--nhid", type=int, default=64, help="Number of dimensions of the hidden state of EARLIEST")
parser.add_argument("--nlayers", type=int, default=2, help="Number of layers for EARLIEST's RNN.")
parser.add_argument("--rnn_cell", type=str, default="LSTM", help="Type of RNN to use in EARLIEST. Available: GRU, LSTM")
parser.add_argument("--lam", type=float, default=0.0, help="Penalty of waiting. This controls the emphasis on earliness: Larger values lead to earlier predictions.")

# Training hyperparameters
parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
parser.add_argument("--nepochs", type=int, default=100, help="Number of epochs.")
parser.add_argument("--learning_rate", type=float, default="0.001", help="Learning rate.")
parser.add_argument("--model_save_path", type=str, default="./snapshots/earliest.pth", help="Where to save the model once it is trained.")
parser.add_argument("--random_seed", type=int, default="42", help="Set the random seed.")

args = parser.parse_args()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    
    os.makedirs(os.path.dirname(args.model_save_path), exist_ok=True)
    exponentials = utils.exponentialDecay(args.nepochs)

    # Data setup
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.join(script_dir, "..", "..",  "BugSenseData", "Usable", "train")
    if args.dataset == "bugsense":
        train_ds = BugSenseData(root_dir, partition="train", sequencelength=args.ntimesteps, split_ratio=(0.8, 0.2, 0))
        test_ds = BugSenseData(root_dir, partition="valid", sequencelength=args.ntimesteps, split_ratio=(0.8, 0.2, 0))
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size)
    validation_loader = DataLoader(test_ds, batch_size=args.batch_size, drop_last=True)

    print(len(train_loader))
    print(len(validation_loader))

    ninp = args.nhid
    nclasses = 6

    # Model setup
    model = EARLIEST(ninp=ninp, nclasses=nclasses, args=args).to(device)  # Move model to GPU
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    # TensorBoard setup
    writer = SummaryWriter(log_dir='./runs')

    # --- Training Loop ---

    validation_accuracies = []
    mean_losses = []
    all_losses = []
    training_locations = [] 
    training_predictions = []
    # --- training ---
    
    for epoch in range(args.nepochs):
        model.train()  # Switch model to training mode
        model._REWARDS = 0
        model._r_sums = np.zeros(args.ntimesteps).reshape(1, -1)
        model._r_counts = np.zeros(args.ntimesteps).reshape(1, -1)
        model._epsilon = exponentials[epoch]
        loss_sum = 0
        
        # tqdm for progress bar
        for i, (X, y) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.nepochs}")):
            X, y = X.to(device), y.to(device)  # Move data to GPU
            X = torch.transpose(X, 0, 1)
            

            # --- Forward pass ---
            logits, halting_points = model(X, epoch)
            _, predictions = torch.max(torch.softmax(logits, dim=1), dim=1)

            training_locations.append(halting_points)
            training_predictions.append(predictions)

            # --- Compute gradients and update weights ---
            optimizer.zero_grad()
            loss = model.computeLoss(logits, y)
            loss.backward()
            loss_sum += loss.item()
            optimizer.step()
            all_losses.append(loss.cpu().detach().numpy())

        # # Log training loss to TensorBoard
        writer.add_scalar('Loss/Train', loss_sum / len(train_loader), epoch)
        print(f"Epoch [{epoch+1}/{args.nepochs}], Loss: {loss_sum/len(train_loader):.4f}")
        mean_losses.append(loss_sum / len(train_loader))
        scheduler.step()

    # --- Run model on validation data --- (Run validation after each epoch)
    
        validation_locations = []
        validation_predictions = []
        validation_labels = []
        with torch.no_grad():  # Disable gradient calculation for validation
            for i, (X, y) in enumerate(tqdm(validation_loader, desc="Validation")):
                model.eval()  # Switch model to evaluation mode
                X, y = X.to(device), y.to(device)  # Move data to GPU
                X = torch.transpose(X, 0, 1)

                # --- Forward pass ---
                logits, halting_points = model(X, test=True)
                _, predictions = torch.max(torch.softmax(logits, dim=1), dim=1)

                validation_locations.append(halting_points.cpu().detach().numpy())
                validation_predictions.append(predictions.cpu().detach().numpy())
                validation_labels.append(y.cpu().detach().numpy())
            
            

            # Convert lists to tensors and move to CPU for further processing
            validation_predictions = np.vstack(validation_predictions).reshape(-1, 1)
            validation_labels = np.vstack(validation_labels).reshape(-1, 1)
            validation_locations = np.vstack(validation_locations).reshape(-1, 1)

            # Log validation accuracy and mean proportion used to TensorBoard
            validation_accuracy = accuracy_score(validation_labels, validation_predictions)
            validation_accuracies.append(validation_accuracy)
            
            print(f"Saving model with validation accuracy: {np.round(validation_accuracy, 3)}")
            torch.save(model.state_dict(), args.model_save_path)
            mean_proportion_used = np.mean(validation_locations)
            writer.add_scalar('Accuracy/Validation', validation_accuracy, epoch)
            writer.add_scalar('Proportion/Validation', mean_proportion_used, epoch)

            print(f"Validation Accuracy: {np.round(100. * validation_accuracy, 3)}%")
            print(f"Mean proportion used: {np.round(100. * mean_proportion_used, 3)}%")


# Close TensorBoard writer
writer.close()


