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
parser.add_argument("--nlayers", type=int, default=4, help="Number of layers for EARLIEST's RNN.")
parser.add_argument("--rnn_cell", type=str, default="LSTM", help="Type of RNN to use in EARLIEST. Available: GRU, LSTM")
parser.add_argument("--lam", type=float, default=0.0, help="Penalty of waiting. This controls the emphasis on earliness: Larger values lead to earlier predictions.")

# Training hyperparameters
parser.add_argument("--batch_size", type=int, default=8, help="Batch size.")
parser.add_argument("--nepochs", type=int, default=100, help="Number of epochs.")
parser.add_argument("--learning_rate", type=float, default="0.0001", help="Learning rate.")
parser.add_argument("--model_save_path", type=str, default="./saved_models/", help="Where to save the model once it is trained.")
parser.add_argument("--random_seed", type=int, default="69", help="Set the random seed.")

args = parser.parse_args()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    model_save_path = args.model_save_path
    utils.makedirs(model_save_path)
    exponentials = utils.exponentialDecay(args.nepochs)

    # Data setup
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.join(script_dir, "..", "..",  "BugSenseData", "Usable")
    if args.dataset == "bugsense":
        train_ds = BugSenseData(root_dir, partition="train", sequencelength=args.ntimesteps)
        test_ds = BugSenseData(root_dir, partition="valid", sequencelength=args.ntimesteps)
    
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

for epoch in range(args.nepochs):
    model._REWARDS = 0
    model._r_sums = np.zeros(args.ntimesteps).reshape(1, -1)
    model._r_counts = np.zeros(args.ntimesteps).reshape(1, -1)
    model._epsilon = exponentials[epoch]
    loss_sum = 0
    losses = []
    training_loss = []
    training_locations = [] 
    training_predictions = []

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
        losses.append(loss)
        loss.backward()
        loss_sum += loss.item()
        optimizer.step()

    training_locations = torch.stack(training_locations).cpu().numpy().reshape(-1, 1)
    # Log training loss to TensorBoard
    writer.add_scalar('Loss/Train', loss_sum / len(train_loader), epoch)
    mean_prop_used = np.mean(training_locations)
    print(f"Epoch [{epoch+1}/{args.nepochs}], Loss: {loss_sum/len(train_loader):.4f}")
    print(f"Mean proportion used: {np.round(100. * mean_prop_used, 3)}%")
    training_loss.append(np.round(loss_sum / len(train_loader), 3))
    scheduler.step()

    # --- Run model on validation data --- (Run validation after each epoch)
    validation_locations = []
    validation_predictions = []
    validation_labels = []
    model.eval()  # Switch model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation for validation
        for i, (X, y) in enumerate(tqdm(validation_loader, desc="Validation")):
            X, y = X.to(device), y.to(device)  # Move data to GPU
            X = torch.transpose(X, 0, 1)

            # --- Forward pass ---
            logits, halting_points = model(X, test=True)
            _, predictions = torch.max(torch.softmax(logits, dim=1), dim=1)

            validation_locations.append(halting_points)
            validation_predictions.append(predictions)
            validation_labels.append(y)

        # Convert lists to tensors and move to CPU for further processing
        validation_predictions = torch.stack(validation_predictions).cpu().numpy().reshape(-1, 1)
        validation_labels = torch.stack(validation_labels).cpu().numpy().reshape(-1, 1)
        validation_locations = torch.stack(validation_locations).cpu().numpy().reshape(-1, 1)

        # Log validation accuracy and mean proportion used to TensorBoard
        validation_accuracy = accuracy_score(validation_labels, validation_predictions)
        mean_proportion_used = np.mean(validation_locations)
        writer.add_scalar('Accuracy/Validation', validation_accuracy, epoch)
        writer.add_scalar('Proportion/Validation', mean_proportion_used, epoch)

        print(f"Validation Accuracy: {np.round(validation_accuracy, 3)}")
        print(f"Mean proportion used: {np.round(100. * mean_proportion_used, 3)}%")

    model.train()  # Switch back to training mode

# --- Save model ---
torch.save(model.state_dict(), os.path.join(model_save_path, "model.pt"))

# Close TensorBoard writer
writer.close()
