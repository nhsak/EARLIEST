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

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Dataset hyperparameters
parser.add_argument("--dataset", type=str, default= "bugsense", help="Dataset to load. Available: Synthetic")
parser.add_argument("--ntimesteps", type=int, default=80, help="Synthetic dataset can control the number of timesteps")
parser.add_argument("--nseries", type=int, default=180, help="Synthetic dataset can control the number of time series")

# Model hyperparameters
parser.add_argument("--nhid", type=int, default=64, help="Number of dimensions of the hidden state of EARLIEST")
parser.add_argument("--nlayers", type=int, default=1, help="Number of layers for EARLIEST's RNN.")
parser.add_argument("--rnn_cell", type=str, default="LSTM", help="Type of RNN to use in EARLIEST. Available: GRU, LSTM")
parser.add_argument("--lam", type=float, default=0.0, help="Penalty of waiting. This controls the emphasis on earliness: Larger values lead to earlier predictions.")

# Training hyperparameters
parser.add_argument("--batch_size", type=int, default=10, help="Batch size.")
parser.add_argument("--nepochs", type=int, default=100, help="Number of epochs.")
parser.add_argument("--learning_rate", type=float, default="0.001", help="Learning rate.")
parser.add_argument("--model_save_path", type=str, default="./saved_models/", help="Where to save the model once it is trained.")
parser.add_argument("--random_seed", type=int, default="42", help="Set the random seed.")

args = parser.parse_args()

if __name__ == "__main__":
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    model_save_path = args.model_save_path
    utils.makedirs(model_save_path)
    exponentials = utils.exponentialDecay(args.nepochs)


    ### CHANGED ### 
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.join(script_dir, "..", "..",  "BugSenseData", "Usable")
    if args.dataset == "bugsense":
        train_ds = BugSenseData(root_dir, partition="train", sequencelength=args.ntimesteps)
        test_ds = BugSenseData(root_dir, partition="valid", sequencelength=args.ntimesteps)
    
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size)
    validation_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size)

    print(len(train_loader))
    print(len(validation_loader))

    ninp = args.nhid
    nclasses = 6
    ###############


    # if args.dataset == "synthetic":
    #     data = SyntheticTimeSeries(args)
    # train_ix, validation_ix, test_ix = utils.splitTrainingData(data.nseries)

    # train_sampler = SubsetRandomSampler(train_ix)
    # validation_sampler = SubsetRandomSampler(validation_ix)
    # test_sampler = SubsetRandomSampler(test_ix)

    # train_loader = torch.utils.data.DataLoader(dataset=data,
    #                                            batch_size=args.batch_size,
    #                                            sampler=train_sampler,
    #                                            drop_last=True)
    # validation_loader = torch.utils.data.DataLoader(dataset=data,
    #                                                 batch_size=args.batch_size,
    #                                                 sampler=validation_sampler,
    #                                                 drop_last=True)

    

    model = EARLIEST(ninp=ninp, nclasses=nclasses, args=args) #nhid=HIDDEN_DIMENSION, rnn_type=CELL_TYPE, nlayers=N_LAYERS, lam=LAMBDA)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    # --- training ---
    training_loss = []
    training_locations = []
    training_predictions = []
    for epoch in range(args.nepochs):
        model._REWARDS = 0
        model._r_sums = np.zeros(args.ntimesteps).reshape(1, -1)
        model._r_counts = np.zeros(args.ntimesteps).reshape(1, -1)
        model._epsilon = exponentials[epoch]
        loss_sum = 0
        losses = []
        for i, (X, y) in enumerate(train_loader):
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

           
        print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, args.nepochs, loss_sum/len(train_loader)))
        print(losses)
        training_loss.append(np.round(loss_sum/len(train_loader), 3))
        scheduler.step()

    # --- Run model on validation data ---
    validation_locations = []
    validation_predictions = []
    validation_labels = []
    for i, (X, y) in enumerate(validation_loader):
        X = torch.transpose(X, 0, 1)
        # --- Forward pass ---
        logits, halting_points = model(X, test=True)
        _, predictions = torch.max(torch.softmax(logits, dim=1), dim=1)

        validation_locations.append(halting_points)
        validation_predictions.append(predictions)
        validation_labels.append(y)

    validation_predictions = torch.stack(validation_predictions).numpy().reshape(-1, 1)
    validation_labels = torch.stack(validation_labels).numpy().reshape(-1, 1)
    validation_locations = torch.stack(validation_locations).numpy().reshape(-1, 1)

    print("Validation Accuracy: {}".format(np.round(accuracy_score(validation_labels, validation_predictions), 3)))
    print("Mean proportion used: {}%".format(np.round(100.*np.mean(validation_locations), 3)))

    # --- save model ---
    torch.save(model.state_dict(), model_save_path+"model.pt")
