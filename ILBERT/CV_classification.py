

import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.metrics import r2_score,mean_absolute_error, mean_squared_error
import yaml
import numpy as np
import torch
import time
import math
import pandas as pd
from model import ILBERT_class
from dataset import SMILES_dataset
import random
from torch.utils.data import Subset
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train(model, loader, optimizer):
    model.train()
    loss_all = []
    all_predictions = []
    all_targets = []

    with torch.set_grad_enabled(True):
        for datas, label, _, _, _ in loader:
            optimizer.zero_grad()

            data = [data.to(device) for data in datas]
            label = label.to(device)

            output = model(data).float()

            label = label.squeeze().long()  

            loss = F.cross_entropy(output, label)

            loss.backward()
            optimizer.step()


            loss_all.append(loss.item())
            all_predictions.append(output.argmax(dim=1).detach().cpu())  
            all_targets.append(label.detach().cpu())

    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)


    accuracy = accuracy_score(all_targets.numpy(), all_predictions.numpy())
    f1 = f1_score(all_targets.numpy(), all_predictions.numpy(), average='weighted')

    loss = np.mean(loss_all)
    return loss, accuracy, f1


def evaluate_dataset(model, dataset, fold, shuffle, name, save=False):
    model.eval()
    loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=shuffle)

    y, pred, smi, probs = [], [], [], []

    with torch.no_grad():
        for datas, label, smiles, _, _ in loader:

            data = [data.to(device) for data in datas]
            label = label.squeeze().long().to(device) 

            output = model(data)

            predicted = output.argmax(dim=1).detach().cpu()
            prob = torch.softmax(output, dim=1).max(dim=1)[0].detach().cpu()

            y.extend(label.detach().cpu().numpy())
            pred.extend(predicted.numpy())
            smi.extend(smiles)
            probs.extend(prob.numpy())

    accuracy = accuracy_score(y, pred)
    f1 = f1_score(y, pred, average='weighted')

    if save:
        df = pd.DataFrame({
            'True Labels': y,
            'Predicted Labels': pred,
            'SMILES': smi,
            'Predicted Probabilities': probs
        })
        df.to_csv(f'training_results/fold_{fold + 1}_predictions_{name}.csv', index=False)

    return accuracy, f1


def calculate_metrics(csv_path):
    data = pd.read_csv(csv_path)
    true_labels = data['True Labels']
    predicted_labels = data['Predicted Labels']

    accuracy = accuracy_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels, average='weighted')

    return accuracy, f1


def cross_validation(dataset,config,df,num_folds):

    kf = KFold(n_splits=num_folds, shuffle=True, random_state=config['seed'])
    indices = df[config['split by']].unique()

    for fold, (train_index, test_index) in enumerate(kf.split(indices)):

        train_indices = indices[train_index]
        test_indices = indices[test_index]
        print(len(train_indices),len(test_index))


        train_dataset = Subset(dataset, [i for i, idx in enumerate(df[config['split by']]) if idx in train_indices])
        test_dataset = Subset(dataset, [i for i, idx in enumerate(df[config['split by']]) if idx in test_indices])


        print("Fold:", fold + 1)
        print("Train dataset size:", len(train_dataset))
        print("Test dataset size:", len(test_dataset))

        model = ILBERT_class(**config["transformer"]).to(device)

        if config['Load pretrained model']== True:
            state_dict = torch.load("model_weight/pretrained_model.pth", map_location=device)
            model.load_state_dict(state_dict, strict=False)

        if config['Freeze']== True:
            for param in model.roberta.parameters():
                param.requires_grad = False

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config['init_lr'], weight_decay=config['weight_decay'])


        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=config['lr_decay_patience'],
                                                                factor=config['lr_decay_factor'], min_lr=config['min_lr'])

        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)


        best_valid_f1 = 0
        best_train_loss = float('inf')
        best_valid_accuracy = 0

        early_stopping_count = 0
        epoch_times = []

        for epoch in range(1, config['epochs'] + 1):
            start_time = time.time()

            model.train()
            train_loss, train_accuracy, train_f1 = train(model, train_loader, optimizer)

            model.eval()
            valid_accuracy, valid_f1 = evaluate_dataset(model,test_dataset ,fold,name='val',shuffle=False)

            end_time = time.time()
            epoch_time = end_time - start_time
            epoch_times.append(epoch_time)
            print(f"Epoch{epoch:3d},Time:{epoch_time:.2f}s,TrainLoss:{train_loss:.3f},TrainR2:{train_accuracy:.3f},valid_accuracy:{valid_accuracy:.3f},valid_f1:{valid_f1:.3f}")


            if valid_f1 > best_valid_f1:
                best_train_loss=train_loss
                best_valid_accuracy = valid_accuracy
                best_valid_f1= valid_f1

                torch.save(model.state_dict(), f'model_weight/fold_{fold + 1}_best_model.pth')

                early_stopping_count = 0
            else:
                early_stopping_count += 1


            if early_stopping_count >= config["early_stop_patience"]:
                print(f"/nEarly stopping at epoch {epoch + 1}")
                break

            lr_scheduler.step(valid_f1)

            current_lr = lr_scheduler.get_last_lr()
            print(f'Epoch {epoch}: Learning rate = {current_lr}')

        average_epoch_time = sum(epoch_times) / len(epoch_times)
        model.load_state_dict(torch.load(f'model_weight/fold_{fold+1}_best_model.pth'))
        test_accuracy, test_f1= evaluate_dataset(model, test_dataset, fold,shuffle=False,name='test',save=True)

        print(f'Best_Epoch:{epoch - early_stopping_count}, Average epoch time: {average_epoch_time:.2f}s,Train_Loss: {best_train_loss:.6f}, test_accuracy: {test_accuracy:.6f},test_f1: {test_f1:.6f}')


    df = pd.DataFrame()
    for i in range(1, num_folds + 1):
        filename = f"training_results/fold_{i}_predictions_test.csv"
        fold_df = pd.read_csv(filename)
        df = pd.concat([df, fold_df])

    y_true = df['True Labels']
    y_pred = df['Predicted Labels']


    accuracy = accuracy_score(y_true, y_pred)
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    f1_macro = f1_score(y_true, y_pred, average='macro')

    print(accuracy, f1_weighted, f1_macro)



if __name__ == "__main__":

    config = yaml.load(open("config_final.yaml", "r", encoding="utf-8"), Loader=yaml.FullLoader)
    setup_seed(config['seed'])
    print(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('running on:',device)
    print('---loading dataset---')

    if config['task'] == 'Melting point':
        target = 'Target'
        print('#--------------------------#')
        print('Training on Melting point classification task')
        print('#--------------------------#')

        df = pd.read_csv("data/MP/modified_MP.csv")
    else:
        print("Invalid Downstream Task!")

    from ILtokenizer import SMILES_Atomwise_Tokenizer
    tokenizer=SMILES_Atomwise_Tokenizer('merged_vocab.txt')

    dataset = SMILES_dataset(df = df, tokenizer = tokenizer,target=target)
    loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False)

    cross_validation(dataset, config, df,config['k'])




