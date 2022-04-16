import wandb
import torch
from torch.autograd import Variable
from timeit import default_timer as timer
from tqdm import tqdm
import os
import pandas as pd
import click
import json
from losses import get_loss
from dataset import get_data, make_loader
from model_selector import get_model
from min_norm_solvers import MinNormSolver, gradient_normalizers
from train_utils import create_optimizer, create_lr_scheduler

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@click.command()
@click.option('--param_file', default='config.json', help='JSON parameters file')
def model_pipeline(param_file):
    with open(param_file) as json_params:
        params = json.load(json_params)

    with wandb.init(project="MTL_ViT", config=params):
        # access all HPs through wandb.config, so logging matches execution!
        config = wandb.config

        tasks = config['tasks']

        # make the model, data, and optimization problem
        model, train_loader, val_loader, test_loader, criterion, optimizer, lr_scheduler = make(config)

        # and use them to train the model
        train(model, train_loader, val_loader, criterion, optimizer, lr_scheduler, config, tasks)

        test(model, test_loader, tasks)

    return model


def make(config):
    # Make the data
    train = get_data(img_path='../Data/train/DWT/img_file/',
                     label_path='../Data/train/DWT/label/label.csv')
    val = get_data(img_path='../Data/dev/DWT/img_file/',
                   label_path='../Data/dev/DWT/label/label.csv')
    test = get_data(img_path='../Data/test/DWT/img_file/',
                    label_path='../Data/test/DWT/label/label.csv')
    train_loader = make_loader(train, batch_size=config['BATCH_SIZE'])
    val_loader = make_loader(val, batch_size=config['BATCH_SIZE'])
    test_loader = make_loader(test, batch_size=config['BATCH_SIZE'])

    # Make the model
    model = get_model(config, device)
    model_params = []
    for m in model:
        model_params += model[m].parameters()

    # Make the loss and optimizer
    criterion = get_loss(config)
    optimizer = create_optimizer(config, model_params)
    lr_scheduler = create_lr_scheduler(optimizer, config)

    return model, train_loader, val_loader, test_loader, criterion, optimizer, lr_scheduler


def MGDA(model, images, labels, criterion, optimizer, config, tasks):
    # Scaling the loss functions based on the algorithm choice
    loss_data = {}
    grads = {}
    scale = {}

    if 'mgda' in config['algorithm']:
        # Will use our MGDA_UB if approximate_norm_solution is True. Otherwise, will use MGDA
        approximate_norm_solution = config['use_approximation']

        if approximate_norm_solution:
            optimizer.zero_grad()
            # First compute representations (z)
            with torch.no_grad():
                images_volatile = Variable(images.data)
                rep = model['rep'](images_volatile)
                # As an approximate solution we only need gradients for input
                if isinstance(rep, list):
                    # This is a hack to handle psp-net
                    rep = rep[0]
                    rep_variable = [Variable(rep.data.clone(), requires_grad=True)]
                    list_rep = True
                else:
                    rep_variable = Variable(rep.data.clone(), requires_grad=True)
                    list_rep = False

            # Compute gradients of each loss function wrt z
            for t in tasks:
                optimizer.zero_grad()
                out_t = model[t](rep_variable)
                loss = criterion[t](out_t, labels[t])
                loss_data[t] = loss.item()
                loss.backward()
                grads[t] = []
                if list_rep:
                    grads[t].append(Variable(rep_variable[0].grad.data.clone(), requires_grad=False))
                    rep_variable[0].grad.data.zero_()
                else:
                    grads[t].append(Variable(rep_variable.grad.data.clone(), requires_grad=False))
                    rep_variable.grad.data.zero_()
        else:
            # This is MGDA
            for t in tasks:
                # Comptue gradients of each loss function wrt parameters
                optimizer.zero_grad()
                rep = model['rep'](images)
                out_t = model[t](rep)
                loss = criterion[t](out_t, labels[t])
                loss_data[t] = loss.item()
                loss.backward()
                grads[t] = []
                for param in model['rep'].parameters():
                    if param.grad is not None:
                        grads[t].append(Variable(param.grad.data.clone(), requires_grad=False))

        # Normalize all gradients, this is optional and not included in the paper.
        gn = gradient_normalizers(grads, loss_data, config['normalization_type'])
        for t in tasks:
            for gr_i in range(len(grads[t])):
                grads[t][gr_i] = grads[t][gr_i] / gn[t]

        # Frank-Wolfe iteration to compute scales.
        sol, min_norm = MinNormSolver.find_min_norm_element([grads[t] for t in tasks])
        for i, t in enumerate(tasks):
            scale[t] = float(sol[i])
    else:
        for t in tasks:
            scale[t] = float(config['scales'][t])

    return scale


def train_batch(images, labels, model, optimizer, criterion, scale, tasks):
    loss_data = {}
    optimizer.zero_grad()
    rep = model['rep'](images)
    for i, t in enumerate(tasks):
        out_t = model[t](rep)
        loss_t = criterion[t](out_t, labels[t])
        loss_data[t] = loss_t.item()
        if i > 0:
            loss = loss + scale[t] * loss_t
        else:
            loss = scale[t] * loss_t
    loss.backward()
    optimizer.step()

    return loss, loss_data


def create_eval():
    training_loss = {'Zero Person': [], 'One Person': [], 'One+ Person': [], 'Fall': [], 'Sit Down': [], 'Stand up': [],
                     'Walk': [], 'lie Down': [], 'Dynamic Activity': [], 'Static Activity': []}
    validation_loss = {'Zero Person': [], 'One Person': [], 'One+ Person': [], 'Fall': [], 'Sit Down': [],
                       'Stand up': [],
                       'Walk': [], 'lie Down': [], 'Dynamic Activity': [], 'Static Activity': []}
    precision_val = {'Zero Person': [], 'One Person': [], 'One+ Person': [], 'Fall': [], 'Sit Down': [], 'Stand up': [],
                     'Walk': [], 'lie Down': [], 'Dynamic Activity': [], 'Static Activity': []}
    recall_val = {'Zero Person': [], 'One Person': [], 'One+ Person': [], 'Fall': [], 'Sit Down': [], 'Stand up': [],
                  'Walk': [], 'lie Down': [], 'Dynamic Activity': [], 'Static Activity': []}
    F1_score_val = {'Zero Person': [], 'One Person': [], 'One+ Person': [], 'Fall': [], 'Sit Down': [], 'Stand up': [],
                    'Walk': [], 'lie Down': [], 'Dynamic Activity': [], 'Static Activity': []}
    accuracy_val = {'Zero Person': [], 'One Person': [], 'One+ Person': [], 'Fall': [], 'Sit Down': [], 'Stand up': [],
                    'Walk': [], 'lie Down': [], 'Dynamic Activity': [], 'Static Activity': []}
    TP = {'Zero Person': [], 'One Person': [], 'One+ Person': [], 'Fall': [], 'Sit Down': [], 'Stand up': [],
          'Walk': [], 'lie Down': [], 'Dynamic Activity': [], 'Static Activity': []}
    TN = {'Zero Person': [], 'One Person': [], 'One+ Person': [], 'Fall': [], 'Sit Down': [], 'Stand up': [],
          'Walk': [], 'lie Down': [], 'Dynamic Activity': [], 'Static Activity': []}
    FP = {'Zero Person': [], 'One Person': [], 'One+ Person': [], 'Fall': [], 'Sit Down': [], 'Stand up': [],
          'Walk': [], 'lie Down': [], 'Dynamic Activity': [], 'Static Activity': []}
    FN = {'Zero Person': [], 'One Person': [], 'One+ Person': [], 'Fall': [], 'Sit Down': [], 'Stand up': [],
          'Walk': [], 'lie Down': [], 'Dynamic Activity': [], 'Static Activity': []}

    return training_loss, validation_loss, precision_val, recall_val, F1_score_val, accuracy_val, TP, TN, FP, FN


def log_table(training_loss, validation_loss, precision_val, recall_val, F1_score_val, accuracy_val):
    train_loss = wandb.Table(dataframe=pd.DataFrame(training_loss))
    val_loss = wandb.Table(dataframe=pd.DataFrame(validation_loss))
    precision = wandb.Table(dataframe=pd.DataFrame(precision_val))
    recall = wandb.Table(dataframe=pd.DataFrame(recall_val))
    F1 = wandb.Table(dataframe=pd.DataFrame(F1_score_val))
    val_accuracy = wandb.Table(dataframe=pd.DataFrame(accuracy_val))
    wandb.log({"train_loss": train_loss, "val_loss": val_loss, "precision": precision,
               "recall": recall, "F1": F1, "val_accuracy": val_accuracy})


def train(model, train_loader, val_loader, criterion, optimizer, lr_scheduler, config, tasks):
    training_loss, validation_loss, precision_val, recall_val, F1_score_val, accuracy_val, TP, TN, FP, FN = create_eval()

    # Run training and track with wandb

    for epoch in tqdm(range(config['NUM_EPOCH'])):
        start = timer()
        for m in model:
            model[m].train()

        for images, total_label in train_loader:
            images = images.squeeze(1)
            images = images.float()
            images = images.to(device)

            labels = {}
            # Read all targets of all tasks
            for i, t in enumerate(tasks):
                labels[t] = total_label[:, i]
                labels[t] = labels[t].long()
                labels[t] = labels[t].to(device)

            scale = MGDA(model, images, labels, criterion, optimizer, config, tasks)
            loss, loss_data = train_batch(images, labels, model, optimizer, criterion, scale, tasks)

            wandb.log({"train_loss": loss}, step=epoch)

            for t in tasks:
                training_loss[t].append(loss_data[t])

        for m in model:
            model[m].eval()

        tot_loss = {}
        tot_loss['all'] = 0.0
        for t in tasks:
            tot_loss[t] = 0.0

        with torch.no_grad():
            for t in tasks:
                TP[t].append(0)
                TN[t].append(0)
                FP[t].append(0)
                FN[t].append(0)

            print('Start test model in val dataset')
            for images_dev, total_labels_dev in val_loader:
                images_dev = images_dev.squeeze(1)
                images_dev = images_dev.float()
                images_dev = images_dev.to(device)
                labels_dev = {}

                for i, t in enumerate(tasks):
                    labels_dev[t] = total_labels_dev[:, i]
                    labels_dev[t] = labels_dev[t].long()
                    labels_dev[t] = labels_dev[t].to(device)

                val_rep = model['rep'](images_dev)
                for t in tasks:
                    out_t_val = model[t](val_rep)
                    loss_t = criterion[t](out_t_val, labels_dev[t])
                    tot_loss['all'] += loss_t.item()
                    tot_loss[t] += loss_t.item()

                    if out_t_val.ndim == 2:
                        out_t_val = out_t_val.argmax(dim=1)

                    TP[t][epoch] += ((out_t_val == 1) & (labels_dev[t].data == 1)).cpu().sum()
                    TN[t][epoch] += ((out_t_val == 0) & (labels_dev[t].data == 0)).cpu().sum()
                    FN[t][epoch] += ((out_t_val == 0) & (labels_dev[t].data == 1)).cpu().sum()
                    FP[t][epoch] += ((out_t_val == 1) & (labels_dev[t].data == 0)).cpu().sum()

            for t in tasks:
                validation_loss[t].append(tot_loss[t] / len(val_loader))
                precision_val[t].append(TP[t][epoch] / (TP[t][epoch] + FP[t][epoch]))
                recall_val[t].append(TP[t][epoch] / (TP[t][epoch] + FN[t][epoch]))
                F1_score_val[t].append(2 * recall_val[t][epoch] * precision_val[t][epoch] / (
                        recall_val[t][epoch] + precision_val[t][epoch]))
                accuracy_val[t].append((TP[t][epoch] + TN[t][epoch]) / (
                        TP[t][epoch] + TN[t][epoch] + FP[t][epoch] + FN[t][epoch]))

            val_loss_total = tot_loss['all'] / (len(tasks)*len(val_loader))

            wandb.log({"validation_loss": val_loss_total}, step=epoch)

        lr_scheduler.step()

        end = timer()
        print('Epoch ended in {}s'.format(end - start))

    log_table(training_loss, validation_loss, precision_val, recall_val, F1_score_val, accuracy_val)


def test(model, test_loader, tasks):
    for m in model:
        model[m].eval()

    # Run the model on some test examples
    with torch.no_grad():
        for images, total_label in test_loader:
            total = len(test_loader)
            images = images.squeeze(1)
            images = images.float()
            images = images.to(device)

            labels = {}
            # Read all targets of all tasks
            for i, t in enumerate(tasks):
                labels[t] = total_label[:, i]
                labels[t] = labels[t].long()
                labels[t] = labels[t].to(device)

            correct = {'Zero Person': 0.0, 'One Person': 0.0, 'One+ Person': 0.0, 'Fall': 0.0, 'Sit Down': 0.0,
                       'Stand up': 0.0, 'Walk': 0.0, 'lie Down': 0.0, 'Dynamic Activity': 0.0, 'Static Activity': 0.0}
            test_rep = model['rep'](images)
            for t in tasks:
                out_t_test = model[t](test_rep)
                if out_t_test.ndim == 2:
                    out_t_test = out_t_test.argmax(dim=1)
                correct[t] += (out_t_test == labels[t]).sum().item()

        for t in tasks:
            wandb.log({t: correct[t] / total})

        torch.onnx.export(model['rep'].module, images, "model_rep.onnx", opset_version=12)
        wandb.save("model_rep.onnx")
        for t in tasks:
            torch.onnx.export(model[t].module, test_rep, "model_{}.onnx".format(t), opset_version=12)
            wandb.save("model_{}.onnx".format(t))




if __name__ == '__main__':
    model_pipeline()
