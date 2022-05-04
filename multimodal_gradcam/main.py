import json
import os
import signal
import sys
import time

import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.nn import BCEWithLogitsLoss

from dataset_util import MultimodalDataset
from model import MultimodalClassifier
from consts import global_consts as gc

def eval_hateful_memes(model_output, labels):
    truth = labels.float()
    preds = torch.round(torch.sigmoid(output.squeeze()))
    acc = torch.abs(truth-preds).view(-1)
    acc = (1. - acc.sum() / acc.size()[0])
    return acc.item()



def train_model(model_name):
    ds = MultimodalDataset
    train_dataset = ds(gc.data_path)
    print(train_dataset.vision.shape)
    train_loader = Data.DataLoader(
        dataset=train_dataset,
        batch_size=gc.batch_size,
        num_workers=0
    )
    print("Data Successfully Loaded. len={}".format(len(train_loader)))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
            torch.cuda.set_device(0)
    print("running device: ", device)

    net = MultimodalClassifier()
    # print(net)
    net.to(device)

    if gc.dataset == "hateful_memes":
        criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(net.parameters(), lr=gc.lr)
    print("Start training...")
    net.train()
    start_epoch = 0
    data_num = 0
    model_path = os.path.join(gc.model_dir, gc.dataset + '_' + model_name + '.pt')
    if gc.load_model and os.path.exists(model_path):
        checkpoint = torch.load(model_path)
        net.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        loaded_loss = checkpoint["loss"]
        loaded_acc = checkpoint["accuracy"]
        print("Loaded model from path {}, at epoch {}, current loss: {}, current acc:{}...".format(model_path, start_epoch, loaded_loss, loaded_acc))
    else:
        print("No model loaded, start training from epoch 0...")

    for epoch in range(start_epoch, gc.epoch_num):
        running_train_loss = 0
        running_train_acc = 0
        for i, data in enumerate(train_loader):
            batch_update_dict = {}
            data_num += len(data)
            max_i = i
            input_ids, attn_mask, vision, labels = data
            input_ids, attn_mask, vision, labels = input_ids.to(device), attn_mask.to(device), vision.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(vision, input_ids, attn_mask)
            batch_loss = criterion(torch.round(torch.sigmoid(output.squeeze())), targets.float())
            batch_loss.backward()
            optimizer.step()
            batch_acc = eval_hateful_memes(outputs, labels)
            running_train_acc += batch_acc
            running_train_loss += batch_loss.item()
        train_acc = running_train_acc / data_num
        train_loss = running_train_loss / data_num
        print("epoch {}/{}: train loss = {}, train accuracy = {}...".format(epoch+1, gc.epoch_num, train_loss, train_acc))
    if gc.save_model:
        torch.save({
                    "epoch": gc.epoch_num-1,
                    "model_state_dict": net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": train_loss,
                    "accuracy": train_acc
                    },
                    model_path)
        print("Successfully saved model at epoch {}, at path {}.".format(gc.epoch_num-1, model_path))


if __name__ == "__main__":
    start_time = time.time()
    print('Start time: ' + time.strftime("%H:%M:%S", time.gmtime(start_time)))
    # main process
    train_model("vanilla")
    elapsed_time = time.time() - start_time
    print('Total time: ' + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
