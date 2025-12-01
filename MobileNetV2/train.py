import torch
from tqdm import tqdm
from .evaluation import QuantStub


def training_set(model, device):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.00001, momentum=0.9, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 1000)
    return criterion,optimizer,scheduler


def train_one_epoch(model, dataloaders, data_config, criterion, optimizer, scheduler, device, symm=True, bits=8):
    model.to(device)
    model.train()
    running_loss = 0.0
    running_corrects = 0
    running_size=0

    for inputs, labels in tqdm(dataloaders):
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        QuantStub(inputs, data_config['fp32_min'], data_config['fp32_max'], symm, bits, isHW=False)
        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        running_size += inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        scheduler.step()

    epoch_loss = running_loss / running_size
    epoch_acc = running_corrects.double() / running_size
    print('Loss: {:.4} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
    return model
