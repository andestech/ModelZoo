import torch
from tqdm import tqdm
from .evaluation import QuantStub


def training_set(model, device):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.00015, momentum=0.1, weight_decay=0.00001, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.000001, max_lr=0.00001, base_momentum=0.1, max_momentum=0.9, mode="triangular2", step_size_up=(80000))
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
