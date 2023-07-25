import torch
import torch
from tqdm import tqdm
from .evaluation import QuantStub

def training_set(model):
    criterion =  model.loss#yolo_loss.Detection(7, 2, 20, 4, 1, 1.0, 0.5, 1.0, 5.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.00015, momentum=0.9, weight_decay=0.0001, nesterov=True)#lr=0.00015
    scheduler= torch.optim.lr_scheduler.CyclicLR(optimizer,base_lr=0.0001,max_lr=0.01,base_momentum=0.1,mode="triangular2",step_size_up=4000)
    return criterion,optimizer,scheduler

def train_one_epoch(model,dataloaders,data_config,criterion,optimizer,scheduler,device,symm=True,bits=8):
    for name,module in model.named_modules():
        if hasattr(module,"p"):
            module.p=0.5
    model.to(device)
    model.train()
    running_loss = 0.0

    for inputs, labels in tqdm(dataloaders):
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        #QuantStub(inputs,data_config['fp32_min'],data_config['fp32_max'],symm,bits,isHW=False)
        with torch.set_grad_enabled(True):
            pred = model(inputs)
            #labels=torch.tensor(labels.cpu().numpy().transpose(0,3,1,2))
            print(pred.size())
            print(labels.size())
            #print(criterion)
            loss = criterion(pred, labels)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        scheduler.step()
    epoch_loss = running_loss
    print('Loss: {:.4}'.format(epoch_loss))
    return model
