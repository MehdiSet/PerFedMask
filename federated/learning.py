import sys

import numpy as np
import torch
import copy
from torch import optim
#from advertorch.context import ctx_noparamgrad_and_eval
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from utils.utils import AverageMeter
from nets.dual_bn import set_bn_mode


def if_use_dbn(model):
    if isinstance(model, DDP):
        return model.module.bn_type.startswith('d')
    else:
        return model.bn_type.startswith('d')
    
    
    
def train_fedprox(mu, model, data_loader, optimizer, loss_fun, device, start_iter=0, max_iter=np.inf, progress=True):
    
    model.train()
    serverModel = copy.deepcopy(model)

    loss_all = 0
    total = 0
    correct = 0
    max_iter = len(data_loader) if max_iter == np.inf else max_iter
    data_iterator = iter(data_loader)
    tqdm_iters = tqdm(range(start_iter, max_iter), file=sys.stdout) \
        if progress else range(start_iter, max_iter)

    # ordinary training.
    set_bn_mode(model, False)  # set clean mode
    for step in tqdm_iters:
    # for data, target in tqdm(data_loader, file=sys.stdout):
        try:
            data, target = next(data_iterator)
        except StopIteration:
            data_iterator = iter(data_loader)
            data, target = next(data_iterator)
        optimizer.zero_grad()
        model.zero_grad()

        data = data.to(device)
        target = target.to(device)
        output = model(data)
        loss = loss_fun(output, target.long())
        
        ##################### fedProx Implementation #####################
        w_diff = torch.tensor(0., device=device)
        for w, w_t in zip(serverModel.parameters(), model.parameters()):
            w_diff += torch.pow(torch.norm(w - w_t), 2)
        loss += mu / 2. * w_diff
        ##################################################################

        loss_all += loss.item() * target.size(0)
        total += target.size(0)
        pred = output.data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()

        loss.backward()
        optimizer.step()
    return loss_all / total, correct / total


def train(model, data_loader, optimizer, loss_fun, device, start_iter=0, max_iter=np.inf, progress=True):

    model.train()
    loss_all = 0
    total = 0
    correct = 0
    max_iter = len(data_loader) if max_iter == np.inf else max_iter
    data_iterator = iter(data_loader)
    tqdm_iters = tqdm(range(start_iter, max_iter), file=sys.stdout) \
        if progress else range(start_iter, max_iter)

    # ordinary training.
    set_bn_mode(model, False)  # set clean mode
    for step in tqdm_iters:
    # for data, target in tqdm(data_loader, file=sys.stdout):
        try:
            data, target = next(data_iterator)
        except StopIteration:
            data_iterator = iter(data_loader)
            data, target = next(data_iterator)
        optimizer.zero_grad()
        model.zero_grad()

        data = data.to(device)
        target = target.to(device)
        output = model(data)
        loss = loss_fun(output, target.long())

        loss_all += loss.item() * target.size(0)
        total += target.size(0)
        pred = output.data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()

        loss.backward()
        optimizer.step()

    return loss_all / total, correct / total


def train_slimmable(model, data_loader, optimizer, loss_fun, device,
                    start_iter=0, max_iter=np.inf,
                    slim_ratios=[0.5, 0.75, 1.0], slim_shifts=0, out_slim_shifts=None,
                    progress=True, loss_temp='none'):
    """If slim_ratios is a single value, use `train` and set slim_ratio outside, instead.
    """
    # expand scalar slim_shift to list
    if not isinstance(slim_shifts, (list, tuple)):
        slim_shifts = [slim_shifts for _ in range(len(slim_ratios))]
    if not isinstance(out_slim_shifts, (list, tuple)):
        out_slim_shifts = [out_slim_shifts for _ in range(len(slim_ratios))]

    model.train()
    total, correct, loss_all = 0, 0, 0
    max_iter = len(data_loader) if max_iter == np.inf else max_iter
    data_iterator = iter(data_loader)

    # ordinary training.
    set_bn_mode(model, False)  # set clean mode
    for step in tqdm(range(start_iter, max_iter), file=sys.stdout, disable=not progress):
        # for data, target in tqdm(data_loader, file=sys.stdout):
        try:
            data, target = next(data_iterator)
        except StopIteration:
            data_iterator = iter(data_loader)
            data, target = next(data_iterator)
        optimizer.zero_grad()
        model.zero_grad()

        data = data.to(device)
        target = target.to(device)
        

        for slim_ratio, in_slim_shift, out_slim_shift \
                in sorted(zip(slim_ratios, slim_shifts, out_slim_shifts), reverse=False,
                          key=lambda ss_pair: ss_pair[0]):
            model.switch_slim_mode(slim_ratio, slim_bias_idx=in_slim_shift, out_slim_bias_idx=out_slim_shift)

            output = model(data)
            if loss_temp == 'none':
                _loss = loss_fun(output, target.long())
            elif loss_temp == 'auto':
                _loss = loss_fun(output/slim_ratio, target) * slim_ratio
            elif loss_temp.replace('.', '', 1).isdigit():  # is float
                _temp = float(loss_temp)
                _loss = loss_fun(output / _temp, target) * _temp
            else:
                raise NotImplementedError(f"loss_temp: {loss_temp}")

            loss_all += _loss.item() * target.size(0)
            total += target.size(0)
            pred = output.data.max(1)[1]
            correct += pred.eq(target.view(-1)).sum().item()

            _loss.backward()
        optimizer.step()

    return loss_all / total, correct / total


# =========== Test ===========


def personalization(model, data_loader_train, data_loader_test, loss_fun, global_lr, device, progress=False):

    
    model.train()
    
    optimizer = optim.SGD(params=model.parameters(), lr=global_lr,
                                      momentum=0.9, weight_decay=5e-4)
    
    loss_all, total, correct = 0, 0, 0
    for iter in range(5):
        for data, target in tqdm(data_loader_train, file=sys.stdout, disable=not progress):
            data, target = data.to(device), target.to(device)
    
            #with torch.no_grad():
            optimizer.zero_grad()
            model.zero_grad()
            output = model(data)
            loss = loss_fun(output, target.long())
    
            loss_all += loss.item()
            total += target.size(0)
            pred = output.data.max(1)[1]
            correct += pred.eq(target.view(-1)).sum().item()
            
            loss.backward()
            optimizer.step()
            
        
    val_loss, val_acc = test(model, data_loader_test, loss_fun, device)
        
    return val_loss, val_acc

def personalization_slimmable(model, data_loader_train, data_loader_test, loss_fun, global_lr, device, progress=False):
    
    model.train()
    
    optimizer = optim.SGD(params=model.parameters(), lr=global_lr,
                                      momentum=0.9, weight_decay=5e-4)
    
    
    atom_slim_ratio = 0.125
    user_n_base = int(1.0 / atom_slim_ratio)
    slim_ratios = [atom_slim_ratio] * user_n_base
    slim_shifts = [ii for ii in range(user_n_base)]
    out_slim_shifts = [None for _ in range(len(slim_ratios))]
    
    set_bn_mode(model, False)
    for iter in range(5):
        for data, target in tqdm(data_loader_train, file=sys.stdout, disable=not progress):
            data, target = data.to(device), target.to(device)


    
            optimizer.zero_grad()
            model.zero_grad()
            
            
            for slim_ratio, in_slim_shift, out_slim_shift \
                    in sorted(zip(slim_ratios, slim_shifts, out_slim_shifts), reverse=False,
                              key=lambda ss_pair: ss_pair[0]):
                model.switch_slim_mode(slim_ratio, slim_bias_idx=in_slim_shift, out_slim_bias_idx=out_slim_shift)
            
                output = model(data)
                loss = loss_fun(output, target.long())
        

                
                loss.backward()
            optimizer.step()
            
    model.switch_slim_mode(1.0)   
    val_loss, val_acc = test(model, data_loader_test, loss_fun, device)
        

    return val_loss, val_acc


def test(model, data_loader, loss_fun, device, progress=False):

    model.eval()

    
    loss_all, total, correct = 0, 0, 0
    for data, target in tqdm(data_loader, file=sys.stdout, disable=not progress):
        data, target = data.to(device), target.to(device)


        with torch.no_grad():
            output = model(data)
            loss = loss_fun(output, target.long())

        loss_all += loss.item()
        total += target.size(0)
        pred = output.data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()
    return loss_all / len(data_loader), correct/total


def refresh_bn(model, data_loader, device, progress=False):
    model.train()
    for data, target in tqdm(data_loader, file=sys.stdout, disable=not progress):
        data, target = data.to(device), target.to(device)


        with torch.no_grad():
            model(data)


def fed_test_model(fed, running_model, test_loaders, loss_fun, device):
    test_acc_mt = AverageMeter()
    for test_idx, test_loader in enumerate(test_loaders):
        fed.download(running_model, test_idx)
        _, test_acc = test(running_model, test_loader, loss_fun, device)
        # print(' {:<11s}| Test  Acc: {:.4f}'.format(fed.clients[test_idx], test_acc))

        # wandb.summary[f'{fed.clients[test_idx]} test acc'] = test_acc
        test_acc_mt.append(test_acc)
    return test_acc_mt.avg

