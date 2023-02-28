"""FedBABU, FedAvg, fedProx, and fedNOVA"""
import os, argparse, time
import numpy as np
import wandb
import copy
import torch
import math
from torch import nn, optim
# federated
from federated.learning import train, test, personalization, train_fedprox
# utils
from utils.utils import set_seed, AverageMeter, CosineAnnealingLR, \
    MultiStepLR, str2bool
from utils.config import CHECKPOINT_ROOT

# NOTE import desired federation
from federated.core import HeteFederation as Federation


def render_run_name(args, exp_folder):
    """Return a unique run_name from given args."""
    if args.model == 'default':
        args.model = {'Digits': 'digit', 'Cifar10': 'preresnet18', 'Cifar100': 'mobile', 'DomainNet': 'alex'}[args.data]
    run_name = f'{args.model}'
    if args.width_scale != 1.: run_name += f'x{args.width_scale}'
    run_name += Federation.render_run_name(args)
    # log non-default args
    if args.seed != 1: run_name += f'__seed_{args.seed}'
    # opt
    if args.lr_sch != 'none': run_name += f'__lrs_{args.lr_sch}'
    if args.opt != 'sgd': run_name += f'__opt_{args.opt}'
    if args.batch != 32: run_name += f'__batch_{args.batch}'
    if args.wk_iters != 1: run_name += f'__wk_iters_{args.wk_iters}'
    # slimmable
    if args.no_track_stat: run_name += f"__nts"
    if args.no_mask_loss: run_name += f'__nml'


    args.save_path = os.path.join(CHECKPOINT_ROOT, exp_folder)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    SAVE_FILE = os.path.join(args.save_path, run_name)
    return run_name, SAVE_FILE


def get_model_fh(data, model):
    if data == 'Digits':
        if model in ['digit']:
            from nets.models import DigitModel
            ModelClass = DigitModel
        else:
            raise ValueError(f"Invalid model: {model}")
    elif data in ['DomainNet']:
        if model in ['alex']:
            from nets.models import AlexNet
            ModelClass = AlexNet
        else:
            raise ValueError(f"Invalid model: {model}")
    elif data == 'Cifar10':
        if model in ['preresnet18']:  # From heteroFL
            from nets.HeteFL.preresne import resnet18
            ModelClass = resnet18
        else:
            raise ValueError(f"Invalid model: {model}")
    elif data == 'Cifar100':
        if model in ['mobile']:  # From heteroFL
            from nets.Nets import MobileNetCifar
            ModelClass = MobileNetCifar
        else:
            raise ValueError(f"Invalid model: {model}")
    else:
        raise ValueError(f"Unknown dataset: {data}")
    return ModelClass


def mask_fed_test(fed, running_model, train_loaders, val_loaders, global_lr, verbose):
    mark = 's'
    val_acc_list_bp = [None for _ in range(fed.client_num)]
    val_loss_mt_bp = AverageMeter()
    
    val_acc_list = [None for _ in range(fed.client_num)]
    val_loss_mt = AverageMeter()
    for client_idx in range(fed.client_num):
        fed.download(running_model, client_idx)
        val_model = copy.deepcopy(running_model)
        # Test
        # Loss and accuracy before personalization
        val_loss_bp, val_acc_bp = test(val_model, val_loaders[client_idx], loss_fun, device) 
        
        # Log
        val_loss_mt_bp.append(val_loss_bp)
        val_acc_list_bp[client_idx] = val_acc_bp
        if verbose > 0:
            print(' {:<19s} Val Before Personalization {:s}Loss: {:.4f} | Val {:s}Acc: {:.4f}'.format(
                'User-'+fed.clients[client_idx], mark.upper(), val_loss_bp, mark.upper(), val_acc_bp))
        wandb.log({
            f"{fed.clients[client_idx]} val_bp_{mark}-acc": val_acc_bp,
        }, commit=False)
        
        if args.test:     
            
            # Personalization
            
            val_loss, val_acc = personalization(val_model, train_loaders[client_idx], val_loaders[client_idx], 
                                                loss_fun, global_lr, device)
    
            # Log
            val_loss_mt.append(val_loss)
            val_acc_list[client_idx] = val_acc
            if verbose > 0:
                print(' {:<19s} Val {:s}Loss: {:.4f} | Val {:s}Acc: {:.4f}'.format(
                    'User-'+fed.clients[client_idx], mark.upper(), val_loss, mark.upper(), val_acc))
            wandb.log({
                f"{fed.clients[client_idx]} val_{mark}-acc": val_acc,
            }, commit=False)
            
    if args.test:  
        
        return val_acc_list, val_loss_mt.avg, val_acc_list_bp, val_loss_mt_bp.avg
    else:   
        
        return val_acc_list_bp, val_loss_mt_bp.avg, val_acc_list_bp, val_loss_mt_bp.avg


if __name__ == '__main__':
    

    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser()
    # basic problem setting
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--data', type=str, default='Cifar10', help='data name') #   'DomainNet'  'Cifar100'
    parser.add_argument('--model', type=str.lower, default='default', help='model name')
    parser.add_argument('--algorithm', type=str, default='fedBABU', help='algorithm name') #  'fedProx' 'fedNOVA' 'fedProx' 
    parser.add_argument('--mu', type=float, default=0.0, help='The hyper parameter for fedProx algorithm') # 0.1
    parser.add_argument('--width_scale', type=float, default=1., help='model width scale')
    parser.add_argument('--no_track_stat', action='store_true', help='disable BN tracking')
    parser.add_argument('--no_mask_loss', action='store_true', help='disable masked loss for class'
                                                                    ' niid')
    # control
    parser.add_argument('--no_log', action='store_true', help='disable wandb log')
    parser.add_argument('--test', action='store_true', help='test the pretrained model')
    #parser.add_argument('--test', type=str2bool, default=True, help='test the pretrained model') #action='store_true'
    parser.add_argument('--resume', action='store_true', help='resume training from checkpoint')
    #parser.add_argument('--resume', type=str2bool, default=True, help='resume training from checkpoint')
    parser.add_argument('--verbose', type=int, default=0, help='verbose level: 0 or 1')
    # federated
    Federation.add_argument(parser)
    # optimization
    parser.add_argument('--lr', type=float, default=1e-1, help='learning rate') #1e-2 1e-1
    parser.add_argument('--lr_sch', type=str, default='multi_step', help='learning rate schedule') #'none'  'cos'
    parser.add_argument('--opt', type=str.lower, default='sgd', help='optimizer')
    parser.add_argument('--iters', type=int, default=80, help='#iterations for communication')#200
    parser.add_argument('--wk_iters', type=int, default=4, help='#epochs in local train')#5 1 
    parser.add_argument('--wk_factor', type=int, default=0.5, help='#decreasing factor of local training epochs for less capable devices') 

    args = parser.parse_args()

    set_seed(args.seed)



    # /////////////////////////////////
    # ///// Fed Dataset and Model /////
    # /////////////////////////////////
    fed = Federation(args.data, args)
    # Data
    train_loaders, val_loaders, test_loaders = fed.get_data()
    mean_batch_iters = int(np.mean([len(tl) for tl in train_loaders]))
    print(f"  mean_batch_iters: {mean_batch_iters}")
    
    # set experiment files, wandb
    exp_folder = f'Alg_{args.algorithm}_C{fed.args.pr_nuser}_{args.data}'
    run_name, SAVE_FILE = render_run_name(args, exp_folder)
    wandb.init(group=run_name[:120], project=exp_folder,
               mode='offline' if args.no_log else 'online',
               config={**vars(args), 'save_file': SAVE_FILE})

    # Model
    ModelClass = get_model_fh(args.data, args.model)
    running_model = ModelClass(
        track_running_stats=False, num_classes=fed.num_classes,
        width_scale=args.width_scale,
    ).to(device)
    



    # Loss
    loss_fun = nn.CrossEntropyLoss()


    # Use running model to init a fed aggregator
    fed.make_aggregator(running_model)
    
    

    # Last layer as head model
    if (args.model == 'alex'):
        
        head_part = 'fc3'
        
    else:
        
        head_part = 'linear'
    
    # Masking elements for each user
    names = []
    paramSize = []
    for name, par in running_model.named_parameters():
        
            
            
        if (args.algorithm == 'fedBABU'):
            if head_part not in name:
                
                names.append(name)
                paramSize.append(np.prod(list(par.size())))
        else:
            names.append(name)
            paramSize.append(np.prod(list(par.size())))
            
        
    
    totalParamNum = sum(paramSize)  
    users_max_comp = fed.get_user_max_slim_ratios()
    
    

    users_max_slim_ratio = [1.0] * fed.client_sampler.tot()
        
        

    computable_body_layers = {userIdx: [] for userIdx in range(fed.client_sampler.tot())}
    totParamNum = 0
    userCounter = 0
    tau_fedNOVA = [1.0] * fed.client_sampler.tot()
    aNorm_fedNOVA = [1.0] * fed.client_sampler.tot()

    
    # For obtaining the parameters related to FedNova, we have used the equations in https://proceedings.neurips.cc/paper/2020/file/564127c03caab942e503ee6f810f54fd-Paper.pdf 
    for userIdx in range(fed.client_sampler.tot()):
        
        if (users_max_comp[userIdx] == 1.0):
            tau_fedNOVA[userIdx] = (args.wk_iters) * len(train_loaders[userIdx]) 
        else:
            # Less capable devices perform lower number of local update iterations
            tau_fedNOVA[userIdx] = math.ceil(args.wk_iters*args.wk_factor) * len(train_loaders[userIdx]) 
            
        

        aNorm_fedNOVA[userIdx] = (tau_fedNOVA[userIdx] - ( (0.9 * (1. - pow(0.9, tau_fedNOVA[userIdx]))) / (1-0.9)) ) / (1-0.9)  # 0.9 is the considered value for the momentum
        
        computableParamNum = 0
        maxComputableLayers = int(users_max_slim_ratio[userIdx] * len(names))
        
        namesCopy = names.copy()
        
        for layerIdx in range(maxComputableLayers):
            
            selectedLayer = np.random.choice(namesCopy, 1, replace=False)
            
            namesCopy.remove(selectedLayer)
            
            selectedLayerParamSize = paramSize[names.index(selectedLayer)]
            
                
            computable_body_layers[userIdx].append(selectedLayer.item())
            computableParamNum += selectedLayerParamSize
                
        totParamNum += computableParamNum
        userCounter += 1
        
    
                 
    wandb.log({'Num_of_Params': totParamNum/userCounter}, commit=False)
    
    effTau_fedNOVA = sum([a*b for a,b in zip(aNorm_fedNOVA,fed.client_weights)])
            
                

    # /////////////////
    # //// Resume /////
    # /////////////////
    # log the best for each model on all datasets
    best_epoch = 0
    best_acc = [0. for j in range(fed.client_num)]
    train_elapsed = [[] for _ in range(fed.client_num)]
    start_epoch = 0
    if args.resume or args.test:
        if os.path.exists(SAVE_FILE):
            print(f'Loading chkpt from {SAVE_FILE}')
            checkpoint = torch.load(SAVE_FILE)
            best_epoch, best_acc = checkpoint['best_epoch'], checkpoint['best_acc']
            train_elapsed = checkpoint['train_elapsed']
            train_dataset = checkpoint['train_dataset']
            global_lr = checkpoint['lr']
            start_epoch = int(checkpoint['a_iter']) + 1
            fed.model_accum.load_state_dict(checkpoint['server_model'])

            print('Resume training from epoch {} with best acc:'.format(start_epoch))
            for client_idx, acc in enumerate(best_acc):
                print(' Best user-{:<10s}| Epoch:{} | Val Acc: {:.4f}'.format(
                    fed.clients[client_idx], best_epoch, acc))
        else:
            if args.test:
                raise FileNotFoundError(f"Not found checkpoint at {SAVE_FILE}")
            else:
                print(f"Not found checkpoint at {SAVE_FILE}\n **Continue without resume.**")


    # ///////////////
    # //// Test /////
    # ///////////////
    if args.test:
        wandb.summary[f'best_epoch'] = best_epoch

        # Set up model with specified width
        print(f"  Test model: {args.model}x{args.width_scale}")

        # Test on clients

            
        test_acc_list, _, test_acc_list_bp, _ = mask_fed_test(fed, running_model, train_dataset, test_loaders,
                                                                                 global_lr, args.verbose) 
            
        
        print(f"\n Average Test Acc Before Personalization: {np.mean(test_acc_list_bp)}")
        wandb.summary[f'avg test acc bp'] = np.mean(test_acc_list_bp)
        print(f"\n Average Test Acc: {np.mean(test_acc_list)}")
        wandb.summary[f'avg test acc'] = np.mean(test_acc_list)
        wandb.finish()

        exit(0)


    # ////////////////
    # //// Train /////
    # ////////////////
    # LR scheduler
    if args.lr_sch == 'cos':
        lr_sch = CosineAnnealingLR(args.iters, eta_max=args.lr, last_epoch=start_epoch)
    elif args.lr_sch == 'multi_step':
        lr_sch = MultiStepLR(args.lr, milestones=[args.iters//2, (args.iters * 3)//4], gamma=0.1, last_epoch=start_epoch)
    else:
        assert args.lr_sch == 'none', f'Invalid lr_sch: {args.lr_sch}'
        lr_sch = None
    for a_iter in range(start_epoch, args.iters):
        # set global lr
        global_lr = args.lr if lr_sch is None else lr_sch.step()
        wandb.log({'global lr': global_lr}, commit=False)
        

        # ----------- Train Client ---------------
        train_loss_mt, train_acc_mt = AverageMeter(), AverageMeter()
        print("============ Train epoch {} ============".format(a_iter))
        selectedUsers = []
        for client_idx in fed.client_sampler.iter():
            selectedUsers.append(client_idx)
            start_time = time.process_time()

            # Download global model to local
            fed.download(running_model, client_idx)
            
            

            if (users_max_comp[client_idx] == 1.0):
                local_iter = args.wk_iters
            else:
                local_iter = math.ceil(args.wk_iters*args.wk_factor)
                    
                    
            if (args.algorithm == 'fedNOVA'):
                
                local_lr = (effTau_fedNOVA/aNorm_fedNOVA[client_idx]) * global_lr
                
            else:
                
                local_lr = global_lr
                    
                    
                    
            
            if (args.algorithm == 'fedBABU'):
                optim_input = []
                
                for name, par in running_model.named_parameters():
                    
                    if name in computable_body_layers[client_idx]:
                        
                        par.requires_grad = True                
                        optim_input.append({'params': par, 'lr': local_lr})
                        
                    else:
                        
                        par.requires_grad = False
                        par.requires_grad_(False)
                        optim_input.append({'params': par, 'lr': 0.0})
                         

            if args.opt == 'sgd':
                
                if (args.algorithm == 'fedBABU'):
                    optimizer = optim.SGD(optim_input, momentum=0.9, weight_decay=5e-4)
                    
                else:
                    
                    optimizer = optim.SGD(params=running_model.parameters(), lr=local_lr,
                                          momentum=0.9, weight_decay=5e-4) 
                
            else:
                raise ValueError(f"Invalid optimizer: {args.opt}")
                
                
            if ((args.algorithm == 'fedProx') and (a_iter > 0)):
                
                train_loss, train_acc = train_fedprox(args.mu,
                    running_model, train_loaders[client_idx], optimizer, loss_fun, device,
                    max_iter=mean_batch_iters * local_iter if args.partition_mode != 'uni'
                                else len(train_loaders[client_idx]) * local_iter,
                    progress=args.verbose > 0
                )
       
            else:
            
                train_loss, train_acc = train(
                    running_model, train_loaders[client_idx], optimizer, loss_fun, device,
                    max_iter=mean_batch_iters * local_iter if args.partition_mode != 'uni'
                                else len(train_loaders[client_idx]) * local_iter,
                    progress=args.verbose > 0,
                )

            # Upload
            fed.mask_upload(running_model, client_idx, computable_body_layers[client_idx])

            # Log
            client_name = fed.clients[client_idx]
            elapsed = time.process_time() - start_time
            wandb.log({f'{client_name}_train_elapsed': elapsed}, commit=False)
            train_elapsed[client_idx].append(elapsed)

            train_loss_mt.append(train_loss), train_acc_mt.append(train_acc)
            print(f' User-{client_name:<10s} Train | Loss: {train_loss:.4f} |'
                  f' Acc: {train_acc:.4f} | Elapsed: {elapsed:.2f} s')
            wandb.log({
                f"{client_name} train_loss": train_loss,
                f"{client_name} train_acc": train_acc,
            }, commit=False)

        # Use accumulated model to update server model
        fed.aggregate()


        # ----------- Validation ---------------
        val_acc_list, val_loss, val_acc_list_bp, val_loss_bp = mask_fed_test(fed, running_model, train_loaders, 
                                                                                    val_loaders, global_lr, args.verbose)

        # Log averaged
        print(f' [Overall] Train Loss {train_loss_mt.avg:.4f} Acc {train_acc_mt.avg*100:.1f}%'
              f' | Val Acc bp {np.mean(val_acc_list_bp) * 100:.2f}%'
              f' | Val Acc {np.mean(val_acc_list) * 100:.2f}%')
        wandb.log({
            f"train_loss": train_loss_mt.avg,
            f"train_acc": train_acc_mt.avg,
            f"val_loss_bp": val_loss_bp,
            f"val_acc_bp": np.mean(val_acc_list_bp),
            f"val_loss": val_loss,
            f"val_acc": np.mean(val_acc_list),
        }, commit=False)

        # ----------- Save checkpoint -----------
        if np.mean(val_acc_list) > np.mean(best_acc):
            best_epoch = a_iter
            for client_idx in range(fed.client_num):
                best_acc[client_idx] = val_acc_list[client_idx]
                if args.verbose > 0:
                    print(' Best site-{:<10s}| Epoch:{} | Val Acc: {:.4f}'.format(
                          fed.clients[client_idx], best_epoch, best_acc[client_idx]))
            print(' [Best Val] Acc {:.4f}'.format(np.mean(val_acc_list)))

            # Save
            print(f' Saving the local and server checkpoint to {SAVE_FILE}')
            save_dict = {
                'server_model': fed.model_accum.state_dict(),
                'train_dataset': train_loaders,
                'lr' : global_lr,
                'best_epoch': best_epoch,
                'best_acc': best_acc,
                'a_iter': a_iter,
                'all_domains': fed.all_domains,
                'train_elapsed': train_elapsed,
            }
            torch.save(save_dict, SAVE_FILE)
        wandb.log({
            f"best_val_acc": np.mean(best_acc),
        }, commit=True)
