"""PerFedMask"""
import os, argparse, time
import numpy as np
import wandb
import copy
import torch
import math
import cvxpy as cp
from torch import nn, optim
# federated
from federated.learning import train, test, personalization, train_fedprox
# utils
from utils.utils import set_seed, AverageMeter, CosineAnnealingLR, MultiStepLR, str2bool
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


def layerWiseModelPartition(dataset, names, paramSize):
    
    if dataset == 'Cifar100':
        
        layersNum = 14
        LayerParams = np.zeros(layersNum)
        
        for layerIdx in range(len(names)):
            
            if ((layerIdx==0) or (layerIdx==1) or (layerIdx==2)):
                LayerParams[0] += paramSize[layerIdx]
                
            else:
                selectedLayer = np.array([names[layerIdx]])
                LayerParams[int(selectedLayer[0].split('.')[1])+1] += paramSize[layerIdx]
                
                
                
        return  layersNum, LayerParams

    elif dataset == 'Cifar10':

        layersNum = 10
        LayerParams = np.zeros(layersNum)
        
        for layerIdx in range(len(names)):
            
            if (layerIdx==0):
                LayerParams[0] += paramSize[layerIdx]
                
            elif ((layerIdx==len(names)-2) or (layerIdx==len(names)-1)):
                LayerParams[layersNum-1] += paramSize[layerIdx]
                
            else:
                selectedLayer = np.array([names[layerIdx]])
                LayerParams[int(selectedLayer[0].split('.')[0][-1])*2 + int(selectedLayer[0].split('.')[1]) - 1] += paramSize[layerIdx]
                
        
        return  layersNum, LayerParams
    
    
        
    elif dataset in ['DomainNet']:
        layersNum = 7
        LayerParams = np.zeros(layersNum)
        
        for layerIdx in range(len(names)):
            
            LayerParams[int(layerIdx//4)] += paramSize[layerIdx]
                
        
        return  layersNum, LayerParams
    
    
    else: 
        raise ValueError(f"Unhandled dataset: {dataset}")
        
        
def expand_subLayer_Mask(dataset, N_users, LayerMaskVec, layersNum, names, paramSize):
    
    computable_body_layers = {userIdx: [] for userIdx in range(N_users)}
    totParamNum = 0

     
    for userIdx in range(N_users):
        
        
        computableParamNum = 0
        
        namesCopy = names.copy()
        
        for layerIdx in range(layersNum):
            
            
            if (LayerMaskVec[userIdx, layerIdx] >= 0.5):
                
                if dataset == 'Cifar100':
                    
                    if (layerIdx==0):
                        
                        for subLayerIdx in range(3):
                    
                            subLayerNames = np.array([names[subLayerIdx]])
                            selectedLayerParamSize = paramSize[names.index(subLayerNames)]
                            computable_body_layers[userIdx].append(subLayerNames.item())
                            computableParamNum += selectedLayerParamSize
                        
                        
                    else:
                        
                        
                        for subLayerNames in names:
                            
                            if 'layers.'+str(layerIdx-1)+'.' in subLayerNames:
                        
                                selectedLayerParamSize = paramSize[names.index(subLayerNames)]
                                computable_body_layers[userIdx].append(subLayerNames)
                                computableParamNum += selectedLayerParamSize
                                
                                
                elif dataset == 'Cifar10':
                    
                    if (layerIdx==0):
                        
                            subLayerNames = np.array([names[0]])
                            selectedLayerParamSize = paramSize[names.index(subLayerNames)]
                            computable_body_layers[userIdx].append(subLayerNames.item())
                            computableParamNum += selectedLayerParamSize
                        
                        
                    elif (layerIdx == layersNum-1):
                          
                        for subLayerIdx in [len(names)-2, len(names)-1]:
                          
                            subLayerNames = np.array([names[subLayerIdx]])
                            selectedLayerParamSize = paramSize[names.index(subLayerNames)]
                            computable_body_layers[userIdx].append(subLayerNames.item())
                            computableParamNum += selectedLayerParamSize

                          
                    else:
                        
                        for subLayerNames in names:
                        
                            if 'layer'+str(int(layerIdx+1)//2)+'.'+str(int(layerIdx+1)%2) in subLayerNames:
                                
                                selectedLayerParamSize = paramSize[names.index(subLayerNames)]
                                computable_body_layers[userIdx].append(subLayerNames)
                                computableParamNum += selectedLayerParamSize
                                

                    
                    
                elif dataset in ['DomainNet']:
                    
                    for subLayerIdx in range(layerIdx*4,(layerIdx+1)*4):
                        
                        subLayerNames = np.array([names[subLayerIdx]])
                        selectedLayerParamSize = paramSize[names.index(subLayerNames)]
                        computable_body_layers[userIdx].append(subLayerNames.item())
                        computableParamNum += selectedLayerParamSize
                    
                    
                    
                else:
                    raise ValueError(f"Unhandled dataset: {dataset}")   

                
        totParamNum += computableParamNum
    
    return computable_body_layers, totParamNum
        
        
def optMask(N_users, layersNum, users_max_comp, totalParamNum, LayerParams):
    
    # Initialize the making vectors with a feasible solution
    LayerMaskVec_old, K_Vec_old = varInit(N_users, layersNum, users_max_comp, totalParamNum, LayerParams)
    
    iterMax = 100
    
    # SCA Algorithm
    for optIter in range(iterMax):
    
    
        LayerMaskVec = cp.Variable((N_users, layersNum)) #, boolean = True
        K_Vec = cp.Variable((N_users, layersNum))
        epsilonPerUser = cp.Variable(N_users)
        t = cp.Variable(1)
        
        constraints = []; 
        
        
        for useridx in range(N_users):
            
            constraints.append( cp.sum(cp.multiply(LayerParams, LayerMaskVec[useridx, :])) == users_max_comp[useridx] * totalParamNum - epsilonPerUser[useridx] )
            
            for layerIdx in range(layersNum):
                
                constraints.append( K_Vec[useridx, layerIdx] >= cp.quad_over_lin(LayerMaskVec[useridx, layerIdx], cp.sum(LayerMaskVec[:,layerIdx])) )
                constraints.append( 0.5* cp.power(K_Vec[useridx, layerIdx] + cp.sum(LayerMaskVec[:,layerIdx]) , 2) -
                                   0.5*np.power(K_Vec_old[useridx, layerIdx] , 2) -
                                   K_Vec_old[useridx, layerIdx] * (K_Vec[useridx, layerIdx]-K_Vec_old[useridx, layerIdx])-
                                   0.5*np.power(np.sum(LayerMaskVec_old[:,layerIdx]) , 2) -
                                   np.sum(LayerMaskVec_old[:,layerIdx]) * (cp.sum(LayerMaskVec[:,layerIdx]) - np.sum(LayerMaskVec_old[:,layerIdx])) <=
                                   LayerMaskVec[useridx,layerIdx]  )
                
        constraints.append( cp.sum(LayerMaskVec)-2*cp.sum(cp.multiply(LayerMaskVec_old, LayerMaskVec-LayerMaskVec_old)) - t <=0 )
        constraints.append(epsilonPerUser >=0 )
        constraints.append(LayerMaskVec >=0 )
        constraints.append(LayerMaskVec <=1 )
        constraints.append(t >= 0 )
                
        
        objective = cp.Minimize( cp.sum(totalParamNum * cp.max(K_Vec, axis=1) -  K_Vec @ LayerParams + epsilonPerUser) + 100*cp.power(t,2)  ) 
        
        prob = cp.Problem(objective, constraints)
        
        try:
            prob.solve(solver=cp.MOSEK, verbose=False, mosek_params={'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': 0.1}) #
            LayerMaskVec_new = LayerMaskVec.value
        except:
            LayerMaskVec_new = LayerMaskVec_old
        
        
        if (np.linalg.norm(LayerMaskVec_new-LayerMaskVec_old)<=1e-3):
            break
        
        LayerMaskVec_old = LayerMaskVec_new
        LayerK_denom = np.tile(np.sum(LayerMaskVec_old, axis=0), (N_users,1))   
        K_Vec_old = np.divide(LayerMaskVec_old, LayerK_denom, where=LayerK_denom!=0)
    
    
    return LayerMaskVec_old, K_Vec_old 


def varInit(N_users, layersNum, users_max_comp, totalParamNum, LayerParams):
    
    bestM = np.random.rand(N_users, layersNum)
    bestK = np.random.rand(N_users, layersNum)
    minPenObjVal = 1e12   
    
    for ii in range(1000):

        LayerMaskVec = np.zeros((N_users, layersNum))
        totEpsilon = 0
        for userIdx in range(N_users):

            computableParamNum = 0
            maxComputableParamNum = users_max_comp[userIdx] * totalParamNum
    
                
            layerIdxCopy = [*range(layersNum)]
            
            for layerIdx in range(layersNum):
                
                if (len(layerIdxCopy)==0):
                    break
                
                if (ii>0):
                    # ii > 0 is the random masking
                    slectedIdx = np.random.choice(layerIdxCopy, 1, replace=False)
                else:
                    # ii == 0 is the sequantial masking
                    slectedIdx = np.array([layerIdxCopy[0]])
                    
                    
                LayerMaskVec[userIdx, slectedIdx] = 1

                    
                layerIdxCopy.remove(slectedIdx)
                selectedLayerParamSize = LayerParams[slectedIdx]
                
                computableParamNum += selectedLayerParamSize
        
                if (computableParamNum > maxComputableParamNum):
                    LayerMaskVec[userIdx, slectedIdx] = 0
                    computableParamNum -= selectedLayerParamSize
                    if (ii == 0):
                        totEpsilon +=  maxComputableParamNum - computableParamNum
                        break
                    
                if (layerIdx==layersNum-1):
                    totEpsilon +=  maxComputableParamNum - computableParamNum

            
        LayerK_denom = np.tile(np.sum(LayerMaskVec, axis=0), (N_users,1))   
        LayerK_Vec = np.divide(LayerMaskVec, LayerK_denom, where=LayerK_denom!=0)
            
        LayerObjVal = 0.0
        
        for userIdx in range(fed.client_sampler.tot()):
            
            LayerObjVal = LayerObjVal + totalParamNum * np.max(LayerK_Vec[userIdx,:]) - np.sum(LayerK_Vec[userIdx,:] * LayerParams)
            
        PenObjVal = LayerObjVal+totEpsilon
        
        
        if (minPenObjVal>PenObjVal) and (len(np.where(np.sum(LayerMaskVec,axis=1) == 0)[0])==0):
            minPenObjVal = PenObjVal
            bestM = LayerMaskVec
            bestK = LayerK_Vec
    

    return bestM, bestK
    
    



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
    parser.add_argument('--data', type=str, default='Cifar10', help='data name') #   'Cifar100' 'DomainNet'
    parser.add_argument('--model', type=str.lower, default='default', help='model name')
    parser.add_argument('--algorithm', type=str, default='perFedMask', help='algorithm name') 
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
        
            
        if head_part not in name:
            
            names.append(name)
            paramSize.append(np.prod(list(par.size())))

            
        
    
    totalParamNum = sum(paramSize)  
    users_max_comp = fed.get_user_max_slim_ratios()
    
    
    
    layersNum, LayerParams = layerWiseModelPartition(args.data, names, paramSize) 
    
    LayerMaskVec, K_Vec = optMask(fed.client_sampler.tot(), layersNum, users_max_comp, totalParamNum, LayerParams)
    
    
    computable_body_layers, totParamNum = expand_subLayer_Mask(args.data, fed.client_sampler.tot(), LayerMaskVec, layersNum, names, paramSize)
     
    
                 
    wandb.log({'Num_of_Params': totParamNum/fed.client_sampler.tot()}, commit=False)
    
            
                

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
            
            

            local_iter = args.wk_iters
            local_lr = global_lr
                    


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
                
                
                optimizer = optim.SGD(optim_input, momentum=0.9, weight_decay=5e-4)
 
                
            else:
                raise ValueError(f"Invalid optimizer: {args.opt}")
                
                
            
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
