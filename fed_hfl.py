"""HeteroFL"""
import os, argparse, time
import numpy as np
import wandb
import torch
import copy
from torch import nn, optim
# federated
from federated.learning import train_slimmable, test, personalization
# utils
from utils.utils import set_seed, AverageMeter, CosineAnnealingLR, \
    MultiStepLR, str2bool
from nets.profile_func import profile_model
from utils.config import CHECKPOINT_ROOT

from federated.core import HeteFederation as Federation



def render_run_name(args, exp_folder):
    """Return a unique run_name from given args."""
    if args.model == 'default':
        args.model = {'Digits': 'digit', 'Cifar10': 'preresnet18', 'Cifar100': 'mobile', 'DomainNet': 'alex'}[args.data]
    run_name = f'{args.model}'
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
    # split-mix
    if not args.rescale_init: run_name += '__nri'
    if not args.rescale_layer: run_name += '__nrl'
    if args.loss_temp != 'none': run_name += f'__lt{args.loss_temp}'

    args.save_path = os.path.join(CHECKPOINT_ROOT, exp_folder)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    SAVE_FILE = os.path.join(args.save_path, run_name)
    return run_name, SAVE_FILE


def get_model_fh(data, model):
    if data == 'Digits':
        if model in ['digit']:
            from nets.slimmable_models import SlimmableDigitModel
            # TODO remove. Function the same as ens_digit
            ModelClass = SlimmableDigitModel
        else:
            raise ValueError(f"Invalid model: {model}")
    elif data in ['DomainNet']:
        if model in ['alex']:
            from nets.slimmable_models import SlimmableAlexNet
            ModelClass = SlimmableAlexNet
        else:
            raise ValueError(f"Invalid model: {model}")
    elif data == 'Cifar10':
        if model in ['preresnet18']:  # From heteroFL
            from nets.HeteFL.slimmable_preresne import resnet18
            ModelClass = resnet18
        else:
            raise ValueError(f"Invalid model: {model}")
    elif data == 'Cifar100':
        if model in ['mobile']:
            from nets.slimmable_Nets import MobileNetCifar
            ModelClass = MobileNetCifar
        else:
            raise ValueError(f"Invalid model: {model}")
    else:
        raise ValueError(f"Unknown dataset: {data}")
    return ModelClass


def fed_test(fed, running_model, train_loaders, val_loaders, global_lr, verbose):
    mark = 's'
    val_acc_list_bp = [None for _ in range(fed.client_num)]
    val_loss_mt_bp = AverageMeter()
    val_acc_list = [None for _ in range(fed.client_num)]
    val_loss_mt = AverageMeter()
    slim_val_acc_bp_mt = {slim_ratio: AverageMeter() for slim_ratio in fed.val_slim_ratios}
    slim_val_acc_mt = {slim_ratio: AverageMeter() for slim_ratio in fed.val_slim_ratios}
    for client_idx in range(fed.client_num):
        fed.download(running_model, client_idx)

        for i_slim_ratio, slim_ratio in enumerate(fed.val_slim_ratios):
            # Load and set slim ratio

            running_model.switch_slim_mode(slim_ratio)
  

            # Test

            val_model = copy.deepcopy(running_model)
            # Loss and accuracy before personalization
            val_loss_bp, val_acc_bp = test(val_model, val_loaders[client_idx], loss_fun, device) 
                
            # Log
            val_loss_mt_bp.append(val_loss_bp)
            val_acc_list_bp[client_idx] = val_acc_bp
            if verbose > 0:
                print(' {:<19s} slim {:.2f}| Val Before Personalization {:s}Loss: {:.4f} | Val {:s}Acc: {:.4f}'.format(
                'User-' + fed.clients[client_idx] if i_slim_ratio == 0 else ' ', slim_ratio,
                mark.upper(), val_loss_bp, mark.upper(), val_acc_bp))
                wandb.log({
                f"{fed.clients[client_idx]} sm{slim_ratio:.2f} val_bp_s-acc": val_acc_bp,
                }, commit=False)
                if slim_ratio == fed.user_max_slim_ratios[client_idx]:
                    wandb.log({
                    f"{fed.clients[client_idx]} val_bp_{mark}-acc": val_acc_bp,
                    }, commit=False)
                slim_val_acc_bp_mt[slim_ratio].append(val_acc_bp)
    
            if args.test:  
          
                # Loss and accuracy after personalization
                val_loss, val_acc = personalization(val_model, train_loaders[client_idx], val_loaders[client_idx], 
                                                    loss_fun, global_lr, device)
                
    
                # Log
                val_loss_mt.append(val_loss)
                val_acc_list[client_idx] = val_acc # NOTE only record the last slim_ratio.
                if verbose > 0:
                    print(' {:<19s} slim {:.2f}| Val {:s}Loss: {:.4f} | Val {:s}Acc: {:.4f}'.format(
                    'User-' + fed.clients[client_idx] if i_slim_ratio == 0 else ' ', slim_ratio,
                    mark.upper(), val_loss, mark.upper(), val_acc))
                    if slim_ratio == fed.user_max_slim_ratios[client_idx]:
                        wandb.log({
                            f"{fed.clients[client_idx]} val_{mark}-acc": val_acc,
                            }, commit=False)
                    slim_val_acc_mt[slim_ratio].append(val_acc)
                    
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
    parser.add_argument('--data', type=str, default='Cifar10', help='data name') # 'DomainNet'  'Cifar100'
    parser.add_argument('--model', type=str.lower, default='default', help='model name')
    parser.add_argument('--algorithm', type=str, default='HFL', help='algorithm name') 
    parser.add_argument('--no_track_stat', action='store_true', help='disable BN tracking')
    # control
    parser.add_argument('--no_log', action='store_true', help='disable wandb log')
    parser.add_argument('--test', action='store_true', help='test the pretrained model')
    #parser.add_argument('--test', type=str2bool, default=True, help='test the pretrained model') #action='store_true'
    parser.add_argument('--resume', action='store_true', help='resume training from checkpoint')
    parser.add_argument('--verbose', type=int, default=0, help='verbose level: 0 or 1')
    # federated
    Federation.add_argument(parser)
    # optimization
    parser.add_argument('--lr', type=float, default=1e-1, help='learning rate') #1e-2 1e-1
    parser.add_argument('--lr_sch', type=str, default='multi_step', help='learning rate schedule') #  'cos'  'none'
    parser.add_argument('--opt', type=str.lower, default='sgd', help='optimizer')
    parser.add_argument('--iters', type=int, default=80, help='#iterations for communication')#200
    parser.add_argument('--wk_iters', type=int, default=4, help='#epochs in local train')#5 1 
    # slimmable test
    parser.add_argument('--test_slim_ratio', type=float, default=1.,
                        help='slim_ratio of model at testing.')
    # split-mix
    parser.add_argument('--rescale_init', type=str2bool, default=True, help='rescale init after'
                                                                            ' slim')
    parser.add_argument('--rescale_layer', type=str2bool, default=True, help='rescale layer outputs'
                                                                             ' after slim')
    parser.add_argument('--loss_temp', type=str, default='none', choices=['none', 'auto'],
                        help='temper cross-entropy loss. auto: set temp as the width scale.')
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
    wandb.init(group=run_name[:120], project=exp_folder, mode='offline' if args.no_log else 'online',
               config={**vars(args), 'save_file': SAVE_FILE})

    # Model
    ModelClass = get_model_fh(args.data, args.model)
    running_model = ModelClass(
        track_running_stats=False,
        num_classes=fed.num_classes, slimmabe_ratios=fed.train_slim_ratios,
    ).to(device)


    # Loss
    loss_fun = nn.CrossEntropyLoss()

    # Use running model to init a fed aggregator
    fed.make_aggregator(running_model)
    
    
    
    totParamNum = 0
    userCounter = 0
    for userIdx in range(fed.client_sampler.tot()):
        slim_ratios, slim_shifts = fed.sample_bases(userIdx)
        temp_model = copy.deepcopy(running_model)
        fed.download(temp_model, userIdx)
        temp_model.switch_slim_mode(slim_ratios[0], slim_bias_idx=slim_shifts[0], out_slim_bias_idx=None) #
        _, computableParamNum = profile_model(temp_model, device=device)
        
        totParamNum += computableParamNum
        userCounter += 1
                
                
    wandb.log({'Num_of_Params': totParamNum/userCounter}, commit=False)


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
    
        
        test_acc_list, _, test_acc_list_bp, _ = fed_test(fed, running_model, train_dataset, 
                                                                test_loaders, global_lr, args.verbose)

        
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
        for client_idx in fed.client_sampler.iter():
            # (Alg 2) Sample base models defined by shift index.
            slim_ratios, slim_shifts = fed.sample_bases(client_idx)

            start_time = time.process_time()

            # Download global model to local
            fed.download(running_model, client_idx)

            # (Alg 3) Local Train
            if args.opt == 'sgd':
                optimizer = optim.SGD(params=running_model.parameters(), lr=global_lr,
                                      momentum=0.9, weight_decay=5e-4)
            elif args.opt == 'adam':
                optimizer = optim.Adam(params=running_model.parameters(), lr=global_lr)
            else:
                raise ValueError(f"Invalid optimizer: {args.opt}")
            train_loss, train_acc = train_slimmable(
                running_model, train_loaders[client_idx], optimizer, loss_fun, device,
                max_iter=mean_batch_iters * args.wk_iters if args.partition_mode != 'uni'
                            else len(train_loaders[client_idx]) * args.wk_iters,
                slim_ratios=slim_ratios, slim_shifts=slim_shifts, progress=args.verbose > 0,
                loss_temp=args.loss_temp
            )

            # Upload
            fed.upload(running_model, client_idx,
                       max_slim_ratio=max(slim_ratios), slim_bias_idx=slim_shifts)

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
        val_acc_list, val_loss, val_acc_list_bp, val_loss_bp = fed_test(fed, running_model, train_loaders, val_loaders, 
                                                    global_lr, args.verbose)
        
        # Log averaged
        print(f' [Overall] Train Loss {train_loss_mt.avg:.4f} Acc {train_acc_mt.avg*100:.1f}% '
              f'| Val Acc bp {np.mean(val_acc_list_bp)*100:.2f}%'
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
