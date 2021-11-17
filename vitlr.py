""" Simple ViT implementation in PyTorch with Latent Replay """

# Python 2-3 compatible
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from data_loader import CORE50
import copy
import os
import json
from models.vit import MyViT
from utils import *
import configparser
import argparse
from pprint import pprint
from torch.utils.tensorboard import SummaryWriter

__global_ave_grads = {}
__global_max_grads = {}

def reset_grad_flow(net, __global_ave_grads, __global_max_grads):
    for n, p in net.named_parameters():
        __global_ave_grads[n] = []
        __global_max_grads[n] = []


def save_grad_flow(net):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    named_parameters = net.named_parameters()
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            if hasattr(p, 'grad'):
                if not p.grad is None:
                    ave_grads.append(p.grad.abs().mean().item())
                    max_grads.append(p.grad.abs().max().item())
        try:
            __global_ave_grads[n].extend(ave_grads)
            __global_max_grads[n].extend(max_grads)
        except KeyError:
            __global_ave_grads[n] = ave_grads
            __global_max_grads[n] = max_grads


# --------------------------------- Setup --------------------------------------

# recover exp configuration name
parser = argparse.ArgumentParser(description='Run CL experiments')
parser.add_argument('--name', dest='exp_name',  default='DEFAULT',
                    help='name of the experiment you want to run.')
args = parser.parse_args()

# set cuda device (based on your hardware)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# recover config file for the experiment
config = configparser.ConfigParser()
config.read("params.cfg")
exp_config = config[args.exp_name]
print("Experiment name:", args.exp_name)
pprint(dict(exp_config))

# recover parameters from the cfg file and compute the dependent ones.
exp_name = eval(exp_config['exp_name'])
comment = eval(exp_config['comment'])
use_cuda = eval(exp_config['use_cuda'])
init_lr = eval(exp_config['init_lr'])
inc_lr = eval(exp_config['inc_lr'])
mb_size = eval(exp_config['mb_size'])
init_train_ep = eval(exp_config['init_train_ep'])
inc_train_ep = eval(exp_config['inc_train_ep'])
init_update_rate = eval(exp_config['init_update_rate'])
inc_update_rate = eval(exp_config['inc_update_rate'])
max_r_max = eval(exp_config['max_r_max'])
max_d_max = eval(exp_config['max_d_max'])
inc_step = eval(exp_config['inc_step'])
rm_sz = eval(exp_config['rm_sz'])
momentum = eval(exp_config['momentum'])
l2 = eval(exp_config['l2'])
freeze_below_layer = eval(exp_config['freeze_below_layer'])
latent_layer_num = eval(exp_config['latent_layer_num'])
reg_lambda = eval(exp_config['reg_lambda'])

# setting up log dir for tensorboard
log_dir = 'logs/' + exp_name
writer = SummaryWriter(log_dir)

# Saving params
hyper = json.dumps(dict(exp_config))
writer.add_text("parameters", hyper, 0)

# Other variables init
tot_it_step = 0
rm = None

# Create the dataset object
dataset = CORE50(root='/home/adequino/work/ar1-pytorch/core50', scenario="nicv2_391")
preproc = preprocess_imgs

# Get the fixed test set
print("Loading Test Set...")
test_x, test_y = dataset.get_test_set()

# Model setup
model = MyViT(pretrained=True)
"""# we replace BN layers with Batch Renormalization layers
replace_bn_with_brn(
    model, momentum=init_update_rate, r_d_max_inc_step=inc_step,
    max_r_max=max_r_max, max_d_max=max_d_max
)"""
model.saved_weights = {}
model.past_j = {i:0 for i in range(50)}
model.cur_j = {i:0 for i in range(50)}
if reg_lambda != 0:
    # the regularization is based on Synaptic Intelligence as described in the
    # paper. ewcData is a list of two elements (best parametes, importance)
    # while synData is a dictionary with all the trajectory data needed by SI
    ewcData, synData = create_syn_data(model)
    
#print(model)

# Optimizer setup
optimizer = torch.optim.SGD(
    model.parameters(), lr=init_lr, momentum=momentum, weight_decay=l2
)
criterion = torch.nn.CrossEntropyLoss()

reset_grad_flow(model, __global_ave_grads, __global_max_grads)

# --------------------------------- Training -----------------------------------
if rm_sz <= 3000:
    preparation_loops = 0
else:
    preparation_loops = (rm_sz - 3000) // 300

first_batch_x = np.empty([0, 128, 128, 3], dtype=np.float32)
first_batch_y = np.empty([0], dtype=np.float32)

# loop over the training incremental batches
for i, train_batch in enumerate(dataset):

    if reg_lambda != 0:
        init_batch(model, ewcData, synData)

    # we freeze the layer below the replay layer since the first batch
    freeze_up_to(model, freeze_below_layer, only_conv=False)

    if i == (preparation_loops + 1):
        """change_brn_pars(
            model, momentum=inc_update_rate, r_d_max_inc_step=0,
            r_max=max_r_max, d_max=max_d_max)"""
        optimizer = torch.optim.SGD(
            model.parameters(), lr=inc_lr, momentum=momentum, weight_decay=l2
        )

    train_x, train_y = train_batch

    if i < preparation_loops:
        first_batch_x = np.concatenate((first_batch_x, train_x), 0)
        first_batch_y = np.concatenate((first_batch_y, train_y), 0)
        continue
    
    if i == preparation_loops:
        first_batch_x = np.concatenate((first_batch_x, train_x), 0)
        first_batch_y = np.concatenate((first_batch_y, train_y), 0)
        train_x = first_batch_x
        train_y = first_batch_y
    
    
    
    train_x = preproc(train_x)

    if i == preparation_loops:
        cur_class = [int(o) for o in set(train_y)]
        model.cur_j = examples_per_class(train_y)
    else:
        cur_class = [int(o) for o in set(train_y).union(set(rm[1]))]
        model.cur_j = examples_per_class(list(train_y) + list(rm[1]))

    print("----------- batch {0} -------------".format(i))
    print("train_x shape: {}, train_y shape: {}"
          .format(train_x.shape, train_y.shape))

    model.train()
    model.start_features.eval()
    model.lat_features.eval()

    reset_weights(model, cur_class)
    cur_ep = 0

    if i == preparation_loops:
        (train_x, train_y), it_x_ep = pad_data([train_x, train_y], mb_size)
    shuffle_in_unison([train_x, train_y], in_place=True)

    model = maybe_cuda(model, use_cuda=use_cuda)
    acc = None
    ave_loss = 0

    train_x = torch.from_numpy(train_x).type(torch.FloatTensor)
    train_y = torch.from_numpy(train_y).type(torch.LongTensor)

    if i == preparation_loops:
        train_ep = init_train_ep
    else:
        train_ep = inc_train_ep

    for ep in range(train_ep):

        print("training ep: ", ep)
        correct_cnt, ave_loss = 0, 0

        # computing how many patterns to inject in the latent replay layer
        if i > preparation_loops:
            #print("train_x.size(0):",train_x.size(0))
            #print("mb_size:", mb_size)
            #print("rm_sz:", rm_sz)
            cur_sz = train_x.size(0) // ((train_x.size(0) + rm_sz) // mb_size)
            #print("cur_sz:", cur_sz)
            it_x_ep = train_x.size(0) // cur_sz
            n2inject = max(0, mb_size - cur_sz)
        else:
            n2inject = 0
        print("train size:", train_x.size(0))
        print("remote memory size:", rm_sz)
        print("total sz:", train_x.size(0) + rm_sz)
        print("n2inject", n2inject)
        print("it x ep: ", it_x_ep)

        for it in range(it_x_ep):

            if reg_lambda !=0:
                pre_update(model, synData)

            start = it * (mb_size - n2inject)
            end = (it + 1) * (mb_size - n2inject)

            optimizer.zero_grad()

            x_mb = maybe_cuda(train_x[start:end], use_cuda=use_cuda)

            if i == preparation_loops:
                lat_mb_x = None
                y_mb = maybe_cuda(train_y[start:end], use_cuda=use_cuda)

            else:
                lat_mb_x = rm[0][it*n2inject: (it + 1)*n2inject]
                lat_mb_y = rm[1][it*n2inject: (it + 1)*n2inject]
                y_mb = maybe_cuda(
                    torch.cat((train_y[start:end], lat_mb_y), 0),
                    use_cuda=use_cuda)
                lat_mb_x = maybe_cuda(lat_mb_x, use_cuda=use_cuda)

            # if lat_mb_x is not None, this tensor will be concatenated in
            # the forward pass on-the-fly in the latent replay layer
            logits, lat_acts = model(
                x_mb, latent_input=lat_mb_x, return_lat_acts=True)

            # collect latent volumes only for the first ep
            # we need to store them to eventually add them into the external
            # replay memory
            if ep == 0:
                lat_acts = lat_acts.cpu().detach()
                if it == 0:
                    cur_acts = copy.deepcopy(lat_acts)
                else:
                    cur_acts = torch.cat((cur_acts, lat_acts), 0)

            _, pred_label = torch.max(logits, 1)
            correct_cnt += (pred_label == y_mb).sum()

            loss = criterion(logits, y_mb)
            if reg_lambda !=0:
                loss += compute_ewc_loss(model, ewcData, lambd=reg_lambda)
            ave_loss += loss.item()

            loss.backward()
            save_grad_flow(model)
            optimizer.step()

            if reg_lambda !=0:
                post_update(model, synData)

            acc = correct_cnt.item() / \
                  ((it + 1) * y_mb.size(0))
            ave_loss /= ((it + 1) * y_mb.size(0))

            if it % 10 == 0:
                print(
                    '==>>> it: {}, avg. loss: {:.6f}, '
                    'running train acc: {:.3f}'
                        .format(it, ave_loss, acc)
                )

            # Log scalar values (scalar summary) to TB
            tot_it_step +=1
            writer.add_scalar('train_loss', ave_loss, tot_it_step)
            writer.add_scalar('train_accuracy', acc, tot_it_step)

        cur_ep += 1

    consolidate_weights(model, cur_class)
    if reg_lambda != 0:
        update_ewc_data(model, ewcData, synData, 0.001, 1)

    # how many patterns to save for next iter
    h = min(rm_sz // (i - preparation_loops + 1), cur_acts.size(0))
    print("h", h)

    print("cur_acts sz:", cur_acts.size(0))
    idxs_cur = np.random.choice(
        cur_acts.size(0), h, replace=False
    )
    rm_add = [cur_acts[idxs_cur], train_y[idxs_cur]]
    print("rm_add size", rm_add[0].size(0))
    print("Pattern size:", cur_acts[0].shape)

    # replace patterns in random memory
    if i == preparation_loops:
        rm = copy.deepcopy(rm_add)
    else:
        idxs_2_replace = np.random.choice(
            rm[0].size(0), h, replace=False
        )
        for j, idx in enumerate(idxs_2_replace):
            rm[0][idx] = copy.deepcopy(rm_add[0][j])
            rm[1][idx] = copy.deepcopy(rm_add[1][j])

    set_consolidate_weights(model)
    ave_loss, acc, accs = get_accuracy(
        model, criterion, mb_size, test_x, test_y, preproc=preproc
    )
    
    labelmap = [0] * 50
    for label in rm[1]:
        labelmap[label.item()] += 1
    
    #for n, label in enumerate(labelmap):
        #print("Label",n,":",label)
        
    

    #print("Accs shape:", accs.shape)
    # Log scalar values (scalar summary) to TB
    writer.add_scalar('test_loss', ave_loss, i)
    writer.add_scalar('test_accuracy', acc, i)
    for n,p in model.named_parameters():
        if(p.requires_grad) and ("bias" not in n): 
            writer.add_scalar(f'{n}'+'/avg_grad', np.mean(np.asarray(__global_ave_grads[n])), i)
            writer.add_scalar(f'{n}'+'/max_grad', np.max(np.asarray(__global_max_grads[n])), i)
    reset_grad_flow(model, __global_ave_grads, __global_max_grads)
    
    for n, class_acc in enumerate(accs):
        writer.add_scalar(f'{n}'+'/class accuracy', class_acc, i)
    
    for n, rm_count in enumerate(labelmap):
        writer.add_scalar(f'{n}'+'/rm count', rm_count, i)


    # update number examples encountered over time
    for c, n in model.cur_j.items():
        model.past_j[c] += n

    print("---------------------------------")
    print("Accuracy: ", acc)
    print("---------------------------------")

writer.close()

torch.save(model.state_dict(), "state_dict_model.pt") 
