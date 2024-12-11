import os
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import json
import clip


from dataset import dataset_TM_eval_bodypart

import models.vqvae_bodypart as vqvae_bodypart
import models.t2m_trans_bodypart_prob_bi as trans_bodypart

from options.get_eval_option import get_opt
from models.evaluator_wrapper import EvaluatorModelWrapper

import utils.utils_model as utils_model
import utils.eval_bodypart_bi as eval_bodypart
from utils.word_vectorizer import WordVectorizer
from utils.misc import EasyDict

import argparse
import warnings
warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser(description='Evaluate the real motion',
                                 add_help=True,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--eval-exp-dir', type=str, help='The trained transformer experiment directory to be evaluated.')
parser.add_argument("--skip-mmod", action='store_true', help="Skip evaluating MultiModality")
parser.add_argument('--select-ckpt', type=str, help='Select which ckpt for use: [last, fid, div, top1, matching]')
parser.add_argument("--fixed-seed", action='store_true', help="Use the same seed used in training, otherwise set the seed randomly")
parser.add_argument("--skip-path-check", action='store_true', help="Skip check of path consistency")


test_args = parser.parse_args()

assert test_args.select_ckpt in ['last', 'fid', 'div', 'top1', 'matching']

eval_exp_dir = test_args.eval_exp_dir
select_ckpt = test_args.select_ckpt
skip_mmod = test_args.skip_mmod
fixed_seed = test_args.fixed_seed
skip_path_check = test_args.skip_path_check

if skip_mmod:
    print('\n\nSkip evaluating MultiModality\n\n')
else:
    print('\n\nEvaluate MultiModality 5 times. (MDM: 5 times. T2M and T2M-GPT: 20 times). \n\n')


assert select_ckpt in [
    'last',  # last  saved ckpt
    'fid',  # best FID ckpt
    'div',  # best diversity ckpt
    'top1',  # best top-1 R-precision
    'matching',  # MM-Dist: Multimodal Distance
]


trans_config_path = os.path.join(eval_exp_dir, 'train_config.json')


# Checkpoint path
if select_ckpt == 'last':
    trans_ckpt_path = os.path.join(eval_exp_dir, 'net_' + select_ckpt + '.pth')
else:
    trans_ckpt_path = os.path.join(eval_exp_dir, 'net_best_' + select_ckpt + '.pth')



with open(trans_config_path, 'r') as f:
    trans_config_dict = json.load(f)  # dict
args = EasyDict(trans_config_dict)

vqvae_train_args = EasyDict(args.vqvae_train_args)

if fixed_seed:
    torch.manual_seed(args.seed)
    test_args.seed = args.seed
else:
    random_seed = torch.randint(0,256,[])
    test_args.seed = int(random_seed)
    print('random_seed:', random_seed)
    torch.manual_seed(random_seed)


if skip_path_check:
    print('\n Skip check of path consistency\n')
else:
    print('\n Checking path consistency...\n')
    print(eval_exp_dir)
    print(args.run_dir)
    #assert os.path.samefile(eval_exp_dir, args.run_dir)


##### ---- Logger ---- #####





w_vectorizer = WordVectorizer('./glove', 'our_vab')

val_loader = dataset_TM_eval_bodypart.DATALoader(args.dataname, True, 32, w_vectorizer)

dataset_opt_path = 'checkpoints/kit/Comp_v6_KLD005/opt.txt' if args.dataname == 'kit' else 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'

wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

##### ---- Network ---- #####

## load clip model and datasets
clip_model, clip_preprocess = clip.load("ViT-B/32", device=torch.device('cuda'), jit=False)  # Must set jit=False for training
clip.model.convert_weights(clip_model)  # Actually this line is unnecessary since clip by default already on float16
clip_model.eval()
for p in clip_model.parameters():
    p.requires_grad = False



print('constructing vqvae')
net = vqvae_bodypart.HumanVQVAEBodyPart(
    vqvae_train_args,  # use args to define different parameters in different quantizers
    vqvae_train_args['vqvae_arch_cfg']['parts_code_nb'],
    vqvae_train_args['vqvae_arch_cfg']['parts_code_dim'],
    vqvae_train_args['vqvae_arch_cfg']['parts_output_dim'],
    vqvae_train_args['vqvae_arch_cfg']['parts_hidden_dim'],
    vqvae_train_args['down_t'],
    vqvae_train_args['stride_t'],
    vqvae_train_args['depth'],
    vqvae_train_args['dilation_growth_rate'],
    vqvae_train_args['vq_act'],
    vqvae_train_args['vq_norm']
)

print('loading checkpoint from {}'.format(args.vqvae_ckpt_path))
ckpt = torch.load(args.vqvae_ckpt_path, map_location='cpu')
net.load_state_dict(ckpt['net'], strict=True)
net.eval()
net.cuda()



trans_encoder = trans_bodypart.TransformerFuseHiddenDim(

    clip_dim=args.clip_dim,
    block_size=args.block_size,
    num_layers=args.num_layers,
    n_head=args.n_head_gpt,
    drop_out_rate=args.drop_out_rate,
    fc_rate=args.ff_rate,

    # FusionModule
    use_fuse=args.use_fuse,
    fuse_ver=args.fuse_ver,
    alpha=args.alpha,

    parts_code_nb=args.trans_arch_cfg['parts_code_nb'],
    parts_embed_dim=args.trans_arch_cfg['parts_embed_dim'],
    num_mlp_layers=args.trans_arch_cfg['num_mlp_layers'],
    fusev2_sub_mlp_out_features=args.trans_arch_cfg['fusev2_sub_mlp_out_features'],
    fusev2_sub_mlp_num_layers=args.trans_arch_cfg['fusev2_sub_mlp_num_layers'],
    fusev2_head_mlp_num_layers=args.trans_arch_cfg['fusev2_head_mlp_num_layers'],

)


print('loading transformer checkpoint from {}'.format(trans_ckpt_path))
ckpt = torch.load(trans_ckpt_path, map_location='cpu')
trans_encoder.load_state_dict(ckpt['trans'], strict=True)
trans_encoder.train()
trans_encoder.cuda()

print(net.apply)
print(trans_encoder.apply)
