
import numpy as np

from utils.environ import env
from utils.helpers import *
from utils.db import SqclDB
from utils.mlogging import mlogging
from utils.exception import ExceptiontContainer
from utils.batch import iter_block_batches

from models.cagnir import CAGNIR, CAGNIRConfig

import os
from math import ceil
import json



def train_expt(model_id, expt_mode, model_config, note=None, to_log=True, **expt_params):

    if to_log:
        log_expt_params('cagnir', expt_mode, model_id, {**model_config.asdict(), **expt_params}, note, line_max_len=120)

    train(model_id=model_id, model_config=model_config, to_log=to_log, **expt_params)

def train(model_id, load_model, model_config, attn_drop, ft_drop, bn_training, log_tfvars, log_freqs, max_step, early_stop_threshold,
          block_repo_name, epochs, val_ratio, adj_type, self_attn, sample_weight_type=None, batch_num=None, block_seed=18035, to_log=True):

    logger = mlogging.get_logger(('cagnir/%s.train.log'%model_id if to_log else None), prefix='CAGNIR')

    logger.info('model id: %s (train)'%model_id)

    model = CAGNIR(model_id, load_model=load_model, model_config=model_config, logger=logger, log_tfvars=log_tfvars)


    block_repo_path = relpath('batch/%s'%block_repo_name, env('SQCL_DIR'))

    np.random.seed(block_seed)
    if os.path.exists(block_repo_path+'/sub_repo.0'):  # 檢查是否包含sub_repo
        block_seqs = np.random.permutation(len(os.listdir(block_repo_path+'/sub_repo.0')))
    else:
        block_seqs = np.random.permutation(len(os.listdir(block_repo_path)))

    val_block_num = ceil(len(block_seqs) * val_ratio) # 將train和val所使用的block_range分開

    if batch_num is not None:

        val_batch_num = round(batch_num * val_ratio)
        train_batch_num = batch_num - val_batch_num

        assert train_batch_num>=1 and val_batch_num>=1, 'val_ratio or batch_num is too small'

    else:
        val_batch_num, train_batch_num = None, None


    batch_gen = lambda sub_repo_seq=None: iter_block_batches(block_repo_path, 'train', sub_repo_seq=sub_repo_seq, block_range=block_seqs[val_block_num:],
                                                             adj_type=adj_type, self_attn=self_attn, batch_limit=train_batch_num, sample_weight_type=sample_weight_type)
    val_batch_gen = lambda sub_repo_seq=None: iter_block_batches(block_repo_path, 'train', sub_repo_seq=sub_repo_seq, block_range=block_seqs[:val_block_num],
                                                                 adj_type=adj_type, self_attn=self_attn, batch_limit=val_batch_num, sample_weight_type=sample_weight_type)


    model.train(batch_gen, val_batch_gen, epochs, attn_drop, ft_drop, bn_training, log_freqs, max_step, early_stop_threshold, report_upon_oom=True)


def search_expt(model_id, block_repo_name, repr_type, adj_type, self_attn, rel_calc_freq, top_k, ckpt_step=None, to_log=False):

    logger = mlogging.get_logger(('cagnir/%s.search.log'%model_id if to_log else None), prefix='CAGNIR')

    logger.info('model id: %s (search)'%model_id)

    model = CAGNIR(model_id, load_model=True, logger=logger)

    block_repo_path_proto = relpath('batch/%s/%%s'%block_repo_name, env('SQCL_DIR'))

    test_queries_batch_gen = lambda: iter_block_batches(block_repo_path_proto%'test_queries', 'search', adj_type=adj_type, self_attn=self_attn)
    docs_batch_gen = lambda: iter_block_batches(block_repo_path_proto%'docs', 'search', adj_type=adj_type, self_attn=self_attn)

    logger.info('Loading records.')

    rec_ids = []
    for rec_type in ['test_query', 'doc']:
        with open(block_repo_path_proto%(rec_type+'_ids.json')) as f:
            rec_ids.append(json.load(f))
    test_query_ids, doc_ids = rec_ids

    logger.info('Records loaded.')

    search_ranks = model.search(test_queries_batch_gen, docs_batch_gen, test_query_ids, doc_ids, rel_calc_freq, top_k, ckpt_step, repr_type)

    if ckpt_step is None:
        output_path = relpath('search_ranks/cagnir/%s.top_%s.json'%(model_id, top_k), env('RESULT_DIR'))
    else:
        output_path = relpath('search_ranks/cagnir/%s.step_%s.top_%s.json'%(model_id, ckpt_step, top_k), env('RESULT_DIR'))

    with open(output_path, 'w') as f:
        json.dump(search_ranks, f)

    logger.info('Complete!')



if __name__ == '__main__':

    args = parse_args({
        '--train': {
            'flags': ['-t'],
            'opts': {'action':'store_true'},
        },
        '--search': {
            'flags': ['-s'],
            'opts': {'action':'store_true'},
        },
        '--model_id': {
            'flags': ['-m'],
            'opts': {
                'help': 'Id of model used, required while searching.',
            }
        },
        '--ckpt_step': {
            'flags': ['--step'],
            'opts': {
                'help': 'Step of checkpoint for searching.',
            }
        },
        '--hibernate': {
            'opts': {'action':'store_true'},
        },
    })


    if args.train:

        load_model = False

        if not load_model:
            model_id = 'cagnir.%s'%get_timestamp()

        else:
            existed_model_id = ''
            model_id = existed_model_id

        note = ''

        to_log, log_tfvars = True, True
        # to_log, log_tfvars = False, False

        sample_weight_type = None  # 'tacm'

        # model params
        model_config = CAGNIRConfig(**{
            'input_size': 68894,
            'agg_layers': 2,
            'loss_type': 'attention_rank',  # 'dssm_loss'、'attention_rank'、'rank_dssm_loss'
            'gamma': 4,
            'irrel_num': 8,

            'learning_rate': 1e-5,
            'n_heads': [4, 2],
            'hid_units': [512, 256, 256, 256, 128],

            'batch_norm': False,
            'residual': False,
            'weight_decay': 0.,

            'use_sample_weights': (sample_weight_type is not None),
            'seed': 18035,
        })

        # train params
        attn_drop = 0.
        ft_drop = 0.
        bn_training = False

        block_repo_name = '[block_repo_name]'

        epochs = 9
        val_ratio = 0.02
        adj_type = 'log_scale'  # 'normal', 'log_scale', 'equal'
        self_attn = 'max_click'
        log_freqs = (300, 2400, 2000, 300, 300, 168000000)
        # 'train', 'val', 'val_info', 'summary', 'init_summary', 'reg_ckpt'
        max_step = 16800
        early_stop_threshold = 0.1

        batch_num = None
        block_seed = 18035

        with ExceptiontContainer(log_prefix='CAGNIR', hibernate=True, use_console=True, beep=True):

            train_expt(model_id, 'train', model_config, note, to_log=to_log,
                      load_model=load_model, attn_drop=attn_drop, ft_drop=ft_drop, bn_training=bn_training, log_tfvars=log_tfvars, log_freqs=log_freqs, max_step=max_step, early_stop_threshold=early_stop_threshold,
                      block_repo_name=block_repo_name, epochs=epochs, val_ratio=val_ratio, adj_type=adj_type, self_attn=self_attn, sample_weight_type=sample_weight_type,
                      batch_num=batch_num, block_seed=block_seed)


    elif args.search:

        assert args.model_id is not None, '`model_id` is not passed'

        model_id = args.model_id

        to_log = False

        block_repo_name = 'npy_batches.v2.ne.wp-q_f8-d_f150.search.ba_64.al_2.bl_2400'

        repr_type = 'cagnir'

        adj_type = 'log_scale'  # 'normal', 'log_scale', 'equal'
        self_attn = 'max_click'

        rel_calc_freq = 2000 # 在每多少個doc batch reprs計算後，一次性計算這些doc和test queries的relevances

        top_k = 10

        with ExceptiontContainer(log_prefix='CAGNIR', hibernate=True, use_console=True, beep=True):

            search_expt(model_id, block_repo_name, repr_type, adj_type, self_attn, rel_calc_freq, top_k, args.ckpt_step, to_log)


    if args.hibernate and not env('WORKING', dynamic=True):
        os_shutdown(hibernate=True)
