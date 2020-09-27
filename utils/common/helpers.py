
from scipy import sparse
import numpy as np
import tensorflow as tf

from .environ import env

import os
import pprint
from datetime import datetime
import argparse
import json
from math import floor
from itertools import zip_longest



pprint = pprint.PrettyPrinter(indent=4).pprint

get_timestamp = lambda format='%Y%m%d%H%M%S': datetime.today().strftime(format)

class_name = lambda instance: instance.__class__.__name__


def relpath(file_relpath, rootpath=env('ROOT_DIR')):
    return os.path.join(rootpath, file_relpath)

def os_shutdown(shutdown=True, hibernate=False):

    # for windows

    if hibernate:
        os.system('shutdown -h')

    elif shutdown:
        os.system('shutdown -s')

def parse_args(arg_params):

    parser = argparse.ArgumentParser()

    for arg_name, params in arg_params.items():

        if type(params)==dict:

            opts = params['opts'] if 'opts' in params else {}

            if 'flags' in params:
                parser.add_argument(*([arg_name]+params['flags']), **opts)
            else:
                parser.add_argument(arg_name, **opts)

        else:
            parser.add_argument(arg_name)

    return parser.parse_args()


def log_expt_params(expt_class, expt_mode, expt_id, expt_params, note=None, line_max_len=80):

    log_path = relpath('%s/expt_params.%s.log'%(expt_class, expt_class), env('LOG_DIR'))

    line_fields_list = list(zip(*sorted(expt_params.items(), key=lambda item: item[0])))

    col_len = max([len(str(field)) for line_fields in line_fields_list for field in line_fields]) + 2
    col_span_len = 2
    line_field_num = floor(line_max_len / col_len)

    model_info = '# expt_id: %s (%s)'%(expt_id, expt_mode)
    hr_len = line_max_len
    log_infos = ['', model_info] + (['> *%s*'%note, '-'*hr_len] if (isinstance(note, str) and note!='') else ['-'*hr_len])

    for head_field in range(0, len(line_fields_list[0]), line_field_num):

        tail_field = head_field + line_field_num

        for i, field_format in enumerate(['- {:<%s}', '  {:<%s}']):

            fields = line_fields_list[i][head_field:tail_field]

            line_format = (' '*col_span_len).join([field_format%col_len]*len(fields))
            log_infos.append(line_format.format(*map(str, fields)))

    log_infos += ['']

    with open(log_path, 'a', encoding='utf8') as f:
        f.write('\n'.join(log_infos))


def tail_padding(iterable, fill_value, min_fixed_length=None):

    padded = np.array(list(zip(*zip_longest(*iterable, fillvalue=fill_value))))

    if isinstance(min_fixed_length, int) and len(padded[0]) < min_fixed_length:
        tail_pad_lenth = min_fixed_length-len(padded[0])
        padded = np.pad(padded, ((0, 0), (0, tail_pad_lenth)), 'constant', constant_values=fill_value)

    return padded


def sparse_l2norm_by_row(sparse_mat, epsilon, batch_size=None):

    if batch_size is None:

        row_norms = sparse.linalg.norm(sparse_mat, axis=-1).reshape(-1,1) + epsilon
        return sparse_mat.multiply(1 / row_norms).tocsr()

    else:

        queue = []

        for batch_head in range(0, sparse_mat.shape[0], batch_size):

            batch_mat = sparse_mat[batch_head:batch_head+batch_size]

            batch_norms = sparse.linalg.norm(batch_mat, axis=-1).reshape(-1,1) + epsilon
            queue.append(batch_mat.multiply(1 / batch_norms))

        return sparse.vstack(queue).tocsr()

def sparse_top_k_by_row(sparse_mat, k):

    # refered and modified from:
    # https://stackoverflow.com/questions/31790819/scipy-sparse-csr-matrix-how-to-get-top-ten-values-and-indices

    sparse_mat_lil = sparse_mat.tolil()

    new_rows, new_data = [], []

    for row, data in zip(sparse_mat_lil.rows, sparse_mat_lil.data):

        if len(data)>k:
            top_rows, top_data = list(zip(*sorted(zip(row, data), key=lambda x: x[1], reverse=True)[:k]))
        else:
            top_rows, top_data = row, data

        new_rows.append(top_rows)
        new_data.append(top_data)

    sparse_mat_lil.rows, sparse_mat_lil.data = new_rows, new_data

    return sparse_mat_lil.tocsr()
