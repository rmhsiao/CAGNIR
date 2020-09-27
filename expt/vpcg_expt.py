
from scipy import sparse

from utils.environ import env
from utils.helpers import *
from utils.db import SqclDB
from utils.mlogging import mlogging
from utils.exception import ExceptiontContainer
from utils.record import RecordVectorMap

from models.vpcg import VPCG
from utils.vsm import vsm_search

import json



def train_with_wordpiece(model_id, wp_id, semantic_space, keep_features, batch_size, max_iter, epsilon, log_freq, to_log=True):

    if to_log:
        log_expt_params('vpcg', 'train_wordpiece', model_id, {'wp_id':wp_id, 'semantic_space':semantic_space, 'keep_features':keep_features, 'epsilon':epsilon})

    logger = mlogging.get_logger(('vpcg/vpcg.wordpiece.train.log' if to_log else None), prefix='VPCG (wordpiece)')

    logger.info('Start!')
    logger.info('model id: %s'%model_id)

    db = SqclDB()
    queries, docs = db.select_queries(), db.select_docs()
    clicks = db.select_clicks()

    logger.info('Records loaded.')

    q_ids, doc_ids = [[rec[0] for rec in records] for records in [queries, docs]]

    wp_model_path = relpath('wordpiece/%s.model'%wp_id, env('MODELBASE_DIR'))

    if semantic_space=='query':
        query_data = RecordVectorMap(queries, wp_model_path).record_vecs, q_ids
        doc_data = None, doc_ids

    else:
        query_data = None, q_ids
        doc_data = RecordVectorMap(docs, wp_model_path).record_vecs, doc_ids

    logger.info('Records data prepared.')

    model = VPCG(model_id, logger)

    model.train(query_data, doc_data, clicks, semantic_space, keep_features, batch_size, max_iter, epsilon, log_freq)

    logger.info('Complete !')

def train_with_tfidf(model_id, non_empties, semantic_space, keep_features, batch_size, max_iter, epsilon, self_aggr, log_freq, note=None, to_log=True):

    if to_log:
        expt_params = {'non_empties':non_empties, 'semantic_space':semantic_space, 'keep_features':keep_features, 'epsilon':epsilon, 'self_aggr':self_aggr}
        log_expt_params('vpcg', 'train_tfidf', model_id, expt_params, note)

    logger = mlogging.get_logger(('vpcg/vpcg.tfidf.train.log' if to_log else None), prefix='VPCG')

    logger.info('Start!')
    logger.info('model id: %s'%model_id)

    dataname_affix = '.ne' if non_empties else ''

    rec_data = {}
    for rec_type in ['query', 'doc']:

        with open(relpath('tfidfs/rec_ids%s.%s.json'%(dataname_affix, rec_type), env('SQCL_DIR'))) as f:
            rec_ids = json.load(f)

        rec_tfidfs = sparse.load_npz(relpath('tfidfs/rec_tfidfs%s.%s.npz'%(dataname_affix, rec_type), env('SQCL_DIR'))) if semantic_space==rec_type else None

        rec_data[rec_type] = (rec_tfidfs, rec_ids)

    clicks = SqclDB().select_clicks()

    logger.info('Records prepared.')

    model = VPCG(model_id, logger)

    model.train(rec_data['query'], rec_data['doc'], clicks, semantic_space, keep_features, batch_size, max_iter, epsilon, self_aggr, log_freq)

    logger.info('Complete !')


def search(model_id, n_iter, doc_batch_size, rel_calc_freq, top_k):

    logger = mlogging.get_logger(prefix='VPCG (search)')

    logger.info('Start!')

    test_query_ids, doc_ids = _get_test_data()

    logger.info('Records loaded.')

    model_path = relpath('vpcg/%s'%model_id, env('MODELBASE_DIR'))
    rec_reprs = sparse.load_npz(relpath('%s.iter_%s.npz'%(model_id, n_iter), model_path))
    with open(relpath('rec_seq_map.json', model_path)) as f:
        rec_seq_map = json.load(f)

    test_query_seqs = [rec_seq_map[rec_id] for rec_id in test_query_ids]
    doc_seqs, doc_ids = list(map(list, zip(*[(rec_seq_map[rec_id], rec_id) for rec_id in doc_ids if rec_id in rec_seq_map])))
    # 有些doc會因不在rec_seq_map內而被篩掉，因此在此一併更新doc_ids

    test_query_reprs = rec_reprs[test_query_seqs]
    doc_reprs = rec_reprs[doc_seqs]

    doc_reprs_gen = (doc_reprs[batch_head:batch_head+doc_batch_size] for batch_head in range(0, len(doc_ids), doc_batch_size))

    logger.info('Records vectors prepared.')

    search_ranks = vsm_search(test_query_reprs, doc_reprs_gen, test_query_ids, doc_ids, rel_calc_freq, top_k, True, logger)

    output_path = relpath('search_ranks/vpcg/%s.iter_%s.top_%s.json'%(model_id, n_iter, top_k), env('RESULT_DIR'))
    with open(output_path, 'w') as f:
        json.dump(search_ranks, f)

    logger.info('Complete!')


def _get_test_data():

    db = SqclDB()

    docs = db.select_docs(field_str='doc_id', order_by='doc_id')

    with db as (conn, cursor):

        sql = '''
            SELECT DISTINCT Q.q_id
            FROM query AS Q JOIN click AS C ON (Q.q_id=C.q_id AND C.dataset='test')
            ORDER BY Q.q_id
            '''
        cursor.execute(sql)
        test_queries = cursor.fetchall()

    return [row[0] for row in test_queries], [row[0] for row in docs]



if __name__ == '__main__':

    args = parse_args({
        '--train_type': None,
        '--search': {
            'flags': ['-s'],
            'opts': {'action':'store_true'},
        },
        '--model_id': None,
        '--n_iter': {
            'flags': ['-i'],
        },
        '--hibernate': {
            # 'flags': ['-h'],
            'opts': {'action':'store_true'},
        },
    })


    if args.train_type=='wordpiece':

        model_id = 'vpcg.wordpiece.%s'%get_timestamp()

        wp_id = 'wordpiece.q_f50.d_n20000'

        semantic_space = 'query'
        keep_features = 20
        batch_size = 20
        max_iter = 20
        epsilon = 1e-12
        log_freq = int(1000000 / batch_size)

        to_log = True

        with ExceptiontContainer(log_prefix='VPCG (wordpiece)', hibernate=True, use_console=True, beep=True):

            train_with_wordpiece(model_id, wp_id, semantic_space, keep_features, batch_size, max_iter, epsilon, log_freq, to_log)

    elif args.train_type=='tfidf':

        non_empties = True

        model_id = 'vpcg.tfidf%s.%s'%('.ne' if non_empties else '', get_timestamp())

        semantic_space = 'doc'
        keep_features = 30
        batch_size = 20
        max_iter = 10
        epsilon = 1e-12
        self_aggr = True
        log_freq = 10 # int(1000000 / batch_size)

        note = ''
        to_log = True

        with ExceptiontContainer(log_prefix='VPCG (tfidf)', hibernate=True, use_console=True, beep=True):

            train_with_tfidf(model_id, non_empties, semantic_space, keep_features, batch_size, max_iter, epsilon, self_aggr, log_freq, note, to_log)


    elif args.search:

        model_id = args.model_id
        n_iter = args.n_iter

        doc_batch_size = 150000
        rel_calc_freq = 1

        top_k = 10

        search(model_id, n_iter, doc_batch_size, rel_calc_freq, top_k)


    if args.hibernate and not env('WORKING', dynamic=True):
        os_shutdown(hibernate=True)
