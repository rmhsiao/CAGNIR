
from scipy import sparse

from utils.environ import env
from utils.helpers import *
from utils.db import SqclDB
from utils.mlogging import mlogging
from utils.record import RecordVectorMap

from utils.vsm import vsm_search

import json



def vsm_search_with_wordpiece(wp_model_path, non_empties, doc_batch_size, rel_calc_freq, top_k):

    logger = mlogging.get_logger(prefix='VSM (wordpiece)')

    logger.info('Start!')

    test_queries, docs, test_query_ids, doc_ids = _get_test_data(non_empties)

    logger.info('Records loaded.')

    test_query_reprs = RecordVectorMap(test_queries, wp_model_path).record_vecs
    doc_reprs = RecordVectorMap(docs, wp_model_path).record_vecs

    doc_reprs_gen = (doc_reprs[batch_head:batch_head+doc_batch_size] for batch_head in range(0, len(doc_ids), doc_batch_size))

    logger.info('Records vectors prepared.')

    search_ranks = vsm_search(test_query_reprs, doc_reprs_gen, test_query_ids, doc_ids, rel_calc_freq, top_k, True, logger)

    wp_model_id = '.'.join(wp_model_path.split('/')[-1].split('.')[:-1])
    rankname_affix = '.ne' if non_empties else ''
    output_path = relpath('search_ranks/vsm/vsm.%s%s.top_%s.json'%(wp_model_id, top_k, rankname_affix), env('RESULT_DIR'))

    with open(output_path, 'w') as f:
        json.dump(search_ranks, f)

    logger.info('Complete!')


def vsm_search_with_tfidf(non_empties, doc_batch_size, rel_calc_freq, top_k):

    logger = mlogging.get_logger(prefix='VSM (wordpiece)')

    logger.info('Start!')

    dataname_affix = '.ne' if non_empties else ''

    rec_ids, rec_reprs = {}, {}
    for rec_type in ['query', 'doc']:

        rec_reprs[rec_type] = sparse.load_npz(relpath('tfidfs/rec_tfidfs%s.%s.npz'%(dataname_affix, rec_type), env('SQCL_DIR')))

        with open(relpath('tfidfs/rec_ids%s.%s.json'%(dataname_affix, rec_type), env('SQCL_DIR'))) as f:
            rec_ids[rec_type] = json.load(f)

    test_query_ids = _get_test_data()[2]
    test_query_seq_map = {rec_id: rec_seq for rec_seq, rec_id in enumerate(rec_ids['query']) if rec_id in set(test_query_ids)}

    test_query_reprs = rec_reprs['query'][[test_query_seq_map[rec_id] for rec_id in test_query_ids]]

    doc_ids = rec_ids['doc']
    doc_reprs_gen = (rec_reprs['doc'][batch_head:batch_head+doc_batch_size] for batch_head in range(0, len(doc_ids), doc_batch_size))

    logger.info('Records vectors prepared.')

    search_ranks = vsm_search(test_query_reprs, doc_reprs_gen, test_query_ids, doc_ids, rel_calc_freq, top_k, True, logger)

    output_path = relpath('search_ranks/vsm/vsm.tfidf%s.top_%s.json'%(dataname_affix, top_k), env('RESULT_DIR'))

    with open(output_path, 'w') as f:
        json.dump(search_ranks, f)

    logger.info('Complete!')



def _get_test_data(non_empties=False):

    db = SqclDB()

    where_clause = 'title!=""' if non_empties else None
    docs = db.select_docs(field_str='doc_id, title', where_clause=where_clause, order_by='doc_id')

    with db as (conn, cursor):

        sql = '''
            SELECT DISTINCT Q.q_id, Q.query
            FROM query AS Q JOIN click AS C ON (Q.q_id=C.q_id AND C.dataset='test')
            ORDER BY Q.q_id
            '''
        cursor.execute(sql)
        test_queries = cursor.fetchall()

    return test_queries, docs, [row[0] for row in test_queries], [row[0] for row in docs]



if __name__ == '__main__':

    args = parse_args({
        '--type': {
            'flags': ['-t'],
        },
        '--hibernate': {
            'opts': {'action':'store_true'},
        },
    })


    if args.type=='wordpiece':

        wp_model_id = 'wordpiece.q_f8.d_f150'
        wp_model_path = relpath('wordpiece/%s.model'%wp_model_id, env('MODELBASE_DIR'))

        non_empties = True

        doc_batch_size = 150000
        rel_calc_freq = 1

        top_k = 10

        vsm_search_with_wordpiece(wp_model_path, non_empties, doc_batch_size, rel_calc_freq, top_k)

    elif args.type=='tfidf':

        non_empties = True

        doc_batch_size = 150000
        rel_calc_freq = 1

        top_k = 10

        vsm_search_with_tfidf(non_empties, doc_batch_size, rel_calc_freq, top_k)

