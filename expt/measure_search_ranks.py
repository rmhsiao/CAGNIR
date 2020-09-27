
from scipy import stats

from utils.environ import env
from utils.helpers import *
from utils.metrics import ndcg

import json


def measure_w_ndcg(rank_ids, groundtruth_labels_path, top_k, ndcg_ver, format_output=True):

    with open(groundtruth_labels_path) as f:
        groundtruth_labels = json.load(f)

    rank_scores_pairs = []

    for r_id in rank_ids:

        model_class = r_id.split('.')[0]

        search_ranks_path = relpath('search_ranks/%s/%s.json'%(model_class, r_id), env('RESULT_DIR'))
        with open(search_ranks_path) as f:
            search_ranks = json.load(f)

        ndcg_scores = ndcg(search_ranks, groundtruth_labels, ver=ndcg_ver, get_mean=False)

        if format_output:
            rank_scores_pairs.append((r_id, ndcg_scores))

        else:
            print('', r_id, ndcg_scores.mean(axis=0), '', sep='\n', end='')

    if format_output:

        line_format = '{:<%s} | '%max(len(r_id) for r_id in rank_ids)+ ''.join('{:<9}' for i in range(top_k))
        infos = [line_format.format('', *['NDCG@%s'%(i+1) for i in range(top_k)])]
        infos.append('-'*len(infos[-1]))

        if len(rank_scores_pairs)==2:
            addt_infos = ['p-value@k:', '-'*16]

        if len(rank_scores_pairs)==1:
            infos.append(line_format.format(rank_scores_pairs[0][0], *['%.4f'%score for score in rank_scores_pairs[0][1].mean(axis=0)]))

        else:

            for i, (r_id, ndcg_scores) in enumerate(rank_scores_pairs):

                ndcg_mean = ndcg_scores.mean(axis=0)
                line_data = [r_id]

                for k, score in enumerate(ndcg_mean):

                    if i==0:
                        sgfnt_1, sgfnt_2 = False, False

                    else:

                        t_stats, p_val = stats.ttest_rel(ndcg_scores[:,k], rank_scores_pairs[0][1][:,k])
                        sgfnt_1 = p_val<5e-2
                        sgfnt_2 = p_val<1e-2

                        if len(rank_scores_pairs)==2:
                            addt_infos.append('@%-2d | %.4f %s'%(k+1, p_val, '.'*sum([sgfnt_1, sgfnt_2])))

                    sgfnt_nmarks = sum([sgfnt_1, sgfnt_2])

                    line_data.append('%.4f%s'%(score, ('.'*sgfnt_nmarks+' '*(2-sgfnt_nmarks))))

                infos.append(line_format.format(*line_data))

        print('\n'+'\n'.join(['  '+line for line in infos]))
        if len(rank_scores_pairs)==2:
            print('\n'+'\n'.join(['  '+line for line in addt_infos]))

    return [(r_id, ndcg_scores.mean(axis=0)) for r_id, ndcg_scores in rank_scores_pairs]



if __name__ == '__main__':

    args = parse_args({
        '--mode': {
            'flags': ['-m'],
            'opts': {
                'default':'ndcg',
                'help': 'mode to measure search ranks',
            },
        },
        '--model_ids': {
            'flags': ['--model_id'],
            'opts': {
                'nargs':'+',
                'help': 'id of models to search ranks',
            }
        },
        '--top_k': {
            'flags': ['-k'],
            'opts': {
                'default': 10,
                'help': '`top_k` value of search ranks',
            }
        },
        '--ndcg_ver': {
            'flags': ['-v'],
            'opts': {
                'default': '1',
                'help': 'version of ndcg',
            }
        },
        '--raw_output': {
            'flags': ['-r'],
            'opts': {
                'action':'store_true',
                'help': 'output raw (unformatted) results',
            }
        },
        '--save': {
            'flags': ['-s'],
            'opts': {
                'action':'store_true',
                'help': 'save raw (unformatted) results',
            }
        },
    })


    if args.mode=='ndcg':

        rank_ids = ['%s.top_%s'%(model_id, args.top_k) for model_id in args.model_ids]

        groundtruth_labels_path = relpath('relevance/relevances.v%s.json'%args.ndcg_ver, env('SQCL_DIR'))

        rank_ndcgs_pairs = measure_w_ndcg(rank_ids, groundtruth_labels_path, args.top_k, args.ndcg_ver, not args.raw_output)

        if args.save:

            for r_id, ndcg_scores in rank_ndcgs_pairs:
                with open(relpath('search_ranks/%s.%s_scores.csv'%(r_id.split('.')[0], args.mode), env('RESULT_DIR')), 'a', encoding='utf8') as f:
                    f.write(', '.join(map(str, [r_id]+ndcg_scores.tolist())) + '\n')
                print('\nndcg result of %s saved'%r_id, end='')
