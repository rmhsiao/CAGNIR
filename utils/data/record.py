
from scipy.sparse import csr_matrix

from .text import WordPieceParser

from collections.abc import Mapping, Iterable



class RecordVectorMap(Mapping):

    def __init__(self, records, wp_model_path, vec_format='bag-of-words'):

        text_parser = WordPieceParser(wp_model_path)

        self.rec_seq_map, self.record_vecs = self.rec2vecs(records, text_parser, vec_format)


    def rec2vecs(self, records, text_parser, vec_format):

        rec_seq_map = {}
        cols, rows, data = [], [], []

        col_dim = 0 if vec_format=='sequence' else text_parser.vocab_size

        for rec_seq, (rec_id, rec_text) in enumerate(records):

            rec_seq_map[rec_id] = rec_seq

            parsed = text_parser.parse(rec_text, parse_format=vec_format)

            if vec_format=='sequence':

                if len(parsed)!=0:
                    rows.extend([rec_seq]*len(parsed))
                    cols.extend(list(range(len(parsed))))
                    data.extend(parsed)
                    if len(parsed)>col_dim:
                        col_dim = len(parsed)

            else:

                for wp_id, tf in parsed.items():
                    rows.append(rec_seq)
                    cols.append(wp_id)
                    data.append(tf)

        record_vecs = csr_matrix((data, (rows, cols)), shape=(len(records), col_dim))

        return rec_seq_map, record_vecs

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.get_by_seqs(self.rec_seq_map[key])
        elif isinstance(key, Iterable):
            return self.get_by_seqs([self.rec_seq_map[a_key] for a_key in key])
        else:
            raise TypeError('Key must be string (key of record) or iterable (list of key of record).')

    def get_by_seqs(self, key):
        if isinstance(key, int):
            return self.record_vecs[key]
        elif isinstance(key, Iterable):
            return self.record_vecs[key]
        else:
            raise TypeError('Seqs must be int (seq of record) or iterable (list of seq of record).')

    def __iter__(self):
        return iter(self.record_vecs)

    def __len__(self):
        return len(self.rec_seq_map)


