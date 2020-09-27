
from .environ import env

import MySQLdb



class DB(object):

    def __init__(self, host=env('DB_HOST'), user=env('DB_USER'), passwd=env('DB_PASSWD'), db=env('DB_NAME'), charset=env('DB_CHARSET')):

        self._host = host
        self._user = user
        self._passwd = passwd
        self._db = db
        self._charset = charset

        self._conn = None
        self._cursor = None


    def __enter__(self):

        self._conn = MySQLdb.connect(host=self._host, user=self._user, passwd=self._passwd, db=self._db, charset=self._charset)
        self._cursor = self._conn.cursor()

        return self._conn, self._cursor


    def __exit__(self, exc_type, exc_val, exc_tb):

        self._cursor.close()
        self._conn.close()


class SqclDB(DB):

    def select_docs(self, field_str='doc_id, title', where_clause=None, order_by='doc_id', limit=None):

        with self as (conn, cursor):

            where_clause = '' if where_clause is None else 'WHERE %s'%where_clause

            sql = """
                SELECT %s
                FROM `doc`
                %s
                ORDER BY %s
                """%(field_str, where_clause, order_by)

            if limit is not None: sql += ' LIMIT %d'%limit

            cursor.execute(sql)

            return cursor.fetchall()


    def select_queries(self, field_str='q_id, query', where_clause=None, order_by='q_id', limit=None):

        with self as (conn, cursor):

            where_clause = '' if where_clause is None else 'WHERE %s'%where_clause

            sql = """
                SELECT %s
                FROM `query`
                %s
                ORDER BY %s
                """%(field_str, where_clause, order_by)

            if limit is not None: sql += ' LIMIT %d'%limit

            cursor.execute(sql)

            return cursor.fetchall()


    def select_clicks(self, field_str='q_id, doc_id, click', dataset=None, where_clause=None, order_by='q_id, doc_id', limit=None):

        with self as (conn, cursor):

            if dataset is None:
                where_clause = '' if where_clause is None else 'WHERE %s'%where_clause
            else:
                if dataset.__class__.__name__=='list':
                    dataset_clause = ' OR '.join('dataset="%s"'%set_name for set_name in dataset)
                else:
                    dataset_clause = 'dataset="%s"'%dataset
                where_clause = 'WHERE %s'%dataset_clause if where_clause is None else 'WHERE (%s) AND %s'%(dataset_clause, where_clause)


            sql = '''
                SELECT %s
                FROM `click`
                %s
                ORDER BY %s
                '''%(field_str, where_clause, order_by)

            if limit is not None: sql += ' LIMIT %d'%limit

            cursor.execute(sql)

            return cursor.fetchall()
