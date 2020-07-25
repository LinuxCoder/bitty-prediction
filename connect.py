#!/usr/bin/python
import psycopg2
from config import config

def load_data(table):
    """ Connect to the PostgreSQL database server """
    conn = None
    try:
        params = config()

        conn = psycopg2.connect(**params)
		
        cur = conn.cursor()
        
        cur.execute('SELECT * FROM {}'.format(table))
        
        data = cur.fetchall()

        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()

    return data


def load_data_from_time(time_in_millis):
    conn = None
    try:
        params = config()
        conn = psycopg2.connect(**params)
        cur = conn.cursor()
        cur.execute('SELECT * FROM btc_value WHERE time > {}'.format(time_in_millis))

        data = cur.fetchall()

        cur.close()

    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()

    return data


def insert_predictions(ids, predictions):
    conn = None
    try:
        params = config()
        conn = psycopg2.connect(**params)
        cur = conn.cursor()
        for id_, pred in zip(ids, predictions):
            cur.execute("UPDATE btc_value_pred SET pred_value={} WHERE id={}".format(pred, id_))

        conn.commit()

        cur.close()

    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()