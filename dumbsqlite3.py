#!/usr/bin/env python
"""Simple and easy way to shove data into a single SQL table."""

import numpy as np
import time
import sqlite3
import io


# Default values
DEF_TABLE = 'sim_results'
DEF_SECONDARY_TABLE = 'barywidths'
DEF_DB = 'simdb.sqlite'
DEF_ASSOC_TABLE = 'type_assoc'

__PRIMARY = 'date'



#----------------------
def set_table(table):
    if type(table).__name__ == 'str':
        DEF_TABLE = table
    else:
        raise TypeError('Tablename must be a string')


#---------------------------
def build_timestamp_id():
    """Builds a timestamp, and appens a random 3 digit number after it"""
    tempo = time.localtime()
    vals = ['year', 'mon', 'mday', 'hour', 'min', 'sec']
    vals = ['tm_' + x for x in vals]

    tstr = [str(getattr(tempo,x)).zfill(2) for x in vals]

    return int(''.join(tstr) + str(np.random.randint(999)).zfill(3))


#---------------------
# TYPE ADAPT FUNCTIONS
def adapt_array(arr):
    """
    http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
    """
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())

def adapt_list(lst):
    bin_list = bytes(repr(lst), 'ascii')
    return sqlite3.Binary(bin_list)

def adapt_bool(boolean):
    if boolean:
        return 1
    else:
        return 0

def adapt_float64(number):
    return float(number)


#----------------------
# TYPE CONVERT FUNCTIONS
def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)

def convert_list(text):
    return(eval(text))

def convert_bool(boolean):
    if boolean == 1:
        return True
    else:
        return False

def convert_float64(number):
    return np.float64(number)



#----------------------
def connect(dbase_file=DEF_DB):
    
    sqlite3.register_adapter(np.ndarray, adapt_array)
    sqlite3.register_adapter(list, adapt_list)
    sqlite3.register_adapter(bool, adapt_bool)
    sqlite3.register_adapter(np.float64, adapt_float64)

    sqlite3.register_converter("ARRAY", convert_array)
    sqlite3.register_converter("LIST", convert_list)
    sqlite3.register_converter("BOOL", convert_bool)
    sqlite3.register_converter("FLOAT64", convert_float64)

    conn = sqlite3.connect(dbase_file, detect_types=sqlite3.PARSE_DECLTYPES)
    return conn





#----------------------
def init(dbase_file=DEF_DB, table_name=DEF_TABLE):
    conn = sqlite3.connect(dbase_file)
    c = conn.cursor()

    field_name = __PRIMARY
    field_type = 'INTEGER'
    
    #Initiate the default table, usually sim_result
    c.execute('CREATE TABLE {tn} ({fn} {ft} PRIMARY KEY)'\
              .format(tn=table_name, fn=field_name, ft=field_type))

    # Initiate a secondary table
    c.execute('CREATE TABLE {tn} ({fn} {ft} PRIMARY KEY)'\
              .format(tn=DEF_SECONDARY_TABLE, fn=field_name, ft=field_type))


    type_assoc = {\
             'int':'INTEGER',\
             'bool':'BOOL',\
             'str':'TEXT',\
             'ndarray':'ARRAY',\
             'float':'REAL',\
             'float64':'FLOAT64',\
             'list':'LIST'\
             }
    
    c.execute("CREATE TABLE {} (ptype TEXT PRIMARY KEY, stype TEXT)"\
              .format(DEF_ASSOC_TABLE))


    for key, val in type_assoc.items():
        c.execute("INSERT INTO {} (ptype, stype)VALUES(?,?)".\
                  format(DEF_ASSOC_TABLE), [key, val])

    
    
    conn.commit()
    conn.close()




#---------------------
def get_type_assoc(conn, tn_assoc=DEF_ASSOC_TABLE):
    c = conn.cursor()
    cursor = c.execute('select * from ' + tn_assoc)
    return dict(cursor.fetchall())





#-------------------------
def add(data, tn=DEF_TABLE, dbase_file=DEF_DB, conn=False):
    """Adds the data dictionary as one row. If new columns are added, older entries will be appropriately initiated"""
    close_conn = False
    if conn == False: 
        conn = connect(dbase_file)
        closeconn = True


    c = conn.cursor()
    cursor = c.execute('select * from ' + tn)
    dbcols = [x[0] for x in cursor.description]
    datacols = list(data.keys())
    type_assoc = get_type_assoc(conn)

    # Exceptions in input data dict
    if not __PRIMARY in datacols:
        raise AttributeError('The data dict is missing the ' + __PRIMARY + ' ID (Primary attribute)')
    elif len(str(data[__PRIMARY])) != len(str(build_timestamp_id())):
        raise ValueError("The date is of improper size. Did you use build_timestamp_id to make it?")


    # find and add nonexistant colums when necessary.
    toadd = []
    for x in datacols:
        if not x in dbcols:
            tmp = type(data[x]).__name__ # Extract python type
            toadd.append([x,type_assoc[tmp]]) # <--- [varname, sql_type]
 
    for x in toadd:
        c.execute("ALTER TABLE {tn} ADD COLUMN '{cn}' {ct}"\
                  .format(tn=tn, cn=x[0], ct=x[1]))
    
    conn.commit()


    #Actually add
    collist = []
    vallist = []
    for cn, val in data.items():
        # Convert booleans to integers
        if cn == __PRIMARY:
            continue

        if type(val) == type(True):
            val = int(val)
        collist.append(cn)
        vallist.append(val)

    c.execute("INSERT INTO {} ({}) VALUES ({})".format(tn, __PRIMARY, data[__PRIMARY]))
    for k in range(len(collist)):
        
        c.execute("UPDATE {} SET {}=(?) WHERE date={}".format(tn, collist[k], data[__PRIMARY])\
                  , (vallist[k],))
    
    conn.commit()

    if close_conn:
        conn.close()





#--------------------
# FETCHING FUNCTIONS
#--------------------

#------------------------
def fetchall(tn=DEF_TABLE, dbase_file=DEF_DB, conn=False):
    """Fetches all entries in table"""
    # This allows is to passs a connection instead of always opening a new one.
    close_conn = False
    if conn == False: 
        conn = connect(dbase_file)
        closeconn = True
    
    c = conn.cursor()
    cursor = c.execute('SELECT * from ' + tn)
    output = cursor.fetchall()

    if close_conn:
        conn.close()

    return output





#-------------------------
def fetch_collist(tn=DEF_TABLE, dbase_file=DEF_DB, conn=False):
    """Fetches all entries and returns it as a dict of tuples, where each tuple represents all the (ordered) values in that column)"""
    
    # This allows is to passs a connection instead of always opening a new one.
    close_conn = False
    if conn == False: 
        conn = connect(dbase_file)
        closeconn = True

    c = conn.cursor()
    cursor = c.execute('select * from ' + tn)
    dbcols = [x[0] for x in cursor.description]

    
    if close_conn:
        conn.close()

    return dbcols



#-----------------------
def fetch_matching(entry_dict, tn=DEF_TABLE, dbase_file=DEF_DB, conn=False, get_data=True):
    """Fetches the dates entries that fully match the dict values"""
    close_conn = False
    if conn == False: 
        conn = connect(dbase_file)
        closeconn = True

    c = conn.cursor()

    if get_data:
        string = "SELECT * FROM " + tn + " WHERE " 
    else:
        string = "SELECT date FROM " + tn + " WHERE " 

    # Build the execute string and data list
    data_list = []
    for key, val in entry_dict.items():
        string += key + " = ? AND "
        data_list.append(val)
    string = string[:-5]

    cursor = c.execute(string, data_list)

    
    if close_conn:
        conn.close()

    return cursor.fetchall()


#-----------------------
def fetchone(date, column , tn=DEF_TABLE, dbase_file=DEF_DB, conn=False):
    """Fetches the columns in collist"""
    close_conn = False
    if conn == False: 
        conn = connect(dbase_file)
        closeconn = True

    c = conn.cursor()

    string = "SELECT " + column + " FROM " + tn + " WHERE date = " + str(date)
    c.execute(string)
    
    if close_conn:
        conn.close()

    return c.fetchall()[0][0]
