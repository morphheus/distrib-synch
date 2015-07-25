#!/usr/bin/env python


from lib import *
import sqlite3
import io


# Default values
DEF_TABLE = 'sim_results'
DEF_DB = 'simdb.sqlite'
DEF_ASSOC_TABLE = 'type_assoc'

__PRIMARY = 'date'



def set_table(table):
    if type(table).__name__ == 'str':
        DEF_TABLE = table
    else:
        raise TypeError('Tablename must be a string')



#----------------------
def adapt_array(arr):
    """
    http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
    """
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())




#----------------------
def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)








#----------------------
def connect(dbase_file=DEF_DB):
    # Converts np.array to TEXT when inserting
    sqlite3.register_adapter(np.ndarray, adapt_array)

    # Converts TEXT to np.array when selecting
    sqlite3.register_converter("ARRAY", convert_array)

    conn = sqlite3.connect(dbase_file, detect_types=sqlite3.PARSE_DECLTYPES)
    return conn




#----------------------
def init(dbase_file, table_name=DEF_TABLE):
    conn = sqlite3.connect(dbase_file)
    c = conn.cursor()

    field_name = __PRIMARY
    field_type = 'INTEGER'
    
    c.execute('CREATE TABLE {tn} ({fn} {ft} PRIMARY KEY)'\
              .format(tn=table_name, fn=field_name, ft=field_type))


    type_assoc = {\
             'int':'INTEGER',\
             'bool':'INTEGER',\
             'str':'TEXT',\
             'ndarray':'ARRAY',\
             'float':'REAL',\
             }
    #TODO: auto type assoc creation
    conn.commit()
    conn.close()




#---------------------
def get_type_assoc(dbase_file=DEF_DB, tn_assoc=DEF_ASSOC_TABLE):
    conn = sqlite3.connect(dbase_file)
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
    type_assoc = get_type_assoc()

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
    for k in range(len(collist)): # Numpy array must be introduced via SQLITE variable call
        c.execute("UPDATE {} SET {}=(?) WHERE date={}".format(tn, collist[k], data[__PRIMARY])\
                  , (vallist[k],))
    
    conn.commit()

    if close_conn:
        conn.close()




#------------------------
def fetchall(tn=DEF_TABLE, dbase_file=DEF_DB, conn=False):
    """Executes the statement and pass the output table"""
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



