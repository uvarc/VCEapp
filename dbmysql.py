from sqlalchemy import create_engine
from sqlalchemy.types import Integer, Text

import pymysql
import config
import numpy as np
import pandas as pd
import os

sqlEngine  = create_engine('mysql+pymysql://vcedb:Pd7hDEmGG2lk@jablonskimysql.marathon.l4lb.thisdcos.directory:13308/vce', pool_recycle=3600)

conn=sqlEngine.connect()



def findnames():
    names=conn.execute('SHOW TABLES')
    #names= conn.fetchall()
    names=[item[0] for item in names]
    names.sort()
    return names

def get_vid_data():
    table=conn.execute('SELECT * FROM prog_table')
    columns=table.keys()
    table=table.fetchall()
    table=pd.DataFrame(table, columns=columns)
    return(table)
    
def update_row(vidname, vals, row):
    sql = ''' UPDATE {}
              SET tract_section = "{}",
                  pathology = "{}",
                  notes = "{}",
                  inflammation= {},
                  edemous_villi={},
                  bleed={},
                  diffuse_bleed={}
              WHERE index_ = {}'''
    sql=sql.format(vidname, vals[0], vals[1],vals[2],vals[3],vals[4],vals[5],vals[6], row)
    conn.execute(sql)

    
## Query to enable table update of multiple rows    
def update_multi_row(rows):
    sql=''
    for vals in rows: 
        print(vals)
        sql_i = ''' UPDATE {}
                  SET tract_section = "{}" ,
                      pathology = "{}" ,
                      notes = "{}",
                      inflammation={},
                      edemous_villi={},
                      bleed={},
                      diffuse_bleed={}
                  WHERE index_ = {}; '''
        sql_i=sql_i.format(vals[0], vals[1], vals[2],vals[3],vals[4],vals[5],vals[6], vals[7],vals[8])
        sql=sql+sql_i
   
    conn.execute(sql)
    conn.commit()
    
    
    
def get_anoms(tables, condition):
    query=''
    for x in tables:
        query = query + 'select index_,video,tract_section,pathology,inflammation,edemous_villi,bleed,diffuse_bleed,notes from ' + x + ' where ' + condition + ' UNION ALL ' 
    query=query[:-11]
    sql = query +';'
    
    cur=conn.execute(sql)
    row=cur.fetchall()
    return row    

def set_rest_tract(vidname, val, row):
    row=str(row)
    sql = ''' UPDATE {}
              SET tract_section = "{}" 
              WHERE index_ >= {}'''
    sql=sql.format(vidname, val, row)
    print(sql)
    conn.execute(sql)
    
    
def read_set(vidname, center, frames):
    min_row=min([frame+center for frame in frames])
    max_row=max([frame+center for frame in frames])
    sql = ''' SELECT tract_section,pathology, notes, inflammation, edemous_villi, bleed, diffuse_bleed FROM {}
              WHERE index_ >= {}
              AND index_ <= {}'''
    sql=sql.format(vidname, min_row, max_row)
    #print(sql)
    cur=conn.execute(sql)
    res=cur.fetchall()
    return res

def update_rows(vname,values):
    sql = '''INSERT INTO {} (index_, tract_section, pathology, notes, inflammation, edemous_villi,bleed, diffuse_bleed) 
        VALUES ({}, "{}", "{}", "{}",{},{},{},{}),'''
    for i in range(0,len(values)-1):
        row='''\n    ({}, "{}", "{}", "{}",{},{},{},{}),'''
        sql=sql + row
    sql=sql[:-1]
    sql=sql+'''\n ON DUPLICATE KEY UPDATE 
        tract_section=VALUES(tract_section),
        pathology=VALUES(pathology),
        notes=VALUES(notes),
        inflammation=VALUES(inflammation),
        edemous_villi=VALUES(edemous_villi),
        bleed=VALUES(bleed),
        diffuse_bleed=VALUES(diffuse_bleed)'''
    #print(sql)
    vals=[vname]
    for item in values:
        vals=vals+item
    sql=sql.format(*vals)
    conn.execute(sql)


def get_max_frame(vname):
    cur=conn.execute('SELECT COUNT(*) FROM ' + vname)
    val=cur.fetchall()
    max_frame=val[0][0]-1-int(np.floor(len(config.frames)/2))
    return max_frame


## Function returns pandas table of labels for display

def get_video_df(vname):
    table=conn.execute('SELECT * FROM ' + vname)
    columns=table.keys()
    table=table.fetchall()
    table=pd.DataFrame(table, columns=columns)
    labelsdf=table[['index_', 'tract_section','notes', 'pathology','inflammation','edemous_villi','bleed','diffuse_bleed']]
    return labelsdf


## Function to Update Table With Pandas DF

def scan_for_new_videos():
    dir_videos=[x for x in os.listdir('/project/DSone/jaj4zcf/Videos/') if '.' not in x and x[0]=='v']
    print(str(dir_videos))
    names=conn.execute('SHOW TABLES')
    #names= conn.fetchall()
    names=[item[0] for item in names if item[0] not in ['prog_table']]
    print(str(names))
    not_loaded_videos=[x for x in dir_videos if x not in names and x not in ['prog_table']]

    for vname in not_loaded_videos:
    #Create Video Table If Does not Exist
        vid_folder='/project/DSone/jaj4zcf/Videos/'+vname
        allfiles=os.listdir(vid_folder)
        # Strip .jpg off file names
        allfiles=[int(item[:-4]) for item in allfiles]
        allfiles.sort()
        df=pd.DataFrame(columns=['index_', 'tract_section', 'pathology','inflammation','edemous_villi','bleed','diffuse_bleed', 'notes'])
        df['index_']=allfiles
        df['tract_section']=config.sectOptions[0]
        df['pathology']=config.abnormalOptions[0]
        df['inflammation']=0
        df['edemous_villi']=0
        df['bleed']=0
        df['diffuse_bleed']=0
        df['notes']=''

        #find video folder name
        index=vid_folder[len(vid_folder)::-1].find('/')
        vname=vid_folder[-index:]

        df.to_sql(vname,conn,if_exists='replace', 
                  dtype={'index_': Integer, 
                         'tract_section': Text,
                         'pathology':Text,
                         'notes':Text,
                         'inflammation':Integer,
                         'edemous_villi':Integer,
                         'bleed':Integer,
                         'diffuse_bleed':Integer}, index=False)
        conn.execute('''ALTER TABLE {}
                    ADD PRIMARY KEY(index_);'''.format(vname))
        
    

    #Update Progress Table

    vids=list(pd.read_sql('SELECT * FROM prog_table',conn)['video'])
    not_loaded_videos=[x for x in names if x not in vids]
    not_loaded_videos=[x for x in not_loaded_videos if x not in ['prog_table'] and x[0]=='v']
   
    for vname in not_loaded_videos:
        sql='''INSERT INTO prog_table (video, notes, progress)
        VALUES ("{}","No Notes" , "Not Started")'''.format(vname)
        cur=conn.execute(sql)



    
def update_video_row(vname, progress, notes):
    sql = ''' UPDATE prog_table
              SET progress = "{}" ,
                  notes = "{}" 
              WHERE video = "{}"'''
    sql=sql.format(progress,notes,vname)
    conn.execute(sql)


    
def get_vid_data_row(vname):
    sql = 'SELECT * FROM prog_table WHERE video = "{}"'.format(vname)
    cur=conn.execute(sql)
    row=cur.fetchall()
    return row[0]
