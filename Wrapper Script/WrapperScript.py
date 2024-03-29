import subprocess
import cx_Oracle
from sqlalchemy import create_engine
import pandas as pd
import time 
import os
import sys 


#Get User Input for Business Date
bus_date = input("Please enter Business Date in YYYYMMDD format: ")
print("The business date your entered is:", bus_date)
confirm = input("Do you want to proceed? Y/N: ")

def check(confirm):
    for i in range(1):
        if confirm != 'Y':
            print('Exiting Script')
            exit()
        
check(confirm)

print('Starting DQ LOG')


#Directory for the validation scripts
directory = ('/informatica/dev/infa_shared/Subledger/Scripts/Python')



#Putty command for DQ
dq =

#Create DB connection and run dq validation query
conn_rfs_dev = create_engine('oracle+cx_oracle:/)                            
dq_query = ()
dq_query = dq_query.replace("#BUSDATE#",bus_date)                            
dq_log = pd.read_sql(dq_query, con = conn_rfs_dev)

#Check DQ log again every 2 minutes to ensure completion, 
idx = 0
while dq_log.empty:
    print('Checking DQ Log Validation in 2 minutes')
    time.sleep(120)
    dq_log = pd.read_sql(dq_log, con = conn_rfs_dev)
    idx += 1
    if idx > 11:
        print('DQ Log script failed!')
        sys.exit()

#If DQ Log completes, check for Sev 1 errors
dq_log = dq_log['COUNT(*)']
dqlen = len(dq_log)

for i in range(1):
    if dq_log[0] != 0 or dqlen != 1:
        print('DQ Log has Severity 1 Issues!') 
        subprocess.run(sendMail, shell = True)
        sys.exit()
    else:
        print('DQ Log Sucessful - No Severity 1 Issues!')      


    #Start ODS Load if there are no Sev 1 errors
    nohup_ods = ()
    enter = ''    
    subprocess.run(nohup_ods, shell = True) 
    
    #Check ODS load status for failures every 5 minutes 
    ODS_status_sql = ()
    ODS_status_sql = ODS_status_sql.replace("#BUSDATE#",bus_date)
    ODS_status = pd.read_sql(ODS_status_sql, con = conn_rfs_dev)                               
                  
    idx2 = 0     
    while ODS_status.empty:
        print('Checking ODS load status in 5 minutes')
        time.sleep(300)
        ODS_status = pd.read_sql(ODS_status_sql
        idx2 += 1
        if idx2 > 16:
            print('ODS Load failed!')
            subprocess.run(sendMail, shell = True)
            sys.exit()  
        
    #Connect to database
    conn_rfs_dev = create_engine('oracle+cx_oracle://)
    
    #Validation query to check if ODS Load is successful
    ods_load_validation = ()
    ods_load_validation = ods_load_validation.replace("#BUSDATE#",bus_date)
                                                      
    #Read in validation query into a dataframe
    df = pd.read_sql(ods_load_validation, con=conn_rfs_dev)
    
    #If the validation query returns null, run the query again every 5 minutes for the next ~hour 
    idx3 = 0     
    while df.empty:
        print('Checking ODS load status in 5 minutes')
        time.sleep(300)
        df = pd.read_sql(ods_load_validation, con=conn_rfs_dev)
        idx3 += 1        
        if idx3 > 28:
            print('ODS Load failed!')
            subprocess.run(sendMail, shell = True)
            sys.exit()      
    
    #Start Validations if ODS load is successful
    ods_sts = df.stts_cd    
    errors = []
    
    #If the validation query returns 1 record = 'COMPLT', kick off validation queries
    if ods_sts.all() == 'COMPLT' and len(ods_sts) == 1:
    
        print('------------ODS Load Successful - Starting Validation Queries-------------')
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            if filename.endswith(".sql"): 
                #print(os.path.join(directory, filename))         
                fd = open(os.path.join(directory, filename))
                sqlFile = fd.read()
                query = sqlFile.replace('\r', ' ').replace('\n',' ')
                query = query.replace("#BUSDATE#",bus_date)
                qry_nm = filename.split('.')[0]
                print(qry_nm)
                
                if qry_nm.startswith('Sanity Checks'):
                    query = query.replace("#BUSDATE#",bus_date)
                    q_split = query.split(';')
                    q_split = q_split[:-1]                    
                    for i in q_split:
                        df2 = pd.read_sql(i, con = conn_rfs_dev)
                        if not df2.empty:
                            errors.append(i)
                else:
                    df1 = pd.read_sql(query, con = conn_rfs_dev)                    
                    qry_desc = [qry_nm, query]                    
                    if not df1.empty:                        
                        errors.append(qry_desc)                    

        errors2 = pd.Series(errors)
        errors2.to_csv('() 
        subprocess.run(sendMail, shell = True)
        
    # If ODS Load fails, create CSV and send mail
    else:
        print('--------------ODS Load Error----------------')
        errors2 = pd.Series()
        errors2.to_csv() 
        subprocess.run(sendMail, shell = True)

    
    
     

        

    

