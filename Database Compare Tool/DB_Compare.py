import cx_Oracle
from sqlalchemy import types, create_engine
import pandas as pd
import time
import numpy as np 
import os 
import string


ctff_dt = input("Please enter Business Date in YYYYMMDD format: ")
print("The business date your entered is:", ctff_dt)
directory = (r'/informatica/dev/infa_shared/Subledger/Scripts/Python')


##############################################################################################################
#
##############################################################################################################

#DB Connections
conn_rfs_sit = create_engine('oracle+cx_oracle://db connection details')
conn_rfs_prd = create_engine('oracle+cx_oracle://db connection details') 

#Dictionary for PK's 
pk_dict = {'SEL_TRNSCTN_HIST' : ['stg_trnsctn_id','ctff_dt'], 'SPLMNTL_ASST_STTS_OVRRD' : ['mss_loan_id', 'efctv_end_dt'], 'SEL_ARM': ['stg_arm_id'], 'SEL_LTGTN': ['mss_loan_id','ctff_dt','case_id', 'ltgtn_rsn_cd'], 'SEL_PRPTY_VALTN':['mss_loan_id','stg_valtn_id'], 'CAR_TRNSCTN_HIST':['car_trnsctn_id'], 'CAR_FEE_ACTVTY':['cn_fee_actvty_id'], 'CAR_FEE_DTL' : ['cn_fee_dtl_id'], 'CAR_CLM' : ['init_clm_fild_dt','ctff_dt', 'mss_loan_id'], 'CAR_CLM_SSPNSN':['mss_loan_id', 'ctff_dt', 'clm_sspnsn_start_dt'], 'CAR_CLM_RESBMT':['mss_loan_id', 'snpsht_dt', 'resbmt_dt']   }                             
                             

#Loop thru files in the directory to get SQL queries                             
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".sql"): 
        #print(os.path.join(directory, filename))         
        fd = open(os.path.join(directory, filename))
        sqlFile = fd.read()
        query = sqlFile.replace('\r', ' ').replace('\n',' ')
        tbl_nm = filename.split('.')[0]        
        query = query         
        
        if tbl_nm in pk_dict.keys():
            pk = (pk_dict[tbl_nm])
        else:
            pk = ['mss_loan_id', 'ctff_dt']    
            
        print(tbl_nm + str(pk))        

        start = time.time()
        print('Starting Data Load...')
         
		#Ready query into a dataframe 
        df1 = pd.read_sql(query, con=conn_rfs_sit)
        df2 = pd.read_sql(query, con=conn_rfs_prd)
        
        end = time.time()
        x = ('Data Load Time: ')
        y =  str(end - start)
        print(x + y)
        
        
        ##############################################################################################################
        #   Merge Data on PK, Sort by suffix (dev, tst, prd), Drop rows that match. Identify any missing/extra
        ##############################################################################################################
        
        start2 = time.time()
        print('Starting Compare...')
                

        if tbl_nm == 'CAR_TRNSCTN_HIST':
            df2.new_ctff_dt = df2.new_ctff_dt.astype(object)
            df1.new_ctff_dt = df1.new_ctff_dt.astype(object)
            df_merge2 = df1.merge(df2,indicator = True, how='outer')
        else:
            df_merge2 = df1.merge(df2,indicator = True, how='outer')
            df_merge2 = df_merge2.set_index(pk)
        
        
        # filter out rows that match 
        df_diffs = df_merge2.loc[lambda x : x['_merge']!='both']
        
        # Get rows that have differnces. Filter out extras/missing
        diff_ids = df_diffs.index.value_counts() > 1
        df_diff_only = df_diffs[diff_ids]
        
        # Sort DF by PK
        df_diff_only = df_diff_only.sort_index(axis = 0)
        
        # Iniialize blank DF for results
        df_diff_final = pd.DataFrame()
        
        
        if not df_diff_only.empty:
        
        
            #   Find Differences
            for i in df_diff_only.index.unique():
                df_test2 = df_diff_only.loc[i]
                colList = df_test2.columns.values.tolist()
                df_test2 = df_test2.replace(np.nan, '', regex=True)
                for c in colList:
                    if df_test2[c].nunique() == 1 :
                        df_test2.drop([c], axis = 1, inplace = True)            
                df_diff_final = df_diff_final.append(df_test2, sort = True)
        
        
              
            # Rename '_merge' columns to correspond w/ database environment & convert all column headers to upper-case
            df_diff_final = df_diff_final.replace('right_only', 'PRD')
            df_diff_final = df_diff_final.replace('left_only', 'SIT')            
            df_diff_final.columns = df_diff_final.columns.str.upper()
                    
        ### Find extras/missing

        
        #Filter out matching rows 
        df_extras = df_merge2.loc[lambda x : x['_merge']!='both']
        df_extras = df_extras.sort_index(axis = 0)
        
        #Find extra missing rows based on PK & environment
        extra_ids = df_extras.index.value_counts() == 1
        df_extras_only = df_extras[extra_ids]
        
        df_extras_prd = df_extras_only[df_extras_only['_merge'] == 'right_only']
        df_extras_dev = df_extras_only[df_extras_only['_merge'] == 'left_only']
        
        if not df_extras_prd.empty:
            df_extras_prd = df_extras_prd.replace('right_only', 'PRD')
            df_extras_prd.columns = df_extras_prd.columns.str.upper()
        
        if not df_extras_dev.empty:        
            df_extras_dev = df_extras_dev.replace('left_only', 'SIT')
            df_extras_dev.columns = df_extras_dev.columns.str.upper()
        
        
        print('Compare complete')
        
        end2 = time.time()
        x2 = ('Compare Time: ')
        y2 =  str(end2 - start2)
        print(x2 + y2)
        
        ##############################################################################################################
        #   If the DF is not empty, write to excel. Close Connections
        ##############################################################################################################
        

         
        if not df_diff_final.empty :
            print('DIFFERENCES FOUND!..................................Writing data to excel...')
            df_diff_final.to_csv()
            
        if not df_extras_prd.empty:
            print('EXTRAS FOUND!..................................Writing data to excel...')
            df_extras_prd.to_csv()
            
        if not df_extras_dev.empty:
            print('MISINGS FOUND!..................................Writing data to excel...')
            df_extras_dev.to_csv()
            
            
        
    
conn_rfs_sit.dispose()
conn_rfs_prd.dispose()



print('Finished...')




     


