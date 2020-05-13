import pandas as pd
import numpy as np
import datetime
import os

##################################################################################################################################################################
#Remaining months in 2020 & 2021

m1= ['3/1/2020', '4/1/2020','5/1/2020','6/1/2020', '7/1/2020','8/1/2020','9/1/2020','10/1/2020','11/1/2020','12/1/2020',  
        '1/1/2021','2/1/2021','3/1/2021', '4/1/2021','5/1/2021','6/1/2021', '7/1/2021','8/1/2021','9/1/2021','10/1/2021','11/1/2021','12/1/2021']
now = datetime.datetime.now()
x = (13 - now.month)
x2 = x + 12
months = m1[-x2:]
month_range = x2


#Number of remaining months in 2020
projected_port_consumption_rate = (x)/12

##################################################################################################################################################################

#Read Input Files
path = os.getcwd() + '\\source_files'
output_path = os.getcwd()

for file in os.listdir(path):
    if ('CF') in file:
        df_cf_hubs = pd.read_csv(path + '\\' + file)
    elif 'Data_Export' in file:
        df_metrocore = pd.read_csv(path + '\\' + file)        
    elif 'Export_Data' in file:
        df_ac = pd.read_csv(path + '\\' + file)  
    elif 'FP4' in file:
        dffp4 = pd.read_csv(path + '\\' + file) 
    elif ('hist_lag_imb') in file:
        df_imb_hist = pd.read_csv(path + '\\' + file) 
    elif ('Historical_Access') in file:
        df_ac_historical = pd.read_csv(path + '\\' + file) 
    elif ('Historical_Metro') in file:
        df_mc_historical = pd.read_csv(path + '\\' + file) 
    elif ('HistoricalPort') in file:
        dfx = pd.read_csv(path + '\\' + file) 
    elif ('OU_Pair') in file:
        df_pairs  = pd.read_csv(path + '\\' + file) 
    elif ('Port_Level') in file:
        df_ports   = pd.read_csv(path + '\\' + file) 
    elif ('SAPs') in file:
        dfsap  = pd.read_csv(path + '\\' + file)  
    elif ('upgrade_')  in file:  
        df_pricing  = pd.read_csv(path + '\\' + file) 
    elif ('sales_')  in file:  
        df_sales  = pd.read_csv(path + '\\' + file)
    elif ('data_definitions')  in file:  
        data_definitions  = pd.read_csv(path + '\\' + file)
    elif ('Belinda')  in file:  
        df_belindas_list  = pd.read_csv(path + '\\' + file)


for file in os.listdir(output_path):
    if ('Scenario_Input') in file and ('~') not in file:
        scenario_input_template = pd.read_excel(output_path + '\\' + file, sheet_name = 'Input') 
        
        
        




############################################################################################################################################################################
#Temp function to reflect XRS20 new 100G cards

def _100G_port_counts():
    xrs20 = ['7950-XRS20']    
    df_ports.loc[(df_ports['License Type'].isnull()) & (df_ports['Port Type'] == '100 Gigabit Ethernet') & (df_ports['Node Type'].isin(xrs20)) ,'Final Port Count'] =  df_ports.loc[(df_ports['License Type'].isnull()) & (df_ports['Port Type'] == '100 Gigabit Ethernet') & (df_ports['Node Type'].isin(xrs20)) ,'Final Port Count'] + 12
    df_ports.loc[ (df_ports['Port Type'] == '10-Gig Ethernet SF') & (df_ports['Node Type'].isin(xrs20)) ,'Final Port Count'] =  df_ports.loc[(df_ports['Port Type'] == '10-Gig Ethernet SF') & (df_ports['Node Type'].isin(xrs20)) ,'Final Port Count'] + 64

    return(df_ports)
    
df_ports =   _100G_port_counts()  



 ############################################################################################################################################################################
#Merge Historical Metro Core Data w/ 2020 Data and Remove Duplicates
   
def clean_merge_metro_core():    
    df_mc_historical['Key1'] = df_mc_historical['Router IP']+df_mc_historical['Lag']
    df_metrocore['Key1'] = df_metrocore['Router IP']+df_metrocore['Lag']
    df_mc = df_mc_historical.append(df_metrocore)
    df_mc['Lag'] = df_mc['Lag'].astype(str)
    df_mc2 = df_mc[df_mc['Lag'].str.contains("lag")]
    result = df_mc2.loc[:,('Pair Code')]
    df_mc2.loc[:,('Key')] = result    
    df_mc2.loc[:,('Key1')] = df_mc2['Router IP']+df_mc2['Lag']
    df_mc2 = df_mc2.drop_duplicates(subset = ['Month of Month','Pair Code','Router IP','Lag','Far End IP'])        
    return(df_mc2)
    
df_mc = clean_merge_metro_core()  




#Ad-hoc

def adhoc():
    df = df_mc.copy()
    df2 = df.loc[df['Month of Month']=='1/1/2020']
    keys = df_pairs['Pair Code'].unique().tolist()   
    df3 = df2.loc[df2['Pair Code'].isin(keys)]
    df3.loc[:,('Month of Month')] = pd.to_datetime(df3['Month of Month'])   
    df3.loc[:,('Pair Utilization ')] = df3['Pair Utilization '].str.replace('%','')
    df3.loc[:,('Pair Utilization ')] = pd.to_numeric(df3['Pair Utilization ']) 
    df3.loc[:,('Pair Utilization ')] = df3.loc[:,('Pair Utilization ')] * 1.8
    df4 = df3.loc[df3['Pair Utilization ']>= 70]
    df10 = df4.loc[df4[ 'Lag Capacity (Gbps)']< 50]
    nodes10 = df10['Router'].tolist() + df10['Far End Router '].tolist()
    df100 = df4.loc[df4[ 'Lag Capacity (Gbps)'] >= 50]
    nodes100 = df100['Router'].tolist() + df100['Far End Router '].tolist()
    return(nodes10, nodes100)

nodes10, nodes100 = adhoc()



############################################################################################################################################################################






def ports():
    #Merge current Port Inventory with Historical Port Consumption static sheet
    df_ports['Key'] = df_ports['IP'] + df_ports['Port Type'] 
    df = df_ports[['Key','Region', 'Market ', 'Facility', 'TLA', 'Node Type', 'Node Name', 'IP', 'Role', 'Port Type','License Type','Available Ports', 'Granite Pending Port Count', 'Final Port Count','Slots Available', 'Total Port Count', 'Used Ports','MDA Free Subslot Count','MDA Free Subslot']]
    dfx['Key'] = dfx['IP'] + dfx['Port Type'] 
    dfx2 = dfx[[ 'Key','2018 Port Consumption', '2019 Port Consumption', 'Hub Class', 'Dec 2019 Used Ports']]
    df1 = df.merge(dfx2, on = 'Key', how = 'left')
    df1['2018 Port Consumption'].replace(np.nan, 0, inplace = True)
    df1['2019 Port Consumption'].replace(np.nan, 0, inplace = True)
   
    
    #Calculate YTD port consumption
    ytd_ports_cons = df1['Used Ports'] - df1['Dec 2019 Used Ports']
    df1.drop('Dec 2019 Used Ports', axis = 1, inplace = True)
    df1.insert(df1.columns.get_loc('2019 Port Consumption')+1, '2020 YTD Port Consumption', ytd_ports_cons)
    
    
    #Get projected ports available for ports not equal to 100G since 100G ports only have 1 year of historical data
    df10 = df1.loc[lambda x:x['Port Type']!='100 Gigabit Ethernet']
    avg_cons = ((df10['2018 Port Consumption'] + df10['2019 Port Consumption']) /2) 
    df10.insert(df10.columns.get_loc('2019 Port Consumption')+1,"Average Yearly Port Consumption", avg_cons )
    
    
    #Add historical port consumption from capped hub to new hub for 10G
   
    
    #Calculate Projected Ports Available 
    t1 = df10['Final Port Count'] - (df10['Average Yearly Port Consumption'] * projected_port_consumption_rate)
    t2 = t1 - df10['Average Yearly Port Consumption']
    df10.insert(df10.columns.get_loc('2019 Port Consumption')+2,"End of 2020 Projected Ports Available", t1 )
    df10.insert(df10.columns.get_loc('2019 Port Consumption')+3,"End of 2021 Projected Ports Available", t2 )
    
      
    
    #Get projected ports available for portsequal to 100G since 100G ports only have 1 year of historical data
    df100 = df1.loc[lambda x:x['Port Type']=='100 Gigabit Ethernet']
    avg_cons = df100['2019 Port Consumption']
    x = df100.columns.get_loc('2019 Port Consumption')
    df100.insert(x+1,"Average Yearly Port Consumption", avg_cons )
    
    
    #Calculate Projected Ports Available 
    t1 = df100['Final Port Count'] - (df100['Average Yearly Port Consumption'] * projected_port_consumption_rate)
    t2 = t1 - df100['Average Yearly Port Consumption']
    df100.insert(df100.columns.get_loc('2019 Port Consumption')+2,"End of 2020 Projected Ports Available", t1 )
    df100.insert(df100.columns.get_loc('2019 Port Consumption')+3,"End of 2021 Projected Ports Available", t2 ) 
    

    #Combine the 10G & 100G Projected Port DF's
    df1 = df10.append(df100)           
    
    #Get list of capped hubs & create new field
    df1['Facility Status'] = 'Not Migrating/Not Capped' 
    capped_migrating = list(zip(df_cf_hubs['Node Name'].dropna(), df_cf_hubs['Facility Status'].dropna()))
    for a,b in capped_migrating:
        df1.loc[(df1['Node Name'] == a) , 'Facility Status'] = b 

    
    new_cf = list(zip(df_cf_hubs['Secondary Node'].dropna(), df_cf_hubs['Facility Status'].dropna()))    
    for a,b in new_cf:
        df1.loc[(df1['Node Name']== a), 'Facility Status'] = 'New Primary CF'
    
    return (df1)

dfx = ports()






def migrations():
    df = dfx.copy()
    core = ['S-PE', 'HYBRID']
    df1 = df.loc[lambda x:x['Role'].isin(core)]
    df1 = df1.drop_duplicates(subset = 'Node Name')
    df1 = df1[df1['TLA'].notnull()]
    df2 = df1[['TLA', 'Node Type']]
    key = df2['TLA'].unique()
    fp3 = ['7750-SR12', '7750-SR7']
    fp4 = ['7750 SR-14s','7950-XRS20','7750 SR-7s']
    
    migrations = []    
    for i in key:
        dfi = df2[df2['TLA']== i]
        df3 = len(dfi[dfi['Node Type'].isin(fp3)])
        df4 = len(dfi[dfi['Node Type'].isin(fp4)])
        if df3 >= 1 and df4 >= 1:
            migrations.append(i)
    df['Facility is FP3 and FP4 enabled']  =  'No'   
    for i in migrations:
        df.loc[(df['TLA']==i) & (df['Role']=='S-PE'), 'Facility is FP3 and FP4 enabled'] = 'Yes'       
          
    
    return(df)
    
dfx = migrations()  







def router_platform():
    df = dfx.copy()
    fp4 = ['7750 SR-14s','7950-XRS20','7750 SR-7s']
    df.loc[~df['Node Type'].isin(fp4), 'Router Platform'] = 'FP3'
    df.loc[df['Node Type'].isin(fp4), 'Router Platform'] = 'FP4'   
    
    return(df)
    
dfx =   router_platform()   




def port_migrations():  
    dfm = dfx.loc[lambda x:x['Facility is FP3 and FP4 enabled']=='Yes']        
    tla = dfm['TLA'].unique()    
    for t in tla:
        df1 = dfm[dfm['TLA']==t] 
        old = df1[df1['Router Platform']=='FP3']
        new = df1[df1['Router Platform']=='FP4']


        for p in new['Port Type'].unique():
            if p in old['Port Type'].unique().tolist():   
                fp3_used_ports = old['Used Ports'].loc[old['Port Type'] == p].max()
                fp3_avg_cons = old['Average Yearly Port Consumption'].loc[old['Port Type'] == p].max()
                fp4_used_ports = new['Used Ports'].loc[new['Port Type'] == p].max()
                fp4_avg_cons = new['Average Yearly Port Consumption'].loc[new['Port Type'] == p].max()
                total_cons_with_migrations = fp3_used_ports +  fp4_used_ports
                fac_level_port_cons_without_migrations = fp3_avg_cons + fp4_avg_cons
                for k in new['Key'].unique():
                    dfx.loc[(dfx['Key']==k) & (dfx['Port Type']== p), 'Total Port Consumption with FP3/FP4 PE Migrations'] = total_cons_with_migrations
                    dfx.loc[(dfx['Key']==k) & (dfx['Port Type']== p), 'Facility_Level Avg Yearly FP3 + FP4 Port Consumption'] = fac_level_port_cons_without_migrations
    
    
    
    cf_migrating = df_cf_hubs[df_cf_hubs['Facility Status']=='Migrating']
    hubs = list(zip(cf_migrating['IP'].dropna(), cf_migrating['Secondary IP'].dropna()))
    for a,b in hubs:
        old = dfx[dfx['IP']==a]
        new = dfx[dfx['IP']==b]
        for p in new['Port Type'].unique():
            if p in old['Port Type'].unique().tolist():
                fp3_used_ports = old['Used Ports'].loc[old['Port Type'] == p].max()
                fp3_avg_cons = old['Average Yearly Port Consumption'].loc[old['Port Type'] == p].max()
                fp4_avg_cons = new['Average Yearly Port Consumption'].loc[new['Port Type'] == p].max()
                fp4_used_ports = new['Used Ports'].loc[new['Port Type'] == p].max()
                total_cons_with_migrations = fp3_used_ports +  fp4_used_ports
                fac_level_port_cons_without_migrations = fp3_avg_cons + fp4_avg_cons
                for k in new['Key'].unique():
                    dfx.loc[(dfx['Key']==k) & (dfx['Port Type']== p), 'Total Port Consumption after CF Migrations'] = total_cons_with_migrations
                    dfx.loc[(dfx['Key']==k) & (dfx['Port Type']== p), 'Facility_Level Avg Yearly Port Consumption with CF Migrations'] = fac_level_port_cons_without_migrations
    

    
    fp3_fp4_migr = dfx['Total Port Count'] - dfx['Total Port Consumption with FP3/FP4 PE Migrations']
    cf_migr = dfx['Total Port Count'] - dfx['Total Port Consumption after CF Migrations']
    dfx.insert(dfx.columns.get_loc('Facility_Level Avg Yearly FP3 + FP4 Port Consumption')+1,'Ports Available after FP3/FP4 Migrations',fp3_fp4_migr)
    dfx.insert(dfx.columns.get_loc('Facility_Level Avg Yearly Port Consumption with CF Migrations')+1,'Ports Available after CF Migrations',cf_migr)

    return(dfx)
    

dfx = port_migrations()





############################################################################################################################################################################

def poi(): 
    #Merge port data with static POI data to determine when facilities are receiving FP4 routers
    #dfx_final = dfx.copy()
    defer = dffp4.loc[dffp4['FP4 Need By Date']== '1/15/2021']
    defer_tlas = defer['TLA'].unique().tolist()
    defer_node_names = defer['Node Name'].unique().tolist()
    
    
    #P Routers & Hybrids
    dffp4['tla_key'] = dffp4['TLA'] + dffp4['Role']
    df = dffp4.loc[lambda x:x['Role']=='P ROUTER']
    dfx2 = dfx.copy()
    dfx['tla_key'] = dfx['TLA'] + dfx['Role']    
    dfx2 = dfx.loc[lambda x:x['Role'] == 'P ROUTER']
    dfz = df[['tla_key','FP4 Router Type', 'FP4 Need By Date']].drop_duplicates()
    df1 = dfx2.merge(dfz,  on = 'tla_key', how = 'left')      

    #PE Routers   
    spe_hybrid = ['HYBRID', 'S-PE']
    dfpe = dffp4.loc[lambda x:x['Role']!= 'P ROUTER']
    dfx2 = dfx.loc[lambda x:x['Role'] != 'P ROUTER']
    dfpe2 = dfpe[['Node Name','FP4 Router Type', 'FP4 Need By Date']]
    df2 = dfx2.merge(dfpe2, on = 'Node Name', how = 'left')
    
    
    dfb = df1.append(df2) 
    dfb.drop_duplicates(inplace = True)

    
    #Get list of facilities which already have a pair of FP4 PE routers
    df = dfb.copy()
    df.loc[df['Facility'] == 'New Holt', 'TLA'] = 'NHT'
    df.loc[df['Facility'] == 'Poydras', 'TLA'] = 'POY'
    tlas = df['TLA'].loc[(df['Router Platform'] == 'FP4') & (df['Role'].isin(spe_hybrid))].unique()    
    df['Facility is currently FP4 enabled'] = 'No'    
    df['Facility will have FP4 PE routers by the end of 2020'] = 'No'
    df['2020 Deferment'] = 'No'
    for i in tlas:
        df.loc[df['TLA']==i, 'Facility is currently FP4 enabled'] = 'Yes'  
        df.loc[df['TLA']==i, 'Facility will have FP4 PE routers by the end of 2020'] = 'Yes' 
        
    #Get list of facilities which are recieving FP4 PE routers in 2020 including the Hybrids
    #hybrid = dffp4['TLA'].loc[(dffp4['FP4 Role'] == 'HYBRID')].drop_duplicates().tolist()
    fp4 = dfpe['TLA'].drop_duplicates().tolist()
    #fp4 = hybrid + pe
    planned =  [x for x in fp4 if x not in tlas]
    planned_minus_deferred =  [x for x in planned if x not in defer_tlas]
    
    for i in planned_minus_deferred:
        df.loc[df['TLA']==i, 'Facility will have FP4 PE routers by the end of 2020'] = 'Yes'    
        
        
    for i in defer_node_names:
        df.loc[df['Node Name']==i, '2020 Deferment'] = 'Yes'  
       
    
    df2 = df.merge(dfsap, how = 'outer', on = 'Node Name')
    df2.drop('Key', axis = 1, inplace = True)
    return(df2)
    
dfx = poi()




############################################################################################################################################################################

def secondary_pair():
    #Function to determine which facilities have secondary pairs
    df1 = df_ports[['TLA','Node Name', 'Node Type','Port Type', 'Role']]
    df2 = df1.loc[lambda x:x['Port Type']=='10-Gig Ethernet SF']
    df2 = df2.loc[lambda x:x['Role']=='S-PE'] 
    df2.dropna(subset = ['TLA'],axis = 0, inplace = True)
    fp4 = ['7750 SR-14s','7950-XRS20','7750 SR-7s']
    df2 = df2[~df2['Node Type'].isin(fp4)]
    df3 = df2.groupby('TLA').filter(lambda x: len(x) > 3)
    tla = df3['TLA'].unique()
    secondary = []
    for i in tla:
        df = df3[df3['TLA']==i]
        if df['Node Name'].nunique() >= 4:
            secondary.append(df['Node Name'].tolist())            
    secondary1 = [y for x in secondary for y in x]        
    dfx['Secondary Pair'] = dfx.apply(lambda x: int(x['Node Name'] in secondary1), axis=1)  
    dfx['Secondary Pair'] = dfx['Secondary Pair'].replace({1: 'Yes', 0:'No', np.nan:'No'})
    return(dfx)
    
dfx = secondary_pair() 


############################################################################################################################################################################


def _100G_enabled():    
    df = dfx.copy()
    df_100 = df_pairs.loc[lambda x:x['Pair Capacity (Gbps)']==100]
    nodes = df_100['Router'].unique().tolist()+ df_100['Far End Router '].unique().tolist()
    df['Router has a 100G LAG'] = 'No'
    for n in nodes:
        df.loc[df['Node Name']==n, 'Router has a 100G LAG'] = 'Yes'      
    
    return(df)

dfx = _100G_enabled()



    
############################################################################################################################################################################




############################################################################################################################################################################

def imb():
    df_imb_hist['Key'] = df_imb_hist['ipaddress'] + df_imb_hist['lag']
    df = df_imb_hist.drop_duplicates(subset = ['ipaddress', 'lag', 'Month'])
    df["Lag Imbalance"] = df[["lagutil_imb.lag_ingress_95_pct", "lagutil_imb.lag_egress_95_pct"]].max(axis=1)
    df = df.round({'Lag Imbalance': 1})
    df['Lag Imbalance'].replace(0,np.nan,inplace = True)
    df = df.loc[lambda x:x['Lag Imbalance'] <= 100]   
    df['lag_capacity'] = df['lag_capacity'].str.split(',').str[0]
    #Get list of lags in current month report
    df['Month'] = pd.to_datetime(df['Month'])
    current_imb = df.loc[lambda x:x['Month'] == df['Month'].unique().max()] 
    current_imb1 = current_imb.loc[current_imb['Lag Imbalance'] != 0]
    keys = current_imb1['Key']
    
    
    df_final = pd.DataFrame()  
    
    for j in keys:
        dfi = df.loc[df['Key']==j]
 
        dfi.sort_values(by = 'Month',   inplace = True)        
        x = dfi['Lag Imbalance'].pct_change()
        x = x.replace([np.inf, -np.inf], np.nan)
        y = x.mean() 
        if y < 0:
            y = 0 
        elif y > 0.06:
            y = 0.015
        

        e = int(dfi['Lag Imbalance'].tail(1))    
        r = y + 1    
        forecast_util = []        
        for i in range(0,month_range):
            z = e*r
            forecast_util.append(z)
            e = z    
            
        test = pd.DataFrame()    
        test['Month'] = months
        ip = str(dfi['ipaddress'].unique()[0])
        test['ipaddress'] = ip
        test['Lag Imbalance'] = forecast_util
        test['Key'] = j
        test['lag_capacity'] = int(dfi['lag_capacity'].tail(1))
        
        df1 = dfi[['Month','ipaddress','Lag Imbalance'  ,'Key', 'lag_capacity']]
    
        jn = pd.concat([df1,test])  
        df_final = df_final.append(jn) 
        
    df_final['Month'] = pd.to_datetime(df_final['Month'])
  
    
    
    return(df_final)
    
imb_forecast = imb()    



############################################################################################################################################################################  


def lag_imb_augment():
    df_imb_hist['Month'] = pd.to_datetime(df_imb_hist['Month'])    
    df = imb_forecast[imb_forecast['Month'] >= df_imb_hist.Month.unique().max()] 
    df1 = df[df['Lag Imbalance'] >= 50]
    df1.drop_duplicates(subset = ['Key'], inplace = True)
    ips = dfx[['Node Name', 'IP']]
    df1.rename(columns= {'ipaddress':'IP'}, inplace = True)
    df2 = df1.merge(ips, on = 'IP')
    df2.drop_duplicates(inplace = True)
    df2['lag_capacity'] =  df2['lag_capacity'].astype('int32')    
    lag_aug_100G = df2['Node Name'][df2['lag_capacity']>=50].tolist()
    lag_aug_10G = df2['Node Name'][df2['lag_capacity'] < 50].tolist()
    
    df = dfx.copy()
    df['Number of Forecasted 100G Augments'] = 0
    df['Number of Forecasted 10G Augments'] = 0
    df['Key'] = df['Node Name'] + df['Port Type']
    for a in lag_aug_100G:
        key = str(a) + '100 Gigabit Ethernet'
        if key in df.Key.unique().tolist():
            df.loc[(df['Key'] == key) , 'Number of Forecasted 100G Augments'] =  float(df.loc[(df['Key'] == key) , 'Number of Forecasted 100G Augments']) + 2.0
    
    for b in lag_aug_10G:
        key = str(b) + '10-Gig Ethernet SF'
        if key in df.Key.unique().tolist():
            df.loc[(df['Key'] == key) , 'Number of Forecasted 10G Augments'] =  float(df.loc[(df['Key'] == key) , 'Number of Forecasted 10G Augments']) + 2.0


    return(df)
    
dfx =  lag_imb_augment()



############################################################################################################################################################################


def adhoc2():
    
    df = dfx.copy()

    df['Key'] = df['Node Name'] + df['Port Type']
    for i in nodes10:
        df.loc[df['Node Name']==i, 'Number of Forecasted 10G Augments'] =  df.loc[df['Node Name']==i, 'Number of Forecasted 10G Augments'] + 1
    
    for i in nodes100:
        df.loc[df['Node Name']==i, 'Number of Forecasted 100G Augments'] =  df.loc[df['Node Name']==i, 'Number of Forecasted 100G Augments'] + 1
    
    
    return(df)

dfx = adhoc2()


############################################################################################################################################################################


def new_uplinks():
    dfa = dfx.copy()   
    fp4nodes = dfa['TLA'] + dfa['Node Type']
    fp4nodes = fp4nodes.drop_duplicates().tolist()
    df = dffp4.copy()
    pe = df.loc[lambda x:x['Role'] == 'S-PE']
    # Filter out Nodes from FP4 Sites which are already in CB PAS Port Inv
    pe.loc[:,('Key')] = pe['TLA'] + pe['FP4 Router Type']    
    pe1 = pe[~pe['Key'].isin(fp4nodes)]
    pe2 = pe1.drop_duplicates(subset = ['TLA'] )
    market = pe2['Market'].tolist()     
    dfa['Number of New Uplinks not yet installed in 2020 based on POI'] = 0    
    for i in market:        
        dfa.loc[(dfa['Market '] == i) & (dfa['Role']=='P ROUTER')  & (dfa['Port Type']=='100 Gigabit Ethernet'), 'Number of New Uplinks not yet installed in 2020 based on POI'] =    dfa.loc[(dfa['Market '] == i) & (dfa['Role']=='P ROUTER')  & (dfa['Port Type']=='100 Gigabit Ethernet'), 'Number of New Uplinks not yet installed in 2020 based on POI'] + 1

    return(dfa)      
    
dfx = new_uplinks()


    
############################################################################################################################################################################

def sales_forecast_total_100G():
    df = dfx.copy()
    df['Sales_Forecast(100G)'] = 0
    sales = list(zip(df_sales['Facility'].dropna(), df_sales['# of Sites'].dropna()))    
    for a,b in sales:
        df.loc[(df['Facility'] == a) ,'Sales_Forecast(100G)'] =  df.loc[(df['Facility'] == a) ,'Sales_Forecast(100G)'] + b
        
    df['Total Projected 100G Growth'] =   df['Number of New Uplinks not yet installed in 2020 based on POI'] + df['Number of Forecasted 100G Augments'] + df['Sales_Forecast(100G)']
    
    return(df)    

dfx = sales_forecast_total_100G()


############################################################################################################################################################################


def ad_hoc_poi():
    df = dfx.copy()
    dfpoi = df_belindas_list.copy()   
    df['Number of new uplinks based on 2021 POI'] = 0
    dfpoi1 = dfpoi[[ 'Market ', 'Unique ID']].drop_duplicates()
    markets = dfpoi1['Market '].tolist()
    for i in markets:
        df.loc[(df['Market ']==i) & (df['Role']=='P ROUTER'),'Number of new uplinks based on 2021 POI'] = df.loc[(df['Market ']==i) & (df['Role']=='P ROUTER'), 'Number of new uplinks based on 2021 POI'] + 1

    
    df['Total Projected 100G Growth'] = df['Total Projected 100G Growth'] + df['Number of new uplinks based on 2021 POI']
    
    return(df)

dfx = ad_hoc_poi()

############################################################################################################################################################################






def projected_100G_port_cons():
    df = dfx.copy()
    proj_port_con_100 = df['End of 2021 Projected Ports Available'] - df['Total Projected 100G Growth']
    proj_port_con_10G = df['End of 2021 Projected Ports Available'] - df['Number of Forecasted 10G Augments']
    df.insert(df.columns.get_loc('End of 2021 Projected Ports Available')+1, 'End of 2021 Projected Ports available minus Total 100G Growth', proj_port_con_100)
    df.insert(df.columns.get_loc('End of 2021 Projected Ports Available')+2, 'End of 2021 Projected Ports available minus Total 10G Growth', proj_port_con_10G)

    return(df)

dfx = projected_100G_port_cons()

df = dfx.loc[dfx['Port Type']=='100 Gigabit Ethernet']






############################################################################################################################################################################

def scenario_input():
    model = scenario_input_template.copy()
    model.set_index(model.columns[0], inplace = True)
    model.rename(index = {'Router Role':'Role','FP4 Router Type':'Node Type'}, inplace = True)
    model['Parameters'] = model['Parameters'].replace(np.nan, '')
    model['Parameters'] = model['Parameters'].str.strip()
    #dfx['Projected number of New Uplinks in 2021 based on Router Upgrades'] = 0 
    dfx['pairs_key'] = dfx['TLA'] + dfx['Port Type'] + dfx['Role']
    dfx['market_key'] = dfx['Market '] + dfx['Role'] + dfx['Port Type']
    dfs = dict()
    
       
       




    #########################################################################################

    df_all_upgrades = pd.DataFrame()
    #Start at 6 to disregard the router upgrade inputs since we are using a static list provided by Belinda
    for i in model.columns[6:]:
        #print(i)
        #i = 'NEW SATELLITE BOX 1B'
        df = dfx.copy()        
        df = df[df['TLA'].notnull()]
        df = df[~df['Facility Status'].isin(['Migrating', 'Capped'])]
        df_verify = df.copy()
        df1 = model[['Parameters',i]].dropna(how = 'any')
        drop_down = df1[i][(df1['Parameters'] == 'Drop-down') |(df1['Parameters'] == '') ]
        greaterthan = df1[i][(df1['Parameters'] == '>=') ]
        lessthan = df1[i][(df1['Parameters'] == '<=')]
        dic_dropdown = drop_down.to_dict()
        dic_greaterthan = greaterthan.to_dict()
        dic_lessthan = lessthan.to_dict()
        if dic_dropdown:
            for a,b in dic_dropdown.items():
                if a in df.columns:
                    if b != 1:
                        c = b.split(sep = ', ')
                    else:
                        c = [b]
                    df = df.loc[df[a].isin(c)]                                  
                else:
                    print('WARNING! Input value does not exist: ' + a )
        if dic_greaterthan:
            for a,b in dic_greaterthan.items():
                if a in df.columns:
                    df = df.loc[(df[a] >= b)]
                else:
                    print('WARNING! Input value does not exist: ' + a )
        if dic_lessthan:
            for a,b, in dic_lessthan.items():
                if a in df.columns:
                    df = df.loc[(df[a] <= b)]
                else:
                    print('WARNING! Input value does not exist: ' + a )
        
        
        if 'ROUTER UPGRADE' in i and len(df) != len(df_verify) and not df.empty:            
            pe = df.loc[(df['Role']=='S-PE') | (df['Role']=='HYBRID')]
            pairs_key = pe['pairs_key'].unique().tolist()
            df = dfx[dfx['pairs_key'].isin(pairs_key)]
        elif 'PE ROUTER' in i and len(df) != len(df_verify) and not df.empty:
            pairs_key = df['pairs_key'].unique().tolist()
            df = dfx[dfx['pairs_key'].isin(pairs_key)]
        elif 'P ROUTER' in i and len(df) != len(df_verify) and not df.empty:
            market_key =  df['market_key'].unique().tolist()
            df = dfx[dfx['market_key'].isin(market_key)]            
        elif 'SATELLITE' in i and len(df) != len(df_verify) and not df.empty:    
            pe = df.loc[(df['Role']=='S-PE') | (df['Role']=='HYBRID')]
            pairs_key = pe['pairs_key'].unique().tolist()
            df_pe = dfx[dfx['pairs_key'].isin(pairs_key)]  
            df_pe1 = df_pe[df_pe['Router Platform']=='FP4']
            p = df.loc[df['Role']=='P ROUTER']
            market_key =  p['market_key'].unique().tolist()
            df_p = dfx[dfx['market_key'].isin(market_key)]        
            df = df_pe1.append(df_p)
       
        
                
        '''
        if 'ROUTER UPGRADE' in i and len(df) != len(df_verify) and not df.empty:
            #Get list of tlas for nodes identified
            p = df.drop_duplicates(subset = 'TLA')        
            market = p['Market '].tolist()
            
            for k in market:
                dfx.loc[(dfx['Market '] == k) & (dfx['Role']=='P ROUTER') , 'Projected number of New Uplinks in 2021 based on Router Upgrades'] =    dfx.loc[(dfx['Market '] == k) & (dfx['Role']=='P ROUTER') , 'Projected number of New Uplinks in 2021 based on Router Upgrades'] + 1
                dfx.loc[(dfx['Market '] == k) & (dfx['Role']=='P ROUTER') ,'End of 2021 Projected Ports available minus Total 100G Growth'] =    dfx.loc[(dfx['Market '] == k) & (dfx['Role']=='P ROUTER') , 'End of 2021 Projected Ports available minus Total 100G Growth'] - 1
                #dfx.loc[(dfx['Port Type']=='100 Gigabit Ethernet') , 'End of 2021 Projected Ports available minus Total 100G Growth'] = dfx.loc[ (dfx['Port Type']=='100 Gigabit Ethernet') , 'End of 2021 Projected Ports available minus Total 100G Growth'] -  dfx.loc[ (dfx['Port Type']=='100 Gigabit Ethernet') , 'Projected number of New Uplinks in 2021 based on Router Upgrades'] 

        '''   
        if not df.empty and len(df) != len(df_verify):
                        
            df_upgrade = pd.DataFrame()            
            df_upgrade['Node Name'] = df['Node Name'].drop_duplicates()
            df_upgrade['Upgrade Type'] = i
            df_upgrade['2020 Deferment'] = 'No'
            df_test = df
            dfs.update({i:df_test})                
            df_all_upgrades = df_all_upgrades.append(df_upgrade)
            
   
            
    
            
    ######################################################################################################        
    #2020 Deferments
    defer = dfx.loc[dfx['2020 Deferment']=='Yes']   
    
    #rank 3A
    nodes_3a = defer.loc[(defer['Slots Available']>=3) , 'Node Name'].unique().tolist()     
    df_defer_3a = pd.DataFrame()    
    df_defer_3a['Node Name'] = nodes_3a
    df_defer_3a['Upgrade Type'] = 'ROUTER UPGRADE: 3A'
    df_defer_3a['2020 Deferment'] = 'Yes'
    df_all_upgrades = df_all_upgrades.append(df_defer_3a)
    
    #rank 2A
    nodes_2a = defer.loc[(defer['Slots Available']==1) | (defer['Slots Available']==2), 'Node Name'].unique().tolist() 
    df_defer_2a = pd.DataFrame()    
    df_defer_2a['Node Name'] = nodes_2a
    df_defer_2a['Upgrade Type'] = 'ROUTER UPGRADE: 2A'
    df_defer_2a['2020 Deferment'] = 'Yes'
    df_all_upgrades = df_all_upgrades.append(df_defer_2a)
    
    #rank 1A
    nodes_1a = defer.loc[(defer['Slots Available']==0) | (defer['Total Projected 100G Growth']> 0) | (defer['# of SAPs >= 10G']> 5), 'Node Name'].unique().tolist() 
    df_defer_1a = pd.DataFrame()    
    df_defer_1a['Node Name'] = nodes_1a
    df_defer_1a['Upgrade Type'] = 'ROUTER UPGRADE: 1A'
    df_defer_1a['2020 Deferment'] = 'Yes'
    df_all_upgrades = df_all_upgrades.append(df_defer_1a)  


    
    
    ######################################################################################################
        
        
        
    dfx_cut = dfx[[ 'TLA','Region', 'Market ', 'Facility','Node Name']].drop_duplicates()
    df_final = df_all_upgrades.merge(dfx_cut, how = 'left', on = 'Node Name')    
    df_final['Rank'] = df_final['Upgrade Type'].str.split(":").str[1]
    df_final['Upgrade Type']= df_final['Upgrade Type'].str.split(":").str[0]
    df_final.sort_values(by = 'Rank', inplace = True)
    df_final.drop_duplicates(subset = ['Node Name', 'Upgrade Type', 'TLA', 'Region', 'Market ', 'Facility'], inplace = True)
    
    dfx_cut2 = dfx[[ 'TLA','Region', 'Market ', 'Facility','Node Name', 'Role']].drop_duplicates()
    df_final2 = df_all_upgrades.merge(dfx_cut2, how = 'left', on = 'Node Name')    
    df_final2['Rank'] = df_final2['Upgrade Type'].str.split(":").str[1]
    df_final2['Upgrade Type']= df_final2['Upgrade Type'].str.split(":").str[0]
    df_final2.sort_values(by = 'Rank', inplace = True)
    df_final2.drop_duplicates(subset = ['Node Name', 'Upgrade Type', 'TLA', 'Region', 'Market ', 'Facility', 'Role'], inplace = True)
        
    return(dfx, df_final, df_final2)  
        


dfx, df_final,df_final2 = scenario_input()   






############################################################################################################################################################################

def output():
    df1 = df_final[[ 'TLA','Region', 'Market ', 'Facility','Node Name','Upgrade Type', 'Rank', '2020 Deferment']]
    df1.rename(columns = { 'Node Name' :'Parent Device ID'}, inplace = True)
    colo = df1.columns.get_loc('TLA')
    df1.insert(colo + 1,"Co-Lo",'')
    needby = df1.columns.get_loc('Facility')
    df1.insert(needby+1,"Material Need by Date",'')
    df1.insert(needby+2,"Activation Month",'')
    df1.insert(needby+3,"Migration Date",'')
    df1.insert(df1.columns.get_loc('Migration Date') + 1, 'Unique ID', df1['Facility'] +'-'+df1['Upgrade Type'])
    df1.insert(df1.columns.get_loc('Unique ID') + 1, "Program", 'CB Metro Core')
    df1.insert(df1.columns.get_loc('Program') + 1, "Activity Type", '')
    df1.insert(df1.columns.get_loc('Activity Type') + 1, "Scenario", '')
    df1.insert(df1.columns.get_loc('Scenario') + 1, "Quantity", 1)
    df1.insert(df1.columns.get_loc('Parent Device ID') + 1, "Equipment", '')
    df1.sort_values(by = ['Region', 'TLA'],inplace = True)
    df1.drop_duplicates( inplace = True)
    return(df1)

df1 = output()



def poi_dates():
    upgrades = [ 'PE ROUTER LICENSE UPGRADE',  'P ROUTER LICENSE UPGRADE',  'P ROUTER NEW 100G CARD', 'PE ROUTER NEW 100G CARD']
    rank1 = ['1A', '1B', '1C', '1D']
    rank2 = ['2A', '2B', '2C', '2D']
    rank3 = ['3A', '3B', '3C', '3D']
    df = df1.copy()
    df['Rank'] = df['Rank'].str.strip()
    routers = df.loc[df['Upgrade Type']=='ROUTER UPGRADE']
    r1 = routers['Unique ID'].loc[routers['Rank'].isin(rank1)].tolist()
    r2 = routers['Unique ID'].loc[routers['Rank'].isin(rank2)].tolist()
    r3 = routers['Unique ID'].loc[routers['Rank'].isin(rank3)].tolist()
    
    for i in r1:
        df.loc[df['Unique ID']==i, 'Material Need by Date'] = 'Q1'
        #df.loc[df['Unique ID']==i, 'Activation Month'] = '4/15/2021'
        
    for i in r2:
        df.loc[df['Unique ID']==i, 'Material Need by Date'] = 'Q2/Q3'
        #df.loc[df['Unique ID']==i, 'Activation Month'] = '7/15/2021'
        
    for i in r3:
        df.loc[df['Unique ID']==i, 'Material Need by Date'] = 'Q4'
        #df.loc[df['Unique ID']==i, 'Activation Month'] = '10/15/2021' 
    
       
    satbox = df.loc[(df['Upgrade Type'] == 'NEW SATELLITE BOX') |(df['Upgrade Type'] == 'SATELLITE BOX')]
    s1 = satbox['Unique ID'].loc[satbox['Rank'].isin(rank1)].tolist()
    s2 = satbox['Unique ID'].loc[satbox['Rank'].isin(rank2)].tolist()
    s3 = satbox['Unique ID'].loc[satbox['Rank'].isin(rank3)].tolist()
    
    for i in s1:
        df.loc[df['Unique ID']==i, 'Material Need by Date'] = 'Q1'
        #df.loc[df['Unique ID']==i, 'Activation Month'] = '3/15/2021'
        
    for i in s2:
        df.loc[df['Unique ID']==i, 'Material Need by Date'] = 'Q2/Q3'
        #df.loc[df['Unique ID']==i, 'Activation Month'] = '6/15/2021'
        
    for i in s3:
        df.loc[df['Unique ID']==i, 'Material Need by Date'] = 'Q4'
        #df.loc[df['Unique ID']==i, 'Activation Month'] = '9/15/2021' 
        
        
    licenses =  df.loc[(df['Upgrade Type'].isin(upgrades))]
    l1 = licenses['Unique ID'].loc[licenses['Rank'].isin(rank1)].tolist()
    l2 = licenses['Unique ID'].loc[licenses['Rank'].isin(rank2)].tolist()
    l3 = licenses['Unique ID'].loc[licenses['Rank'].isin(rank3)].tolist()
    
    for i in l1:
        df.loc[df['Unique ID']==i, 'Material Need by Date'] = 'Q1'
        #df.loc[df['Unique ID']==i, 'Activation Month'] = '2/15/2021'
        
    for i in l2:
        df.loc[df['Unique ID']==i, 'Material Need by Date'] = 'Q2/Q3'
        #df.loc[df['Unique ID']==i, 'Activation Month'] = '5/15/2021'
        
    for i in l3:
        df.loc[df['Unique ID']==i, 'Material Need by Date'] = 'Q4'
        #df.loc[df['Unique ID']==i, 'Activation Month'] = '8/15/2021'        
        
    
    return(df)

df1 = poi_dates() 



############################################################################################################################################################################

def belindas_list():
    b1 = df_belindas_list.drop_duplicates()
    
    df2 = df1.append(b1)       
    return(df2)

df1 = belindas_list()

############################################################################################################################################################################



def pricing():
            
    dfoutput = df1.copy()    
    dfoutput['Key'] = dfoutput['Parent Device ID'] + dfoutput['Upgrade Type']
    df_ports1 = df_ports[df_ports['Port Type']=='100 Gigabit Ethernet']
    thirty6 = df_ports1['Node Name'].loc[df_ports1['Total Port Count'] >= 48].drop_duplicates().tolist()
    ports_lic = df_ports[['Node Name','License Type']]
    df = df_pricing.merge(ports_lic, how = 'right', on = 'License Type')
    for t in thirty6:
        df.loc[(df['Node Name']==t) & (df['Upgrade Type']!= 'ROUTER UPGRADE') & (df['Upgrade Type']!= 'NEW SATELLITE BOX'), 'Equipment Price'] = 7395
        df.loc[(df['Node Name']==t) & (df['Upgrade Type']!= 'ROUTER UPGRADE') & (df['Upgrade Type']!= 'NEW SATELLITE BOX'), 'Tax'] = 550.93
        df.loc[(df['Node Name']==t) & (df['Upgrade Type']!= 'ROUTER UPGRADE') & (df['Upgrade Type']!= 'NEW SATELLITE BOX'), 'Shipping'] = 36.98

    df['Key'] = df['Node Name'] + df['Upgrade Type']
    df.drop(['Upgrade Type', 'License Type','Node Name'], axis = 1, inplace = True)
    df_prices = dfoutput.merge(df, on = 'Key', how = 'left' )
    df_prices.drop_duplicates(inplace = True)    
    df_prices.drop(['Key'],axis = 1, inplace = True)
    return(df_prices)

df1= pricing()



def total_cost():
    df_prices = df1.copy()
    df_prices.loc[:,('Total Cost')] = df_prices['Equipment Price'] +  df_prices['Tax'] +  df_prices['Shipping'] 
    df_prices2 = df_prices.append(df_prices.sum(numeric_only = True), ignore_index=True)
    df_prices2['Total Cost'] = df_prices2['Total Cost'].apply(lambda x: "${:,.2f}".format(x))  
    df_prices2['Equipment Price'] = df_prices2['Equipment Price'].apply(lambda x: "${:,.2f}".format(x))  
    df_prices2['Tax'] = df_prices2['Tax'].apply(lambda x: "${:,.2f}".format(x))  
    df_prices2['Shipping'] = df_prices2['Shipping'].apply(lambda x: "${:,.2f}".format(x))  
    return(df_prices2)

df1 = total_cost()

############################################################################################################################################################################

def format_export():
    df = dfx.copy()
    df.sort_values(by = 'TLA', inplace = True) 
    df = df.round({'End of 2021 Projected Ports available minus Forecasted 10G Augments':1,'End of 2021 Projected Ports available minus Total 100G Growth':1,'End of 2020 Projected Ports Available': 1,'End of 2021 Projected Ports Available': 1 ,'Total Projected Port Consumption in 2020 + 2021': 1 }) 
    df.dropna(how = 'all',subset = ['Final Port Count',
       'Slots Available','MDA Free Subslot Count'], inplace = True)
    df.drop(['Key','tla_key','pairs_key','market_key'], axis = 1, inplace = True)
    

    
    #core 10G & 100G
    port_types = ['100 Gigabit Ethernet','10-Gig Ethernet SF']
    df10 = df.loc[df['Port Type'].isin(port_types)]
    df10 = df10.loc[~df10['TLA'].isnull()]
    df10 = df10.loc[df10['TLA']!='N/A']
    core = ['S-PE', 'HYBRID', 'P ROUTER']
    df10 = df10.loc[df10['Role'].isin(core)]
    
    return(df, df10)

dfx, df10 = format_export()    
    

############################################################################################################################################################################
def get_var_value(filename="varstore.dat"):
    with open(filename, "a+") as f:
        f.seek(0)
        val = int(f.read() or 0) + 1
        f.seek(0)
        f.truncate()
        f.write(str(val))
        return val

runscript_counter = get_var_value()


############################################################################################################################################################################

   
writer = pd.ExcelWriter(output_path + '\Scenario_Output_v' + str(runscript_counter) + ".xlsx", engine='xlsxwriter')
df1.to_excel(writer, sheet_name='POI', index = False)
dfx.to_excel(writer, sheet_name='Data Export', index = False)
#df10.to_excel(writer, sheet_name='Export - Core Routers 100G&10G', index = False)
data_definitions.to_excel(writer, sheet_name='Data Definitions', index = False)
writer.save()





############################################################################################################################################################################
#Test Cases
dftest1 = df_ports[[ 'Key','Final Port Count',
       'Slots Available','MDA Free Subslot Count']]
dftest2 = dfx.copy()
dftest2['Key'] = dftest2['IP'] + dftest2['Port Type']
dftest2 = dftest2[['Key','Final Port Count',
       'Slots Available','MDA Free Subslot Count']]
dftest2.dropna(how = 'all', inplace = True)
df_test = dftest1.merge(dftest2, on = 'Key', how = 'outer', indicator = True)
df_test1 = df_test.loc[lambda x:x['_merge'] != 'both']

if not df_test1.empty:
    print('WARNING! OUTPUT DATA DOES NOT MATCH SOURCE DATA. PLEASE PERFORM QA TO ENSURE ACCURACY')
    
    
    
print(' ------------VERSION '+ str(runscript_counter)+' COMPLETE: CHECK FOLDER FOR OUTPUT FILE------------ ')



  
    
    
