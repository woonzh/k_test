import pandas as pd
from datetime import datetime
from dateutil import parser

def convertStringToTime(string):
    obj=datetime.strptime(string, '%d/%m/%Y')
    return obj

def productStringSplit(string):
    products=['Product_S' ,'Product_B','Product_C']
    store=string.split('  ')
    
    results=[1 if x in store else 0 for x in products]
    return results

def productBreakdown(stringLst):
    temDf=pd.DataFrame(columns=['Product_S' ,'Product_B','Product_C'])
    for count, val in enumerate(list(stringLst)):
        if (count % 10000) == 0:
            print(count)
        temDf.loc[count]=productStringSplit(val)
        
    return temDf

def dataClean(df, fname):
    df['Date of Visit']=[convertStringToTime(x) for x in df['Date of Visit']]
    productDf=productBreakdown(df['Product_List'])
    df=pd.concat([df, productDf], axis=1)
    
    df.to_csv(fname, index=False)
    
    return df

fname='Customer_Visits_Interview_Exercise_Data.csv'
cleanFname='Customer_Visits_Interview_Exercise_Data(cleaned).csv'

df=pd.read_csv(fname)
cleanedDf=dataClean(df,cleanFname)

cleanedDf=pd.read_csv(cleanFname)