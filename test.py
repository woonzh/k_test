import pandas as pd
from datetime import datetime
from dateutil import parser
import matplotlib.pyplot as plt
import plotly.plotly as py
from plotly.offline import plot
import plotly.graph_objs as go

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

def linePlot(df, columnNames, fname=''):
    linewidth=2
    for name in columnNames:
        plt.plot(df.index, df[name], linewidth=linewidth, label=name)
    if fname!='':
        plt.savefig(fname)
    
    plt.legend()
    plt.show()
    
def linePlotly(df, columnNames, fname):
    data=[]
    for name in columnNames:
        trace = go.Scatter(x=df.index, y=df[name], name=name)
        data.append(trace)
    
    plot(data, filename=fname, image='jpeg')
    

fname='Customer_Visits_Interview_Exercise_Data.csv'
cleanFname='Customer_Visits_Interview_Exercise_Data(cleaned).csv'
products=['Product_S' ,'Product_B','Product_C']

#df=pd.read_csv(fname)
##cleanedDf=dataClean(df,cleanFname)
#
#cleanedDf=pd.read_csv(cleanFname)

##-----daily plot for all 3 products-------
#dayCompiled=cleanedDf[['Date of Visit', 'Product_S', 'Product_B', 'Product_C']].groupby(['Date of Visit'], axis=0).sum()
#linePlotly(dayCompiled, products, 'all_products_time_series.html')
#
#percen=0.4
#dayCompiledCleaned=dayCompiled[(dayCompiled['Product_S']>percen*dayCompiled['Product_S'].mean()) & \
#                               (dayCompiled['Product_B']>percen*dayCompiled['Product_B'].mean()) & \
#                               (dayCompiled['Product_C']>percen*dayCompiled['Product_C'].mean())]
#linePlotly(dayCompiledCleaned, products, 'all_products_time_series(cleaned).html')

##------plot for product S----------
