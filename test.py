import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from plotly.offline import plot
import plotly.graph_objs as go
from plotly import tools
from util import storage
import math
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

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

def periodLabel(df):
    df['month']=[str(x.year)+ '-' + str(x.month_name()) for x in df['Date of Visit']]
    df['week']=[str(x.year)+' week ' + str(x.week) for x in df['Date of Visit']]
    df['quarter']=[str(x.year)+ ' Q'+str(math.ceil(x.month/3)) for x in df['Date of Visit']]
    
    return df

def dataClean(df, fname):
    df['Date of Visit']=[convertStringToTime(x) for x in df['Date of Visit']]
    productDf=productBreakdown(df['Product_List'])
    df=pd.concat([df, productDf], axis=1)
    
    df=periodLabel(df)
    
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
    
def linePlotly(df, columnNames, row, col):
    global fig
#    data=[]
    for name in columnNames:
        trace = go.Scatter(x=df.index, y=df[name], name=name)
        fig.append_trace(trace, row, col)
    
#    plot(data, filename=fname, image='jpeg')

def plotHtml(fname):
    global fig
    layout=fig['layout']
    for opt in layout:
        if 'xaxis' in opt:
            layout[opt].update(title='date/period')
        if 'yaxis' in opt:
            layout[opt].update(title='products sold')
            
    plot(fig, filename=fname, image='jpeg')
    
def periodSplit(df, colName, row, col):
    periodicDf=df[['Product_S', colName]].groupby([colName]).sum()
    periodicDf['unqiue_employees']=list(df[['Anonymized_Employee_ID', colName]].groupby([colName]).Anonymized_Employee_ID.nunique())
    if colName == 'Date_of_Visit':
        periodicDf['days']=[1]*len(periodicDf)
    else:
        periodicDf['days']=list(df[['Date_of_Visit', colName]].groupby([colName]).Date_of_Visit.nunique())
    periodicDf['daily_avg_per_employee']=[round(x,1) for x in (periodicDf['Product_S']/periodicDf['unqiue_employees']/periodicDf['days'])]
    
    if fname!='':
        linePlotly(periodicDf, ['daily_avg_per_employee'], row, col)
    
    return periodicDf

def employeeSplit(df):
    emDf=df[['Anonymized_Employee_ID','Product_S', 'Product_B', 'Product_C', 'total_sold']].groupby(['Anonymized_Employee_ID']).sum()
    emDf['days']=df[['Anonymized_Employee_ID','Date_of_Visit']].groupby(['Anonymized_Employee_ID']).count()
    emDf['unique_customers']=df[['Anonymized_Employee_ID','Anonymized_Customer_ID']].groupby(['Anonymized_Employee_ID']).Anonymized_Customer_ID.nunique()
    return emDf

def elbow_curve(df):
    n_cluster = range(1, 20)
    kmeans = [KMeans(n_clusters=i).fit(df) for i in n_cluster]
    scores = [kmeans[i].score(df) for i in range(len(kmeans))]
    
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(n_cluster, scores)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Score')
    plt.title('Elbow Curve')
    plt.show()
    
def kmeans(df, clusters):
    df.reset_index(drop=True)
    km=KMeans(n_clusters=clusters)
    km.fit(df)
    km.predict(df)
    
    labels=km.labels_
    
    fig = plt.figure(1, figsize=(7,7))
    ax = Axes3D(fig, rect=[0, 0, 0.95, 1], elev=48, azim=134)
    ax.scatter(df.iloc[:,0], df.iloc[:,1], df.iloc[:,2],
              c=labels.astype(np.float), edgecolor="k")
    ax.set_xlabel("total sold")
    ax.set_ylabel("days")
    ax.set_zlabel("unique customers")
    plt.title("K Means", fontsize=14);
    
    return labels
    
storer=storage()
fname='Customer_Visits_Interview_Exercise_Data.csv'
cleanFname='Customer_Visits_Interview_Exercise_Data(cleaned).csv'
products=['Product_S' ,'Product_B','Product_C']
filedir='logs'
metadata = 'metadata.txt'

fig = tools.make_subplots(rows=3,
                      cols=2,
                      print_grid=True,
                      vertical_spacing=0.2,
                      horizontal_spacing=0.085,
                      subplot_titles=('products sold','products sold (cleaned)', 'average product S sold daily per employee (aggregated quarterly)', 'average product S sold daily per employee (aggregated monthly)', 'average product S sold daily per employee (aggregated weekly)', 'average product S sold daily per employee (aggregated daily)'))

df=pd.read_csv(fname)
#storer.store('raw_data.p', df)
#cleanedDf=dataClean(df,cleanFname)
#storer.store('raw_data_cleaned.p', cleanedDf)

cleanedDf=pd.read_csv(cleanFname)
cleanedDf.columns=['Date_of_Visit', 'Product_List', 'Anonymized_Employee_ID', 'Anonymized_Customer_ID', 'Product_S', 'Product_B', 'Product_C', 'month', 'week', 'quarter']
cleanedDf['total_sold'] =cleanedDf['Product_S'] + cleanedDf['Product_B'] + cleanedDf['Product_C']

emDf = employeeSplit(cleanedDf)

#elbow_curve(emDf[['total_sold', 'days', 'unique_customers']])
clusters=5
#labels=kmeans(emDf[['total_sold', 'days', 'unique_customers']], clusters)

subDf=emDf[['total_sold', 'days', 'unique_customers']]
data=subDf.values
subDf.to_csv(filedir+'/'+ metadata, sep='\t')

tf_data = tf.Variable(data)

with tf.Session() as sess:
    saver = tf.train.Saver([tf_data])
    sess.run(tf_data.initializer)
    saver.save(sess, filedir+'/tf_data.ckpt')
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = tf_data.name
    embedding.metadata_path = metadata
    projector.visualize_embeddings(tf.summary.FileWriter(filedir), config)


##-----daily plot for all 3 products-------
#dayCompiled=cleanedDf[['Date_of_Visit', 'Product_S', 'Product_B', 'Product_C']].groupby(['Date_of_Visit'], axis=0).sum()
#linePlotly(dayCompiled, products, 1,1)
#
#percen=0.4
#dayCompiledCleaned=dayCompiled[(dayCompiled['Product_S']>percen*dayCompiled['Product_S'].mean()) &
#                               (dayCompiled['Product_B']>percen*dayCompiled['Product_B'].mean()) &
#                               (dayCompiled['Product_C']>percen*dayCompiled['Product_C'].mean())]
#linePlotly(dayCompiledCleaned, products, 1,2)
#
#product_s_quarter=periodSplit(cleanedDf, 'quarter', 2,1)
#product_s_month=periodSplit(cleanedDf, 'month', 2,2)
#product_s_week=periodSplit(cleanedDf, 'week', 3,1)
#product_s_day=periodSplit(cleanedDf, 'Date_of_Visit', 3,2)
#
#plotHtml('test.html')
#data=storer.storeLogs

