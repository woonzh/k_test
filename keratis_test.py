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
from sklearn.decomposition import PCA
from sklearn import preprocessing
import scipy.stats

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
    df['month']=[str(x.year)+ '-' + str(x.month).zfill(2) + '('+str(x.month_name())+')' for x in df['Date of Visit']]
    df['week']=[str(x.year)+' week ' + str(x.week) for x in df['Date of Visit']]
    df['quarter']=[str(x.year)+ ' Q'+str(math.ceil(x.month/3)) for x in df['Date of Visit']]
    
    return df

def dataClean(df, fname):
    #string to date conversion
    df['Date of Visit']=[convertStringToTime(x) for x in df['Date of Visit']]
    #product extraction
    productDf=productBreakdown(df['Product_List'])
    df=pd.concat([df, productDf], axis=1)
    
    #date breakdown
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
    
def periodSplitNoAvg(df, colName, row, col):
    periodicDf=df[['Product_S', 'Product_B', 'Product_C', colName]].groupby([colName]).sum()
    linePlotly(periodicDf, ['Product_S', 'Product_B', 'Product_C'], row, col)
    
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
    cols=['Product_S', 'Product_B', 'Product_C']
    for col in cols:
        emDf[col]=emDf[col]/emDf['total_sold']
        emDf[col]=[round(x,2) for x in emDf[col]]
        
    emDf['first_day']=df[['Anonymized_Employee_ID','Date_of_Visit']].groupby(['Anonymized_Employee_ID']).Date_of_Visit.min()
    emDf['last_day']=df[['Anonymized_Employee_ID','Date_of_Visit']].groupby(['Anonymized_Employee_ID']).Date_of_Visit.max()
    tem = emDf['last_day']-emDf['first_day']
    emDf['length(days)'] = [(x.days)+1 for x in tem]
    
    lastday=max(df['Date_of_Visit'])
    emDf['inactive_period']=[(lastday-x).days for x in emDf['last_day']]
    emDf['days_with_sales']=df[['Anonymized_Employee_ID','Date_of_Visit']].groupby(['Anonymized_Employee_ID']).Date_of_Visit.nunique()
    emDf['days_with_sales(percen)']=emDf['days_with_sales']/emDf['length(days)']
    emDf['days_with_sales(percen)']=[round(x, 2) for x in emDf['days_with_sales(percen)']]
    emDf['sales_per_day']=emDf['total_sold']/emDf['days_with_sales']
    emDf['sales_per_day']=[round(x,2) for x in emDf['sales_per_day']]
    emDf['unique_customers']=df[['Anonymized_Employee_ID','Anonymized_Customer_ID']].groupby(['Anonymized_Employee_ID']).Anonymized_Customer_ID.nunique()
    emDf['unique_customers(percen)']=emDf['unique_customers']/df[['Anonymized_Employee_ID','Anonymized_Customer_ID']].groupby(['Anonymized_Employee_ID']).Anonymized_Customer_ID.count()
    
    inactive=emDf[emDf['inactive_period']>90]
    active=emDf[emDf['inactive_period']<90]
    return active
#    return emDf

def PCATransform(df):
    values=df.values
    pca = PCA(n_components=3)
    result=pca.fit_transform(values)
    
    min_max_scaler = preprocessing.StandardScaler()
    result = min_max_scaler.fit_transform(result)
    return result

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
    
def findDist(data):
    dist_names = ["norm", "exponweib", "weibull_max", "weibull_min", "beta"]
    dist_names = ["norm"]
    
    params = {}
    dist_results=[]
    
    for dist_name in dist_names:
        dist = getattr(scipy.stats, dist_name)
        param = dist.fit(data)
        params[dist_name] = param
        D, p = scipy.stats.kstest(data, dist_name, args=param)
        dist_results.append((dist_name, p))
    
    best_dist, best_p = (max(dist_results, key=lambda item: item[1]))
    
    return best_dist, params[best_dist]
    
def findAnomaly(labels, distance, threshold):
    df=pd.DataFrame()
    df['labels']=labels
    df['distance']=distance
    
    unique_labels=set(labels)
    
    stats=[]
    new_labels=[]
    
    for label in unique_labels:
        subDf=df[df['labels']==label]

        best_dist, params = findDist(subDf['distance'])
        stats.append([best_dist, params])
        
        plt.hist(distance, bins=25, density=True, alpha=0.6)
        title=str(label)+' ' +best_dist
        plt.title(title)
        plt.show()
        
    for count in range(len(df)):
        label=df.iloc[count, 0]
        dist=df.iloc[count,1]
        distribution=getattr(scipy.stats, stats[label][0])
        prob=distribution(stats[label][1][0], stats[label][1][1]).pdf(dist)
        if prob<threshold:
            new_labels.append(len(unique_labels))
        else:
            new_labels.append(label)
    
    return new_labels
    
    
def kmeans(df, clusters, axislabels, threshold=0.1):    
    df.reset_index(drop=True)
    km=KMeans(n_clusters=clusters)
    distance=np.min(km.fit_transform(df), axis=1)
    labels=km.predict(df)
    
    labels=findAnomaly(labels, distance, threshold)
    
    fig = plt.figure(1, figsize=(7,7))
    ax = Axes3D(fig, rect=[0, 0, 0.95, 1], elev=48, azim=134)
    ax.scatter(df.iloc[:,0], df.iloc[:,1], df.iloc[:,2],
              c=labels, edgecolor="k")
    
    ax.set_xlabel(axislabels[0])
    ax.set_ylabel(axislabels[1])
    ax.set_zlabel(axislabels[2])
    plt.title("K Means", fontsize=14);
    
    return labels
    
storer=storage()
fname='data/Customer_Visits_Interview_Exercise_Data.csv'
cleanFname='data/Customer_Visits_Interview_Exercise_Data(cleaned).csv'
products=['Product_S' ,'Product_B','Product_C']
filedir='logs'
metadata = 'metadata.txt'

fig = tools.make_subplots(rows=3,
                      cols=2,
                      print_grid=True,
                      vertical_spacing=0.2,
                      horizontal_spacing=0.085,
                      subplot_titles=('products sold','products sold (cleaned)', 'average product S sold daily per employee (aggregated quarterly)', 'average product S sold daily per employee (aggregated monthly)', 'average product S sold daily per employee (aggregated weekly)', 'average product S sold daily per employee (aggregated daily)'))

##-------data cleaning and preperation-------
df=pd.read_csv(fname)
#storer.store('store/raw_data.p', df)
#cleanedDf=dataClean(df,cleanFname)
#storer.store('store/raw_data_cleaned.p', cleanedDf)

##-----importing cleaned and transformed data------
cleanedDf=pd.read_csv(cleanFname)
cleanedDf.columns=['Date_of_Visit', 'Product_List', 'Anonymized_Employee_ID', 'Anonymized_Customer_ID', 'Product_S', 'Product_B', 'Product_C', 'month', 'week', 'quarter']
cleanedDf['Date_of_Visit'] =  pd.to_datetime(cleanedDf['Date_of_Visit'], format='%Y-%m-%d')
cleanedDf['total_sold'] =cleanedDf['Product_S'] + cleanedDf['Product_B'] + cleanedDf['Product_C']


#-----daily plot for all 3 products-------
#q1title=('products sold','products sold (cleaned)', 'products sold by week', 'products sold by month')
#fig = tools.make_subplots(rows=2,
#                      cols=2,
#                      print_grid=True,
#                      vertical_spacing=0.2,
#                      horizontal_spacing=0.085,
#                      subplot_titles=q1title)
#dayCompiled=cleanedDf[['Date_of_Visit', 'Product_S', 'Product_B', 'Product_C']].groupby(['Date_of_Visit'], axis=0).sum()
#linePlotly(dayCompiled, products, 1,1)
#
##remove anomalies (weekends etc)
#percen=0.4
#dayCompiledCleaned=dayCompiled[(dayCompiled['Product_S']>percen*dayCompiled['Product_S'].mean()) &
#                               (dayCompiled['Product_B']>percen*dayCompiled['Product_B'].mean()) &
#                               (dayCompiled['Product_C']>percen*dayCompiled['Product_C'].mean())]
#linePlotly(dayCompiledCleaned, products, 1,2)
#periodSplitNoAvg(cleanedDf, 'week', 2,1)
#periodSplitNoAvg(cleanedDf, 'month', 2,2)
#
#plotHtml('graphs/q1_visualisations.html')
#
## could improve with more accurate unique employee count. Based on assumptions on inactive period to revise number of unique employees
## could plot graph of employee growth to check for time series relationship
## remove anomalies such as weekends
#q2title=('average product S sold daily per employee (aggregated quarterly)', 'average product S sold daily per employee (aggregated monthly)', 'average product S sold daily per employee (aggregated weekly)', 'average product S sold daily per employee (aggregated daily)')
#fig = tools.make_subplots(rows=2,
#                      cols=2,
#                      print_grid=True,
#                      vertical_spacing=0.2,
#                      horizontal_spacing=0.085,
#                      subplot_titles=q2title)
#
#product_s_quarter=periodSplit(cleanedDf, 'quarter', 1,1)
#product_s_month=periodSplit(cleanedDf, 'month', 1,2)
#product_s_week=periodSplit(cleanedDf, 'week', 2,1)
#product_s_day=periodSplit(cleanedDf, 'Date_of_Visit', 2,2)
#
#plotHtml('graphs/q2_visualisations.html')
#data=storer.storeLogs

#------machine learning for anomaly detection--------
#active, inactive=employeeSplit(cleanedDf)
emDf = employeeSplit(cleanedDf)

subDf=emDf[['Product_S', 'Product_B', 'Product_C', 'sales_per_day', 'days_with_sales(percen)', 'unique_customers(percen)', 'length(days)']]
#subDf=emDf[['total_sold', 'days', 'length(days)']]

pcaDf=PCATransform(subDf)

elbow_curve(pcaDf)
clusters=2
if len(list(subDf))>3:
    axislabels=[1,2,3]
else:
    axislabels=list(subDf)

threshold=0.1
labels=kmeans(pd.DataFrame(pcaDf), clusters, axislabels, threshold)
emDf['label']=labels
metadataDf=emDf[['label','Product_S', 'Product_B', 'Product_C', 'total_sold', 'length(days)', 'days_with_sales(percen)', 'sales_per_day', 'unique_customers', 'unique_customers(percen)']]

data=pcaDf
metadataDf.to_csv(filedir+'/'+ metadata, sep='\t')

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

customer_purchase=cleanedDf[['Date_of_Visit','Anonymized_Customer_ID','Anonymized_Employee_ID']].groupby(['Date_of_Visit','Anonymized_Customer_ID']).count()
customer_purchase['unique_employees']=cleanedDf[['Date_of_Visit','Anonymized_Customer_ID','Anonymized_Employee_ID']].groupby(['Date_of_Visit','Anonymized_Customer_ID']).Anonymized_Employee_ID.nunique()
tem=cleanedDf[['Date_of_Visit','Anonymized_Customer_ID','Product_S', 'Product_B', 'Product_C']].groupby(['Date_of_Visit','Anonymized_Customer_ID']).sum()
customer_purchase=pd.concat([customer_purchase,tem], axis=1, sort=False)