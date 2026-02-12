import streamlit as st

st.title("Retail Data Analysis")

with st.echo():
    import pandas as pd
    import numpy as np
    import datetime as dt
    from rfm import RFM
    import math
    from pandas import DataFrame
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from sklearn.cluster import KMeans
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import silhouette_score, silhouette_samples
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import MinMaxScaler
    from scipy.cluster.hierarchy import dendrogram, linkage

    data = pd.read_csv("retail.csv")

    # Given a customer ID, return a dataframe of all purchases made by that customer

    # Assumes clean data

    # Given a customer ID, return a dataframe of all purchases made by that customer

    # Assumes clean data

    def findAllPurchases(dataFrame):
        allPurchases = {} # id: {stockCode: amountPurchased}

        for index, row in dataFrame.iterrows():
            id = int(row['CustomerID'])
            stockCode = row['StockCode']
            quantity = float(row['Quantity'])

            m = None

            if (id != None) and (stockCode != None) and (not math.isnan(id)):
                l = allPurchases.get(id)

                if l != None:
                    m = l.get(stockCode)
                else:
                    # New ID
                    amount = quantity
                    l = {stockCode: amount}
                
                if m == None:
                    # New item
                    amount = quantity
                    l.update({stockCode: amount})
                else:
                    amount = quantity + m
                    l.update({stockCode: amount})

                allPurchases.update({id:l})

        return allPurchases

    def findMostPurchasedItem(dataFrame):
        stockCodes = dataFrame['StockCode'].tolist()
        amounts = dataFrame['AmountPurchased'].tolist()

        max = 0
        maxIndex = 0

        for i in range(len(amounts)):
            if amounts[i] > max:
                max = amounts[i]
                maxIndex = i

        return stockCodes[maxIndex]



st.markdown("""## Assumptions

- All UnitPrices are positive
- All Quantities are positive
- All sales have a 5% profit margin
""")

st.write(data)

# Clean data

st.markdown("## Cleaning Data")

with st.echo():
    cleanData = data.copy()

    sortedClean = cleanData.sort_values(by=['UnitPrice'], inplace=False, ascending=True)
    sortedClean = sortedClean.drop([0, 1])

    cleanData = sortedClean.copy()

    sortedClean = cleanData.sort_values(by=['Quantity'], inplace=False, ascending=True)

    cleanData.reset_index()

    sortedClean = sortedClean.drop(sortedClean.index[:10624])

    cleanData = sortedClean.copy()
    cleanData = cleanData.dropna()

    data = cleanData.copy()
    data.reset_index()

st.write(data)

st.markdown("## Converting Invoice Date to readable string")

with st.echo():
    data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])

st.write(data)

st.markdown("## Calculating Total Monetary Value")

with st.echo():
    data['Total Amount'] = data['Quantity'] * data['UnitPrice']
st.write(data)

st.markdown("## Creating the values")

with st.echo():
    # Use the latest date in the dataset as the snapshot date
    snapshot_date = data['InvoiceDate'].max() + pd.DateOffset(days=1)  

    rfm_data = data.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,  # Recency
        'InvoiceNo': 'count',  # Frequency
        'UnitPrice': 'sum'  # Monetary
    }).reset_index()

    rfm_data.rename(columns={
        'InvoiceDate': 'Recency',
        'InvoiceNo': 'Frequency',
        'UnitPrice': 'Monetary'}, inplace=True)

st.markdown("## Dividing customers into segments based on quartile values")

with st.echo():
    r_label = range(4,0,-1)
    f_label = range(1,5)
    m_label = range(1,5)

    rfm_data['R'] = pd.qcut(rfm_data['Recency'],  q=4, labels = r_label)
    rfm_data['F'] = pd.qcut(rfm_data['Frequency'],  q=4, labels = f_label)
    rfm_data['M'] = pd.qcut(rfm_data['Monetary'],  q=4, labels = m_label)

    rfm_data['RFM'] = rfm_data['R'].astype(str) + rfm_data['F'].astype(str) + rfm_data['M'].astype(str)

st.write(rfm_data)

with st.echo():
    data.info()

st.markdown("## For each customer calculate total revenue")

with st.echo():
    revenuedata = data[['Quantity', 'CustomerID', 'UnitPrice', 'InvoiceNo']]

    revenue = {} # ID : [Total Revenue]
    orders = {} # ID: [InvoiceNums]

    for index, row in data.iterrows():
        id = row['CustomerID']

        if (not math.isnan(id)):
            id = int(id)

            oldRev = revenue.get(id)
            invoices = orders.get(id)

            if oldRev == None:
                revVal = 0
            else:
                revVal = float(oldRev[0])

            if revVal >= 0 and float(row['Quantity']) >= 0:
                if invoices == None:
                    invoices = []

                newRev = revVal + (float(row['Quantity']) * float(row['UnitPrice']))

                rev = []
                rev.append(newRev)
                rev.append(id)

                revenue.update({id: rev})

                if row['InvoiceNo'] not in invoices:
                    invoices.append(row['InvoiceNo'])
                    orders.update({id:invoices})

st.write(revenuedata)

with st.echo():
    clvData = pd.DataFrame.from_dict(revenue, orient = 'index')

    clvData.rename(columns={0:"Total Revenue", 1:"CustomerID"}, inplace=True)

st.write(clvData)

with st.echo():
    numOrders = {}
    totalInvoices = 0

    for id in orders:
        customerTotalOrders = len(orders.get(id))
        totalInvoices += customerTotalOrders
        numOrders.update({id:customerTotalOrders})

    clvData.sort_values(by=['CustomerID'], inplace=True)

    clvData2 = pd.DataFrame.from_dict(numOrders, orient = 'index')

st.write(clvData2)

with st.echo():
    clvData.insert(2, "Total Orders", clvData2)

st.write(clvData)

with st.echo():
    clvData["Average Order Value"] = clvData['Total Revenue'] / clvData['Total Orders']

st.write(clvData)

with st.echo():
    clvData['Purchase Frequency'] = clvData['Total Orders'] / totalInvoices

st.write(clvData)

with st.echo():
    clvData['Customer Value'] = clvData['Average Order Value'] * clvData['Purchase Frequency']

st.write(clvData)

with st.echo():
    sortedData = clvData.sort_values(by=['Customer Value'], inplace=False)

st.write(sortedData)

with st.echo():
    customersWhoOrderedOnce = 0
    totalCustomers = 0

    for index, row in clvData.iterrows():
        if row['Total Orders'] == 1:
            customersWhoOrderedOnce += 1

        totalCustomers += 1

    churnRate = customersWhoOrderedOnce / totalCustomers

st.write("The churn rate is " + str(churnRate))

with st.echo():
    clvData['Total Profit'] = clvData['Total Revenue'] * 0.05

st.write(clvData)

with st.echo():
    clvData['Churn Value'] = clvData['Customer Value'] / churnRate

st.write(clvData)

with st.echo():
    clvData['Customer Lifetime Value'] = clvData['Total Profit'] * clvData['Churn Value']
st.write(clvData)

st.markdown("""The CLTV we obtained here is calculated in a different way than RFM.

We will calculate a different CLTV using the following formula below:

(Recency score x Recency weight) + (Frequency score x Frequency weight) + (Monetary score x Monetary weight)
""")

with st.echo():
    recency_weight = 0.3
    frequency_weight = 0.3
    monetary_weight = 0.3

    rfm_data['R'] = rfm_data['R'].astype(float)
    rfm_data['F'] = rfm_data['F'].astype(float)
    rfm_data['M'] = rfm_data['M'].astype(float)

    rfm_data = rfm_data.dropna()
    rfm_data = rfm_data[(rfm_data['F']>0) & (rfm_data['M']>0)]

    rfm_data['CLV'] = (rfm_data['R'] * recency_weight) + (rfm_data['F'] * frequency_weight) + (rfm_data['M'] * monetary_weight)

st.write(rfm_data)

st.markdown("""## Segmentation

Next we want to segment customers into different categories based on RFM and CLV.
The rfm library already does some of this for us
""")

st.markdown('## Create Revenue Column (per item)')

with st.echo():
    data['Revenue'] = data['Quantity'] * data['UnitPrice']

st.write(data)

with st.echo():
    r = RFM(data, customer_id='CustomerID', transaction_date='InvoiceDate', amount='Revenue')

st.write(r.rfm_table)

st.write('Below are the possible segments')
st.write(r.rfm_table['segment'])

st.markdown("""## 3D Visualization to find clusters""")

with st.echo():
    fig = plt.figure(figsize = (5,5))
    threedee = fig.add_subplot(projection='3d')

    threedee.scatter(rfm_data['R'], rfm_data['F'], rfm_data['M'])
    threedee.set_xlabel('Recency')
    threedee.set_ylabel('Frequency')
    threedee.set_zlabel('Monetary Value')

st.pyplot(fig)

st.markdown("""This isn't that helpful, everything is overlapping. Lets try a different graph""")

st.markdown("## Plot RFM score against CLTV")

with st.echo():
    x = rfm_data.sort_values(by=['CustomerID'], inplace=False)['CLV']
    y = clvData.sort_values(by=['CustomerID'], inplace=False)['Customer Lifetime Value']

    ax = plt.axes()

    ax.set_title("RFM Score vs CLTV")
    ax.set_ylabel('CLTV')
    ax.set_ylim(0, 600000)
    ax.set_xlabel('RFM Score')
    ax.set_xlim(0, 4.5)

    plt.scatter(x, y)

st.pyplot(ax.figure)

st.markdown("""This graph is a bit odd, there aren't any customers with an FRM score of 4.5 (the max).""")

with st.echo():
    zip = list(zip(x, y))
    inertias = []
    fig = plt.figure()

    for i in range(1,11):
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(zip)
        inertias.append(kmeans.inertia_)

    plt.plot(range(1,11), inertias, marker='o')
    plt.title('Elbow method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')

st.pyplot(fig)

st.markdown("""We can see here that 2 is a good value for k""")

with st.echo():
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(zip)

    fig = plt.scatter(x, y, c=kmeans.labels_)

st.pyplot(fig.figure)

st.markdown("""Once again we don't really learn a lot from this. Lets drop outliers than do this again.""")

with st.echo():
    x = rfm_data.sort_values(by=['CustomerID'], inplace=False, ascending=True)[['CustomerID', 'CLV']]
    y = clvData.sort_values(by=['CustomerID'], inplace=False, ascending=True)[['CustomerID', 'Customer Lifetime Value']]

    y.reset_index(inplace=True)

    x.insert(2, "CLTV", y['Customer Lifetime Value'].squeeze())

    Q1 = x.quantile(0.25)             #To find the 25th percentile and 75th percentile.
    Q3 = x.quantile(0.75)

    IQR = Q3 - Q1                           #Inter Quantile Range (75th perentile - 25th percentile)

    lower=Q1-1.5*IQR                        #Finding lower and upper bounds for all values. All values outside these bounds are outliers
    upper=Q3+1.5*IQR

st.write(x)

with st.echo(): 
    keep = ((x.select_dtypes(include=['float64','int64'])<lower) | (x.select_dtypes(include=['float64','int64'])>upper))

st.write(keep)

with st.echo():
    for index, row in keep.iterrows():
        if row['CLV'] or row['CLTV']:
            x.drop(index=index, inplace=True)

st.write(x)

st.markdown("""We have dropped the outliers from CLV (RFM calculated) and CLTV, so now lets graph again""")

st.markdown("## Plot RFM score against CLTV")

with st.echo():
    ax = plt.axes()

    ax.set_title("RFM Score vs CLTV")
    ax.set_ylabel('CLTV')
    ax.set_ylim(0, 60)
    ax.set_xlabel('RFM Score')
    ax.set_xlim(0, 4.5)

    plt.scatter(x['CLV'], x['CLTV'])

st.pyplot(ax.figure)

st.markdown("""We can learn a lot more from this than the other plot""")

with st.echo():
    del zip
    zipNew = list(zip(x['CLV'], x['CLTV']))
    inertias = []

    for i in range(1,11):
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(zipNew)
        inertias.append(kmeans.inertia_)

    plt.plot(range(1,11), inertias, marker='o')
    plt.title('Elbow method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')

st.pyplot(plt.figure())

st.markdown("""Once again, we see 2 is the optimal number of clusters using the elbow method""")

with st.echo():
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(zipNew)

    fig = plt.scatter(x['CLV'], x['CLTV'], c=kmeans.labels_)

st.pyplot(fig.figure)

st.markdown("""Lets also look at 3 clusters too""")

with st.echo():
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(zipNew)

    fig = plt.scatter(x['CLV'], x['CLTV'], c=kmeans.labels_)

st.pyplot(fig.figure)

st.markdown("""And now, just to see what it will look like, lets try 4""")

with st.echo():
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(zipNew)

    fig = plt.scatter(x['CLV'], x['CLTV'], c=kmeans.labels_)

st.pyplot(fig.figure)

st.markdown("""The k-means clustering shows that CLTV divides the groups more than CLV calculated from RFM score.

This doesn't really mean anything useful though, other than that we don't need to be using both of these values.

From this point forward, lets use the CLV calculated from RFM since that was what was instructed.

Now lets plot total orders vs CLV
""")

with st.echo():
    x.reset_index(inplace=True)
    clvData.reset_index(inplace=True)
    x.drop('CLTV', axis=1, inplace=True)

    totalOrders = clvData['Total Orders']

    x.insert(3, 'Total Orders', totalOrders)

    totalOrders = pd.DataFrame(clvData['Total Orders'])

    x.drop(['level_0', 'index', 'CustomerID'], axis=1, inplace=True, errors='ignore')

    Q1 = x.quantile(0.25)             #To find the 25th percentile and 75th percentile.
    Q3 = x.quantile(0.75)

    IQR = Q3 - Q1                           #Inter Quantile Range (75th perentile - 25th percentile)

    lower=Q1-1.5*IQR                        #Finding lower and upper bounds for all values. All values outside these bounds are outliers
    upper=Q3+1.5*IQR

st.write(x)

with st.echo():
    left, right = x.align(totalOrders, join="outer", axis=1)

    keep = ((x.select_dtypes(include=['float64','int64'])<lower) | (x.select_dtypes(include=['float64','int64'])>upper))

    for index, row in keep.iterrows():
        if row['CLV'] or row['Total Orders']:
            x.drop(index=index, inplace=True)

st.write(keep)

st.markdown("## Plot Total Orders vs CLV")

with st.echo():
    ax = plt.axes()

    ax.set_title("Total Orders vs CLV")
    ax.set_ylabel('CLV')
    ax.set_ylim(0, 5)
    ax.set_xlabel('Total Orders')
    ax.set_xlim(0, 4.5)

    plt.scatter(x['Total Orders'], x['CLV'])

st.pyplot(ax.figure)

with st.echo():
    zipNew = list(zip(x['Total Orders'], x['CLV']))
    inertias = []

    for i in range(1,11):
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(zipNew)
        inertias.append(kmeans.inertia_)

    fig = plt.plot(range(1,11), inertias, marker='o')
    plt.title('Elbow method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')

st.pyplot(plt.figure())

with st.echo():
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(zipNew)

    fig = plt.scatter(x['Total Orders'], x['CLV'], c=kmeans.labels_)

st.pyplot(plt.figure())

with st.echo():
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(zipNew)

    fig = plt.scatter(x['Total Orders'], x['CLV'], c=kmeans.labels_)

st.pyplot(plt.figure())

st.markdown("""Not much correlation between total number or orders vs CLV.

Lets try looking at the RFM library's segmentation
""")

st.markdown("## Plot Total Orders vs CLV")

with st.echo():
    x = rfm_data.sort_values(by=['CustomerID'], inplace=False, ascending=True)['CLV']

    ax = plt.axes()

    ax.set_title("Segment vs CLV")
    ax.set_ylabel('CLV')
    ax.set_xlabel('Segment')

    plt.scatter(x, r.rfm_table['segment'])

st.pyplot(ax.figure)

st.markdown("""Once again the data is not varied enough to give good clusters. We need to pick better axes.""")

st.markdown("## Drop non-numeric columns")

with st.echo():
    data_numeric = data.select_dtypes(include=[np.number])

st.write(data_numeric)

st.markdown("## Handle missing values by imputing with the mean")

with st.echo():
    imputer = SimpleImputer(strategy='mean')
    data_imputed = pd.DataFrame(imputer.fit_transform(data_numeric), columns=data_numeric.columns)

st.write(data_imputed)

st.markdown("## Scale the data")

with st.echo():
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_imputed)

st.write(data_scaled)

st.markdown("## Initialize an empty list to store the within-cluster sum of squares (WCSS) values")

with st.echo():
    wcss = []

    # Set the range of the number of clusters to try
    min_clusters = 1
    max_clusters = 10

    # Perform K-means clustering for each value of k
    for k in range(min_clusters, max_clusters + 1):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(data_scaled)
        wcss.append(kmeans.inertia_)

with st.echo():
    # Plot the WCSS values against the number of clusters
    fig = plt.plot(range(min_clusters, max_clusters + 1), wcss)
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.title('Elbow Method')

st.pyplot(plt.figure())

st.markdown("## Customer Segmentation via RFM & K-Means")

with st.echo():
    # Scaling the Features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(rfm_data[['R', 'F', 'M']])

    # Applying K-means clustering
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(rfm_data)

    # Get cluster labels and add them to the RFM dataframe
    rfm_data['Cluster'] = kmeans.labels_

    cluster_sizes = rfm_data['Cluster'].value_counts()
    cluster_composition = rfm_data.groupby('Cluster').mean()

st.write(cluster_sizes)

st.write(cluster_composition)

st.markdown("## Visualize the Clusters")

with st.echo():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(rfm_data['R'], rfm_data['F'], rfm_data['M'], c=rfm_data['Cluster'], cmap='viridis')
    ax.set_xlabel('R')
    ax.set_ylabel('F')
    ax.set_zlabel('M')

st.pyplot(plt.figure())

st.markdown("""## Analyzing the Clusters

Looking at the cluster sizes, it is clear to see that there is relatively the same amout of plot points in each cluster, aside from the last one where there is only 4. The last one is very imbalanced when compared to the other three.
""")

with st.echo(): 
    silhouette_avg = silhouette_score(rfm_data, rfm_data['Cluster'])
st.write("The average Silhouette Coefficient is:", silhouette_avg)

st.markdown("""A silhouette coefficient of 0.5716531164367413 is a reasonably good value. It tells us that the KMeans clustering method did a decent job of grouping the variables into clusters and the clusters are a little separate from each other.""")

st.markdown("## Customer Segmentation via Geographical Location")

with st.echo():
    segmented_data = data.groupby('Country').size().reset_index(name='CustomerID')
    segmented_data.rename(columns={"Country":"Country", "CustomerID":"Customers"}, inplace=True)
    segmented_data.sort_values(by=['Customers'], inplace=True, ascending=False)
    segmented_data.reset_index(inplace=True)
    segmented_data.drop('index', axis=1, inplace=True)

st.write(segmented_data)

st.write("These results don't really tell us much except that most of the consumers are from the UK and the least amount of consumers are from Saudi Arabia. The 244 that are unspecified are the rows with missing Country data.")

st.markdown("## Customer segmentation based on CLV")

with st.echo():
    df = rfm_data.copy()

    # Normalize CLV values
    scaler = MinMaxScaler()
    df['Normalized_CLV'] = scaler.fit_transform(df['CLV'].values.reshape(-1, 1))

    # Perform customer segmentation using K-means clustering
    X = df[['Normalized_CLV']]
    kmeans = KMeans(n_clusters=2)  # Adjust the number of clusters as needed
    kmeans.fit(X)
    df['Segment'] = kmeans.labels_

    # Evaluate the validity of customer segmentation
    segment_counts = df['Segment'].value_counts()
    segment_clv_mean = df.groupby('Segment')['CLV'].mean()
    segment_clv_std = df.groupby('Segment')['CLV'].std()

st.write("Segment Counts: ", segment_counts)
st.write("Segment CLV Mean: ", segment_clv_mean)
st.write("Segment CLV Standard Deviation:", segment_clv_std)

st.write("""Looking at the results, we can see that the second segment had a very very low standard deviation, which indicates that the customers in that segment have very similar CLV's, which tell us that they don't vary with their purchasing habits. They have a good, stable relationship with the business and a stable spending patterns.

Next lets see if we can get sillhouette score down by dropping outliers
""")

with st.echo():
    # Drop non-numeric columns
    data_numeric = data.select_dtypes(include=[np.number])

    # Handle missing values by imputing with the mean
    imputer = SimpleImputer(strategy='mean')
    data_imputed = pd.DataFrame(imputer.fit_transform(data_numeric), columns=data_numeric.columns)

    Q1 = data_imputed.quantile(0.25)             #To find the 25th percentile and 75th percentile.
    Q3 = data_imputed.quantile(0.75)

    IQR = Q3 - Q1                           #Inter Quantile Range (75th perentile - 25th percentile)

    lower=Q1-1.5*IQR                        #Finding lower and upper bounds for all values. All values outside these bounds are outliers
    upper=Q3+1.5*IQR

    keep = ((data_imputed.select_dtypes(include=['float64','int64'])<lower) | (data_imputed.select_dtypes(include=['float64','int64'])>upper))

    for index, row in keep.iterrows():
        if row['CustomerID']:
            data_imputed.drop(index=index, inplace=True)

    # Scale the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_imputed)

    # Initialize an empty list to store the within-cluster sum of squares (WCSS) values
    wcss = []

    # Set the range of the number of clusters to try
    min_clusters = 1
    max_clusters = 10

    # Perform K-means clustering for each value of k
    for k in range(min_clusters, max_clusters + 1):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(data_scaled)
        wcss.append(kmeans.inertia_)

    # Plot the WCSS values against the number of clusters
    plt.plot(range(min_clusters, max_clusters + 1), wcss)
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.title('Elbow Method')

st.pyplot(plt.figure())

with st.echo():
    # Customer Segmentation via RFM & K-Means

    # Scaling the Features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(rfm_data[['R', 'F', 'M']])

    # Applying K-means clustering
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(rfm_data)

    # Get cluster labels and add them to the RFM dataframe
    rfm_data['Cluster'] = kmeans.labels_

    cluster_sizes = rfm_data['Cluster'].value_counts()
    cluster_composition = rfm_data.groupby('Cluster').mean()

st.write(cluster_sizes)

st.write(cluster_composition)

with st.echo():
    # Visualize the clusters in a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Replace 'recency_column', 'frequency_column', and 'monetary_column' with the respective column names in your dataset
    ax.scatter(rfm_data['R'], rfm_data['F'], rfm_data['M'], c=rfm_data['Cluster'], cmap='viridis')
    ax.set_xlabel('R')
    ax.set_ylabel('F')
    ax.set_zlabel('M')

st.pyplot(plt.figure())

with st.echo(): 
    silhouette_avg = silhouette_score(rfm_data, rfm_data['Cluster'])
st.write("The average Silhouette Coefficient is:", silhouette_avg)

"""Lets try 3 clusters instead this time"""

# Customer Segmentation via RFM & K-Means

# Scaling the Features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(rfm_data[['R', 'F', 'M']])

# Applying K-means clustering
kmeans = KMeans(n_clusters=3)
kmeans.fit(rfm_data)

# Get cluster labels and add them to the RFM dataframe
rfm_data['Cluster'] = kmeans.labels_

cluster_sizes = rfm_data['Cluster'].value_counts()
cluster_composition = rfm_data.groupby('Cluster').mean()

st.write(cluster_sizes)

st.write(cluster_composition)

with st.echo():
    # Visualize the clusters in a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Replace 'recency_column', 'frequency_column', and 'monetary_column' with the respective column names in your dataset
    ax.scatter(rfm_data['R'], rfm_data['F'], rfm_data['M'], c=rfm_data['Cluster'], cmap='viridis')
    ax.set_xlabel('R')
    ax.set_ylabel('F')
    ax.set_zlabel('M')

st.pyplot(plt.figure())

with st.echo(): 
    silhouette_avg = silhouette_score(rfm_data, rfm_data['Cluster'])
st.write("The average Silhouette Coefficient is:", silhouette_avg)

st.write("""The score is only 0.5% better than the previous one""")

st.write(r.rfm_table)

with st.echo():
    segments = r.rfm_table[['CustomerID','segment']]

st.write(segments)

with st.echo():
    allPurchases = findAllPurchases(data)

with st.echo():
    filter = segments['segment'] == 'Champions'
    champs = segments.where(filter, inplace=False)
    champs.dropna(inplace=True)

st.write(champs)

with st.echo():
    allMostPurchased = {}

    for index, row in champs.iterrows():
        id = int(row['CustomerID'])

        # Get the customer's purchases
        customerPurchases = allPurchases.get(id)

        # Convert this to a data frame

        customerPurchasesDf = pd.DataFrame.from_dict(customerPurchases, orient='index')
        customerPurchasesDf.reset_index(inplace=True)
        customerPurchasesDf.rename(columns={"index":"StockCode", 0:"AmountPurchased"}, inplace=True)

        # Get the customer's most purchsed item

        stockCode = findMostPurchasedItem(customerPurchasesDf)

        # Add this to the dictionary

        currAmount = allMostPurchased.get(stockCode)

        if currAmount is None:
            currAmount = 0

        allMostPurchased.update({stockCode: currAmount + 1})

    # Convert allMostPurchased to a data frame

    allMostPurchasedDf = pd.DataFrame(list(allMostPurchased.items()), columns=['StockCode', 'AmountPurchased'])
    allMostPurchasedDf.sort_values(by=['AmountPurchased'], inplace=True, ascending=False)

st.write(allMostPurchasedDf)

st.write('Lets make a dataframe with everything in it')

with st.echo():
    clvData.drop('index', axis=1, inplace=True)

    def swap_columns(df, col1, col2):
        col_list = list(df.columns)
        x, y = col_list.index(col1), col_list.index(col2)
        col_list[y], col_list[x] = col_list[x], col_list[y]
        df = df[col_list]
        return df
    
    clvData = swap_columns(clvData, 'CustomerID', 'Total Revenue')

st.write(clvData)

with st.echo():
    # Solution that I didnt want to make but pandas/RFM library would not sort for the life of them

    dfdc = {}

    for index, row in r.rfm_table.iterrows():
        custid = int(row['CustomerID'])
        da = [row['recency'], row['frequency'], row['monetary_value'], row['r'], row['f'], row['m'], row['rfm_score'], row['segment']]
        

        dfdc.update({custid: da})

    # Convert dfdc to a data frame

    dfdcDf = pd.DataFrame.from_dict(dfdc, orient='index')
    dfdcDf.reset_index(inplace=True)

st.write(dfdcDf)

with st.echo():
    clvData.insert(9, 'Recency', dfdcDf[0])
    clvData.insert(10, 'Frequency', dfdcDf[1])
    clvData.insert(11, 'Monetary Value', dfdcDf[2])
    clvData.insert(12, 'R', dfdcDf[3])
    clvData.insert(13, 'F', dfdcDf[4])
    clvData.insert(14, 'M', dfdcDf[5])
    clvData.insert(15, 'RFM Score', dfdcDf[6])
    clvData.insert(16, 'Segment', dfdcDf[7])

st.write(clvData)

with st.echo():
    purchases = []

    for index, row in clvData.iterrows():    
        id = int(row['CustomerID'])

        # Get the customer's purchases
        customerPurchases = allPurchases.get(id)

        # Convert this to a data frame

        customerPurchasesDf = pd.DataFrame.from_dict(customerPurchases, orient='index')
        customerPurchasesDf.reset_index(inplace=True)
        customerPurchasesDf.rename(columns={"index":"StockCode", 0:"AmountPurchased"}, inplace=True)

        # Get the customer's most purchsed item

        stockCode = findMostPurchasedItem(customerPurchasesDf)

        purchases.append(stockCode)

    clvData['Most Purchased Item'] = purchases

st.write(clvData)

with st.echo():
    # Get description from stock code

    def getDescription(stockCode):
        filter = data['StockCode'] == stockCode
        stock = data.where(filter, inplace=False)
        stock.dropna(inplace=True)
        return stock['Description'].values[0]
    
    # This takes like 15 minutes to run, so I'm not going to run it again. I'll just save the data frame to a csv file

    descs = []

    for index, row in clvData.iterrows():
        stockCode = row['Most Purchased Item']
        description = getDescription(stockCode)
        descs.append(description)

    clvData['Most Purchased Item Description'] = descs

st.write(clvData)

with st.echo():
    clvData['RFM CLV'] = rfm_data['CLV']

st.write(clvData)