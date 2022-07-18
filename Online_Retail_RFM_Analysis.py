import datetime as dt
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

df_=pd.read_excel("online_retail_II.xlsx", sheet_name="Year 2010-2011")
df= df_.copy()

df.head()
df.shape
df.describe().T
df.isnull().sum()
df.dropna(inplace=True)

#number of unique products
df["Description"].nunique() #4459

#How many of each product are there?
df["Description"].value_counts().head(7)

#Sort the 5 most ordered products from most to least
df.groupby("Description").agg({"Quantity": "sum"}).sort_values("Quantity", ascending=False).head(5)  #

#The 'C' in the invoices shows the canceled transactions. Remove the canceled transactions from the dataset.
df=df[~df["Invoice"].str.contains("C",na=False)]

#Create a variable named 'TotalPrice' that represents the total earnings per invoice
df["TotalPrice"] = df["Quantity"] * df["Price"]

##RFM Metrics
analysis_time=dt.datetime(2011, 12, 11)

rfm_=df.groupby("Customer ID").agg({"InvoiceDate":lambda date:(analysis_time-date.max()).days,
                               "Invoice":lambda num:num.nunique(),
                                "TotalPrice":lambda TotalPrice:TotalPrice.sum()})

rfm_.columns=["recency","frequency","monetary"]

rfm_=rfm_[rfm_["monetary"]>0]

## RFM Scores
rfm_["recency_score"]=pd.qcut(rfm_["recency"],5,labels=[5,4,3,2,1])
rfm_["frequency_score"]=pd.qcut(rfm_["frequency"].rank(method="first"),5,labels=[1,2,3,4,5])  ## duplicate hatası
rfm_["monetary_score"]=pd.qcut(rfm_["monetary"],5,labels=[1,2,3,4,5])

rfm_["RFM_SCORE"] = (rfm_['recency_score'].astype(str) +
                    rfm_['frequency_score'].astype(str))
rfm_.head()

## RFM Segments

seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}

rfm_["segment"]=rfm_["RFM_SCORE"].replace(seg_map,regex=True)

##Case1:Select 3 segments that you consider important.
# Interpret these three segments both in terms of action decisions and in terms of the structure of the segments (mean RFM values).

rfm_[["segment", "recency", "frequency", "monetary"]].groupby("segment").agg(["mean", "count"])

##Case2:Select the customerIDs of the "LoyalCustomers" class and get the output in excel.
rfm_[rfm_["segment"]=="loyal_customers"].index

new_df = pd.DataFrame()
new_df["loyal_customers_ID"] = rfm_[rfm_["segment"]=="loyal_customers"].index
new_df["loyal_customers_ID"] = new_df["loyal_customers_ID"].astype(int)
new_df.to_csv("loyal_customers_ID.csv")