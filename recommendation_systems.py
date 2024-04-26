#############################################
### Business Problem
#############################################

# Product recommendations are requested based on the basket information of 3 different users.
# Utilize association rules to generate the most suitable product recommendations, which can be one or more products.
# Derive the decision rules from the 2010-2011 data of customers in Germany.
# Product ID in User 1's basket: 21987
# Product ID in User 2's basket: 23235
# Product ID in User 3's basket: 22747

# Dataset Story
# The 'Online Retail II' dataset represents the online sales transactions of a retail company based in the UK from 01/12/2009 to 09/12/2011.
# The company’s product catalog includes gift items and most of its customers are known to be wholesalers.

# Task 1: Data Preparation
# Step 1: Load the 2010-2011 sheet from the Online Retail II dataset.

import pandas as pd
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
# Ensures output appears on one line.
pd.set_option('display.expand_frame_repr', False)
from mlxtend.frequent_patterns import apriori, association_rules

# Data source: https://archive.ics.uci.edu/ml/datasets/Online+Retail+II

df_ = pd.read_excel("recommender_systems/datasets/online_retail_II.xlsx",
                    sheet_name="Year 2010-2011")
df = df_.copy()
df.head()
# Step 2: Drop observations where StockCode is 'POST'. ('POST' is a charge added to every invoice and does not represent a product.)

df = df[df['StockCode'] != 'POST']

# Step 3: Drop observations containing missing values.
df.dropna(inplace=True)

# Step 4: Remove entries from the dataset where Invoice contains 'C' (indicating cancellation).
df = df[~df['Invoice'].astype(str).str.contains('C')]

# Step 5: Filter out observations where the Price is less than zero.
df = df[df['Price'] > 0]

# Step 6: Examine and, if necessary, cap outliers in the Price and Quantity variables.
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def retail_data_prep(dataframe):
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    return dataframe

# Task 2: Generating Association Rules for German Customers

# Step 1: Define the create_invoice_product_df function to create an invoice-product pivot table as follows:
def create_invoice_product_df(dataframe, id=False):
    if id:
        return dataframe.groupby(['Invoice', "StockCode"])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)

def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)
# Step 2: Define the create_rules function to generate rules and find rules for German customers.

def create_rules(dataframe, id=True, country="Germany"):
    dataframe = dataframe[dataframe['Country'] == country]
    dataframe = create_invoice_product_df(dataframe, id)
    frequent_itemsets = apriori(dataframe, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
    return rules

df = df_.copy()
df = retail_data_prep(df)
rules = create_rules(df)

# Task 3: Making Product Recommendations Based on Product IDs in Users' Baskets

# Step 1: Use the check_id function to find the names of the specified products.
product_id = 21987
check_id(df, product_id)

sorted_rules = rules.sort_values("lift", ascending=False)

recommendation_list = []

for i, product in enumerate(sorted_rules["antecedents"]):
    for j in list(product):
        if j == product_id:
            recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

recommendation_list[0:3]

# Step 2: Use the arl_recommender function to make product recommendations for 3 users.
def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

    return recommendation_list[0:rec_count]


arl_recommender(rules, 21987, 1)
arl_recommender(rules, 23235, 2)
arl_recommender(rules, 22747, 3)

# Step 3: Check the names of the recommended products.

# ['PACK OF 6 SKULL PAPER CUPS']
# Output[17]: [21989, 21988, 21989]
check_id(df, 21989)
# ['PACK OF 20 SKULL PAPER NAPKINS']
check_id(df, 21988)
# ['PACK OF 6 SKULL PAPER PLATES']
check_id(df, 22746)
# ["POPPY'S PLAYHOUSE LIVINGROOM "]
check_id(df, 23243)
# ['SET OF TEA COFFEE SUGAR TINS PANTRY']
