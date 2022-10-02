# rcParams: contains a number of parameters, used for customizing matplotlib

# import warnings
#import itertools
import numpy as np
import matplotlib.pyplot as plt
# warnings.filterwarnings("ignore")
#plt.style.use('fivethirtyeight')
import pandas as pd
import statsmodels.api as sm
import matplotlib

# Customsing MatPlotLib
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'

# Load Excel data
df = pd.read_excel("Superstore.xls")
#print("df =", df)

# Time Series Analysis and Forecast of Furniture Sales
furniture = df.loc[df['Category'] == 'Furniture'] # all the rows with Category 'Furniture'
#print("furniture =", furniture)
#print("min date =", furniture['Order Date'].min(), "\nmax date =", furniture['Order Date'].max()) # represents 4 year data

# ***************  DATA PREPROCESSING  ********************

# 1. Dropping Irrevant Attributes
irrelevant_attrs = ['Row ID', 'Order ID', 'Ship Date', 'Ship Mode', 'Customer ID', 'Customer Name', 'Segment', 'Country', 'City', 'State', 'Postal Code', 'Region', 'Product ID', 'Category', 'Sub-Category', 'Product Name', 'Quantity', 'Discount', 'Profit']
#print("type =",type(furniture))
furniture.drop(irrelevant_attrs,axis=1,inplace=True) # drops cols from furniture attribute
#print("\nAfter Dropping Irrelevant Attributes:\n", furniture)
furniture = furniture.sort_values('Order Date')
#print("\nAfter Sorting:\n", furniture)
#print()

# 2. Looking for Missing Values
#print(furniture.isnull().sum())
#print()

# ***************  Indexing with Time Series  ********************
furniture = furniture.groupby('Order Date')['Sales'].sum().reset_index()
#print("\nAfter Grouping by Order Date:\n", furniture) # Furniture sales per day
furniture = furniture.set_index('Order Date')
#print("furniture.index =", furniture.index)
y = furniture['Sales'].resample('MS').mean() # avergae furniture sales of each month
#print(y)
#print(furniture)
#print(y['2017'])

# ***************  Visualizing Average Furniture Sales per month   ********************
#y.plot(figsize=(15,6))
#plt.show()

from pylab import rcParams
rcParams['figure.figsize'] = 18, 8

decomposition = sm.tsa.seasonal_decompose(y, model='additive')
fig = decomposition.plot()
plt.show()

# ***************  Sales Forecasting with Arima   ********************

