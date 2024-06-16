import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from mpl_toolkits.mplot3d import Axes3D

# Load the dataset
file_path = 'updated_dataset.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataset and its general information
data.head(), data.info(), data.describe(include='all')



# Convert 'DateKey' to datetime format
data['DateKey'] = pd.to_datetime(data['DateKey'])

data['Month'] = data['DateKey'].dt.month
data['DayOfWeek'] = data['DateKey'].dt.dayofweek
data['YearMonth'] = data['DateKey'].dt.to_period('M')


# Set DateKey as the index
data.set_index('DateKey', inplace=True)

# Resample the data weekly, summing up Sales Quantity and Sales Amount for each item
weekly_sales = data.groupby('Item').resample('W').agg({'Sales Quantity': 'sum', 'Sales Amount': 'sum'}).reset_index()

# Display the structure of the aggregated weekly data and a sample for any random item
weekly_sales.info()

print("Weekly Sales")
print(weekly_sales.head(10))


# Excluding the item with insufficient data and performing decomposition for the rest
valid_items = ['Thresher Spicy Mints', 'Even Better Whole Milk', 'Best Choice Potato Chips', 'Fast Corn Chips']

# Preparing plots
fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(14, 12), sharex=True)

for i, item in enumerate(valid_items):
    # Filter and prepare data
    item_data = weekly_sales[weekly_sales['Item'] == item].set_index('DateKey')
    item_data = item_data.asfreq('W', fill_value=0)
    
    # Decompose the time series
    decomposition = seasonal_decompose(item_data['Sales Quantity'], model='additive')
    
    # Plotting the observed and trend (baseline) sales
    axes[i, 0].plot(decomposition.observed.index, decomposition.observed, label='Observed Sales')
    axes[i, 0].plot(decomposition.trend.index, decomposition.trend, label='Baseline (Trend)', linestyle='--')
    axes[i, 0].set_title(f'Observed vs. Baseline for {item}')
    axes[i, 0].legend()
    
    # Plotting the seasonality
    axes[i, 1].plot(decomposition.seasonal.index, decomposition.seasonal)
    axes[i, 1].set_title(f'Seasonality for {item}')

plt.tight_layout()
plt.show()

print("ITEM DATA BASELINE SALES")
print(item_data)

# Save to HTML file
item_data_html_path = 'Weekly_Sales_Data.html'
item_data.to_html(item_data_html_path)



# Randomly select an item from the dataset
random_item = np.random.choice(weekly_sales['Item'].unique())

# Filter data for the selected item
item_data = weekly_sales[weekly_sales['Item'] == random_item].set_index('DateKey')

# Handle cases where there might be missing weeks by filling with zeros
item_data = item_data.asfreq('W', fill_value=0)

# Decompose the time series of Sales Quantity
decomposition = seasonal_decompose(item_data['Sales Quantity'], model='additive')

# Plot the decomposition
fig, ax = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
decomposition.observed.plot(ax=ax[0], title='Observed Sales')
decomposition.trend.plot(ax=ax[1], title='Trend')
decomposition.seasonal.plot(ax=ax[2], title='Seasonality')
decomposition.resid.plot(ax=ax[3], title='Residuals')

plt.tight_layout()
plt.show()

# Return the name of the randomly selected item
print("Select random item")
print(random_item)


#Cannibalization Analysis

# Convert 'Invoice Date' to datetime format for easier manipulation
data['Invoice Date'] = pd.to_datetime(data['Invoice Date'])

# Extract year and month from 'Invoice Date' for grouping
data['YearMonth'] = data['Invoice Date'].dt.to_period('M')

# Group by 'YearMonth' and 'Item' to calculate monthly sales
monthly_sales_by_item = data.groupby(['YearMonth', 'Item'])['Sales Amount'].sum().unstack()

print("Monthly Sales by Item")
print(monthly_sales_by_item.head())



# Find the first sales date for each item
first_sales_date = data.groupby('Item')['Invoice Date'].min().sort_values()

print("first sales date for each item")
print(first_sales_date.head())


# Convert first_sales_date to a DataFrame for merging
launch_dates = first_sales_date.reset_index()
launch_dates.columns = ['Item', 'Launch Date']

# Merge the launch dates with the main dataset
data_with_launch = pd.merge(data, launch_dates, on='Item', how='left')

# Filter records to keep only those from the launch date onwards
data_post_launch = data_with_launch[data_with_launch['Invoice Date'] >= data_with_launch['Launch Date']]

# Display the first few rows of the filtered dataset
print("Date post launch")
print(data_post_launch.head())
print(data_post_launch.shape)


# Group data post-launch by item and month, summarizing sales amount
grouped_data = data_post_launch.groupby(['Item', 'YearMonth'])['Sales Amount'].sum().unstack()

# Display the first few rows of the grouped data
print("Post Launch by Item and Month with Sales Amount ")
print(grouped_data.head())



# Aggregate sales data by item and month
monthly_sales_data = data.groupby(['Item', 'YearMonth'])['Sales Amount'].sum().reset_index()

# Pivot data for better visualization
monthly_pivot_data = monthly_sales_data.pivot(index='YearMonth', columns='Item', values='Sales Amount')

# Display the head of the pivoted data
print(monthly_pivot_data.head())


# Remove non-numeric entries from the DataFrame
monthly_pivot_data_cleaned = monthly_pivot_data.dropna()

# Check the cleaned DataFrame
print(monthly_pivot_data_cleaned.info())
print(monthly_pivot_data_cleaned.head())

# Fill NaN values with zero
monthly_pivot_data_filled = monthly_pivot_data.fillna(0)

# Check the DataFrame after filling NaNs
print(monthly_pivot_data_filled.info())
print(monthly_pivot_data_filled.head())



# Filter data for 'American' and 'Atomic' prefixed products
american_products = monthly_pivot_data_filled.filter(regex='^American')
atomic_products = monthly_pivot_data_filled.filter(regex='^Atomic')

# Calculate the total sales for each group over time
american_total_sales = american_products.sum(axis=1)
atomic_total_sales = atomic_products.sum(axis=1)

# Convert PeriodIndex to DateTimeIndex for plotting
american_total_sales.index = pd.to_datetime(american_total_sales.index.to_timestamp())
atomic_total_sales.index = pd.to_datetime(atomic_total_sales.index.to_timestamp())

# Retry plotting
plt.figure(figsize=(12, 8), facecolor='white')
plt.plot(american_total_sales, label='American Products')
plt.plot(atomic_total_sales, label='Atomic Products')
plt.title('Total Sales of American vs Atomic Products Over Time')
plt.xlabel('Month')
plt.ylabel('Total Sales')
plt.legend()
plt.show()

# Filter data for 'BBB Best' prefixed products
bbb_best_products = monthly_pivot_data_filled.filter(regex='^BBB Best')
# Calculate the total sales for the 'BBB Best' group over time
bbb_best_total_sales = bbb_best_products.sum(axis=1)
# Convert PeriodIndex to DateTimeIndex for plotting
bbb_best_total_sales.index = pd.to_datetime(bbb_best_total_sales.index.to_timestamp())


# Filter data for 'Applause Canned' prefixed products
applause_canned_products = monthly_pivot_data_filled.filter(regex='^Applause Canned')
# Calculate the total sales for the 'Applause Canned' group over time
applause_canned_total_sales = applause_canned_products.sum(axis=1)
# Convert PeriodIndex to DateTimeIndex for plotting
applause_canned_total_sales.index = pd.to_datetime(applause_canned_total_sales.index.to_timestamp())


# Filter data for 'Golden' prefixed products
golden_products = monthly_pivot_data_filled.filter(regex='^Golden')
# Calculate the total sales for the 'Golden' group over time
golden_total_sales = golden_products.sum(axis=1)
# Convert PeriodIndex to DateTimeIndex for plotting
golden_total_sales.index = pd.to_datetime(golden_total_sales.index.to_timestamp())


# Plot the total sales over time for each group to visualize potential cannibalization
plt.figure(figsize=(14, 8), facecolor='white')
plt.plot(bbb_best_total_sales, label='BBB Best Products')
plt.plot(applause_canned_total_sales, label='Applause Canned Products')
plt.plot(golden_total_sales, label='Golden Products')
plt.title('Total Sales of BBB Best, Applause Canned, and Golden Products Over Time')
plt.xlabel('Month')
plt.ylabel('Total Sales')
plt.legend()
plt.show()


# Aggregate sales data over time for the top 'BBB Best' products
bbb_best_apple_preserves_sales = monthly_pivot_data_filled['BBB Best Apple Preserves']
bbb_best_grape_jelly_sales = monthly_pivot_data_filled['BBB Best Grape Jelly']
bbb_best_extra_chunky_pb_sales = monthly_pivot_data_filled['BBB Best Extra Chunky Peanut Butter']

# Convert PeriodIndex to DateTimeIndex for plotting
bbb_best_apple_preserves_sales.index = pd.to_datetime(bbb_best_apple_preserves_sales.index.to_timestamp())
bbb_best_grape_jelly_sales.index = pd.to_datetime(bbb_best_grape_jelly_sales.index.to_timestamp())
bbb_best_extra_chunky_pb_sales.index = pd.to_datetime(bbb_best_extra_chunky_pb_sales.index.to_timestamp())

# Plot the sales trends
plt.figure(figsize=(14, 8), facecolor='white')
plt.plot(bbb_best_apple_preserves_sales, label='BBB Best Apple Preserves')
plt.plot(bbb_best_grape_jelly_sales, label='BBB Best Grape Jelly')
plt.plot(bbb_best_extra_chunky_pb_sales, label='BBB Best Extra Chunky Peanut Butter')
plt.title('Sales Trends of Top BBB Best Products Over Time')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.legend()
plt.show()


# Aggregate sales data over time for the top 'Applause Canned' products
applause_canned_mixed_fruit_sales = monthly_pivot_data_filled['Applause Canned Mixed Fruit']
applause_canned_peaches_sales = monthly_pivot_data_filled['Applause Canned Peaches']

# Convert PeriodIndex to DateTimeIndex for plotting
applause_canned_mixed_fruit_sales.index = pd.to_datetime(applause_canned_mixed_fruit_sales.index.to_timestamp())
applause_canned_peaches_sales.index = pd.to_datetime(applause_canned_peaches_sales.index.to_timestamp())

# Plot the sales trends
plt.figure(figsize=(14, 8), facecolor='white')
plt.plot(applause_canned_mixed_fruit_sales, label='Applause Canned Mixed Fruit')
plt.plot(applause_canned_peaches_sales, label='Applause Canned Peaches')
plt.title('Sales Trends of Top Applause Canned Products Over Time')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.legend()
plt.show()


# Aggregate sales data over time for the top 'Golden' products
golden_frozen_corn_sales = monthly_pivot_data_filled['Golden Frozen Corn']
golden_frozen_broccoli_sales = monthly_pivot_data_filled['Golden Frozen Broccoli']
golden_frozen_chicken_thighs_sales = monthly_pivot_data_filled['Golden Frozen Chicken Thighs']

# Convert PeriodIndex to DateTimeIndex for plotting
golden_frozen_corn_sales.index = pd.to_datetime(golden_frozen_corn_sales.index.to_timestamp())
golden_frozen_broccoli_sales.index = pd.to_datetime(golden_frozen_broccoli_sales.index.to_timestamp())
golden_frozen_chicken_thighs_sales.index = pd.to_datetime(golden_frozen_chicken_thighs_sales.index.to_timestamp())

# Plot the sales trends
plt.figure(figsize=(14, 8), facecolor='white')
plt.plot(golden_frozen_corn_sales, label='Golden Frozen Corn')
plt.plot(golden_frozen_broccoli_sales, label='Golden Frozen Broccoli')
plt.plot(golden_frozen_chicken_thighs_sales, label='Golden Frozen Chicken Thighs')
plt.title('Sales Trends of Top Golden Products Over Time')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.legend()
plt.show()


# Assuming the data is correctly indexed and loaded
plt.figure(figsize=(14, 8))
plt.plot(bbb_best_apple_preserves_sales, label='Apple Preserves', marker='o', linestyle='-')
plt.plot(bbb_best_grape_jelly_sales, label='Grape Jelly', marker='^', linestyle='--')
plt.plot(bbb_best_extra_chunky_pb_sales, label='Extra Chunky Peanut Butter', linestyle=':')

# Add markers only if dates exist
promo_dates = [pd.to_datetime('2017-01-01'), pd.to_datetime('2018-05-01')]
for date in promo_dates:
    if date in bbb_best_apple_preserves_sales.index:
        plt.scatter(date, bbb_best_apple_preserves_sales.loc[date], color='red', s=100, label='Promotion', zorder=5)

plt.title('Sales Trends of Top BBB Best Products Over Time')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.legend()
plt.show()


# Assuming data is loaded and DataFrame indices are converted to datetime format appropriately
applause_canned_mixed_fruit_sales.index = pd.to_datetime(applause_canned_mixed_fruit_sales.index)
applause_canned_peaches_sales.index = pd.to_datetime(applause_canned_peaches_sales.index)

# Create the plot
plt.figure(figsize=(14, 8))
plt.plot(applause_canned_mixed_fruit_sales, label='Mixed Fruit', marker='o', linestyle='-')
plt.plot(applause_canned_peaches_sales, label='Peaches', marker='^', linestyle='--')

# Add markers for promotions if dates exist
promo_dates = [pd.to_datetime('2019-06-01'), pd.to_datetime('2020-01-01')]  # Example promo dates
for date in promo_dates:
    if date in applause_canned_mixed_fruit_sales.index:
        plt.scatter(date, applause_canned_mixed_fruit_sales.loc[date], color='red', s=100, label='Promotion on Mixed Fruit', zorder=5)
    if date in applause_canned_peaches_sales.index:
        plt.scatter(date, applause_canned_peaches_sales.loc[date], color='purple', s=100, label='Promotion on Peaches', zorder=5)

plt.title('Sales Trends of Applause Canned Products Over Time')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.legend()
plt.show()


# Assuming data is loaded and DataFrame indices are converted to datetime format appropriately
golden_frozen_corn_sales.index = pd.to_datetime(golden_frozen_corn_sales.index)
golden_frozen_broccoli_sales.index = pd.to_datetime(golden_frozen_broccoli_sales.index)
golden_frozen_chicken_thighs_sales.index = pd.to_datetime(golden_frozen_chicken_thighs_sales.index)

# Create the plot
plt.figure(figsize=(14, 8))
plt.plot(golden_frozen_corn_sales, label='Frozen Corn', linestyle='-', marker='o')
plt.plot(golden_frozen_broccoli_sales, label='Frozen Broccoli', linestyle='--', marker='^')
plt.plot(golden_frozen_chicken_thighs_sales, label='Frozen Chicken Thighs', linestyle=':', marker='s')

# Add markers for promotions if dates exist
promo_dates = [pd.to_datetime('2019-08-01'), pd.to_datetime('2020-02-01')]  # Example promo dates
for date in promo_dates:
    if date in golden_frozen_corn_sales.index:
        plt.scatter(date, golden_frozen_corn_sales.loc[date], color='red', s=100, label='Promotion on Corn', zorder=5)
    if date in golden_frozen_broccoli_sales.index:
        plt.scatter(date, golden_frozen_broccoli_sales.loc[date], color='purple', s=100, label='Promotion on Broccoli', zorder=5)
    if date in golden_frozen_chicken_thighs_sales.index:
        plt.scatter(date, golden_frozen_chicken_thighs_sales.loc[date], color='green', s=100, label='Promotion on Chicken Thighs', zorder=5)

plt.title('Sales Trends of Golden Products Over Time')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.legend()
plt.show()



# Print the total sales data for American and Atomic products
print("Total Sales of American Products:")
print(american_total_sales)

print("\nTotal Sales of Atomic Products:")
print(atomic_total_sales)

# Print the sales data for specific product lines within BBB Best, Applause Canned, and Golden categories
print("\nSales Trends of Top BBB Best Products:")
print("BBB Best Apple Preserves Sales:")
print(bbb_best_apple_preserves_sales)
print("\nBBB Best Grape Jelly Sales:")
print(bbb_best_grape_jelly_sales)
print("\nBBB Best Extra Chunky Peanut Butter Sales:")
print(bbb_best_extra_chunky_pb_sales)

print("\nSales Trends of Top Applause Canned Products:")
print("Applause Canned Mixed Fruit Sales:")
print(applause_canned_mixed_fruit_sales)
print("\nApplause Canned Peaches Sales:")
print(applause_canned_peaches_sales)

print("\nSales Trends of Top Golden Products:")
print("Golden Frozen Corn Sales:")
print(golden_frozen_corn_sales)
print("\nGolden Frozen Broccoli Sales:")
print(golden_frozen_broccoli_sales)
print("\nGolden Frozen Chicken Thighs Sales:")
print(golden_frozen_chicken_thighs_sales)




# Define thresholds for different types of promotions based on discount distribution
low_threshold = data['Discount Amount'].quantile(0.33)
high_threshold = data['Discount Amount'].quantile(0.66)


# Create categorical variables for promotion types
data['Promo_Type'] = pd.cut(data['Discount Amount'], 
                            bins=[0, low_threshold, high_threshold, data['Discount Amount'].max()], 
                            labels=['Low', 'Medium', 'High'],
                            include_lowest=True)

# Promotional effectiveness analysis
promo_effectiveness = data.groupby('Promo_Type').agg({
    'Sales Amount': ['mean', 'std'],
    'Sales Quantity': ['mean', 'std'],
    'Discount Amount': ['count']
}).reset_index()



# Define a threshold to determine when a discount is significant enough to be considered a promotion
discount_threshold = data['Discount Amount'].quantile(0.75)  # Using the 75th percentile as a threshold

# Print the discount threshold we're using to identify promotions
print("Discount Threshold for Identifying Promotions:")
print(f"Discounts above {discount_threshold:.2f} are considered promotions.")

# Display summary statistics for Discount Amount
print("Discount Amount Summary:")
print(data['Discount Amount'].describe())

# Example of setting a new, informed threshold
# Example of reassessing the threshold
# Setting the threshold to a new percentile based on distribution
discount_threshold = data['Discount Amount'].quantile(0.75)  # Adjust as needed
data['Is_Promotion'] = (data['Discount Amount'] >= discount_threshold).astype(int)

print("Is_Promotion Value Counts:")
print(data['Is_Promotion'].value_counts())


data['Is_Promotion'] = (data['Discount Amount'] >= discount_threshold).astype(int)

# Check how the new threshold affects the Is_Promotion indicator
print("Updated Promotional Data Summary (Head):")

# Check if 'DateKey' is the index
if data.index.name == 'DateKey':
    data.reset_index(inplace=True)

# Verify columns
print("Columns in DataFrame:", data.columns)

print(data[['DateKey', 'Sales Amount', 'Discount Amount', 'Is_Promotion']].head(20))


# Preparing the dataset for the enhanced regression model, including the promotional indicator
X_promo = data[['Month', 'DayOfWeek', 'Discount Amount', 'Sales Quantity', 'Is_Promotion']]  # Predictor variables
X_promo = sm.add_constant(X_promo)  # Adds a constant term to the predictor variables
y_promo = data['Sales Amount']  # Response variable

# Running the regression model including the promotional indicator
model_promo = sm.OLS(y_promo, X_promo, missing='drop').fit()  # 'drop' handles any missing values in predictors
regression_results_promo = model_promo.summary()

print("Print regression")
print(regression_results_promo)


# Define thresholds for different types of promotions based on discount distribution
low_threshold = data['Discount Amount'].quantile(0.33)
high_threshold = data['Discount Amount'].quantile(0.66)

# Create categorical variables for promotion types
data['Promo_Type'] = pd.cut(data['Discount Amount'], 
                            bins=[0, low_threshold, high_threshold, data['Discount Amount'].max()], 
                            labels=['Low', 'Medium', 'High'],
                            include_lowest=True)

# Group by Promo_Type and calculate average sales, quantities, and count of transactions
promo_effectiveness = data.groupby('Promo_Type').agg({
    'Sales Amount': ['mean', 'std'],
    'Sales Quantity': ['mean', 'std'],
    'Discount Amount': ['count']
}).reset_index()

promo_effectiveness.columns = ['Promo_Type', 'Average Sales', 'Sales Std', 'Average Quantity', 'Quantity Std', 'Transactions Count']

print("Promo Effectiveness")
print(promo_effectiveness)



# Create a binary indicator for promotional periods
data['Is_Promotion'] = (data['Discount Amount'] >= discount_threshold).astype(int)

# Filter data into promotional and non-promotional dataframes
promo_df = data[data['Is_Promotion'] == 1]
non_promo_df = data[data['Is_Promotion'] == 0]


# Calculate total sales amount during promotional and non-promotional periods for each region
promo_sales_by_region = promo_df.groupby('Region')['Sales Amount'].sum()
non_promo_sales_by_region = non_promo_df.groupby('Region')['Sales Amount'].sum()

# Combine promotional and non-promotional sales data for each region
total_sales_by_region = promo_sales_by_region.add(non_promo_sales_by_region, fill_value=0)

# Visualize the results
plt.figure(figsize=(10, 6))
total_sales_by_region.sort_values().plot(kind='barh', color='skyblue')
plt.title('Total Sales Amount by Region')
plt.xlabel('Total Sales Amount')
plt.ylabel('Region')
plt.grid(axis='x')
plt.show()

# Display total sales by region
print("Total Sales Amount by Region:")
print(total_sales_by_region)

# Ensure the 'Invoice Date' column is converted to datetime format
if 'Invoice Date' in data.columns:
    data['Invoice Date'] = pd.to_datetime(data['Invoice Date'])
else:
    raise ValueError("Invoice Date column not found in the dataset")

# Now that 'Invoice Date' is a datetime object, create the 'Month' period column
data['Month'] = data['Invoice Date'].dt.to_period('M')

# Assuming 'Region' column is correctly set, group data by region and month
sales_by_region_monthly = data.groupby(['Region', 'Month']).agg({
    'Sales Amount': 'sum',
    'Sales Quantity': 'sum'
}).reset_index()

# Get unique regions to plot data
regions = data['Region'].unique()

# Plot sales trends for each region
for region in regions:
    region_data = sales_by_region_monthly[sales_by_region_monthly['Region'] == region]
    plt.figure(figsize=(10, 4))
    # Ensure the 'Month' data is sorted if not in sequential order
    region_data = region_data.sort_values(by='Month')
    plt.plot(region_data['Month'].astype(str), region_data['Sales Amount'], marker='o', linestyle='-', color='b')
    plt.title(f'Total Sales Amount Trend for {region}')
    plt.xlabel('Month')
    plt.ylabel('Total Sales Amount')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()




    # Customer Segmentation using RFM metrics
# Calculate RFM metrics
today = pd.to_datetime('today')
rfm = data.groupby('Custkey').agg({
    'Invoice Date': lambda x: (today - x.max()).days,
    'Invoice Number': 'nunique',
    'Sales Amount': 'sum'
}).rename(columns={'Invoice Date': 'Recency', 'Invoice Number': 'Frequency', 'Sales Amount': 'Monetary'})

# Apply K-means clustering
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm)
kmeans = KMeans(n_clusters=4, random_state=42).fit(rfm_scaled)
rfm['Cluster'] = kmeans.labels_

print(rfm)


fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot for each cluster
scatter = ax.scatter(rfm['Recency'],
                     rfm['Frequency'],
                     rfm['Monetary'],
                     c=rfm['Cluster'],  # Color by cluster
                     cmap='viridis',    # Color map to use
                     marker='o',        # Marker style
                     alpha=0.6)         # Transparency

# Labeling
ax.set_xlabel('Recency')
ax.set_ylabel('Frequency')
ax.set_zlabel('Monetary')
ax.set_title('3D Scatter Plot of RFM Clusters')

# Legend with cluster labels
legend = ax.legend(*scatter.legend_elements(), title="Clusters")
ax.add_artist(legend)

plt.show()

# 2D Scatter Plot: Recency vs. Monetary
plt.figure(figsize=(10, 5))
plt.scatter(rfm['Recency'], rfm['Monetary'], c=rfm['Cluster'], cmap='viridis', alpha=0.6)
plt.colorbar(label='Cluster')
plt.xlabel('Recency (days)')
plt.ylabel('Monetary (total spend)')
plt.title('Recency vs. Monetary with Cluster Coloring')
plt.show()

# 2D Scatter Plot: Frequency vs. Monetary
plt.figure(figsize=(10, 5))
plt.scatter(rfm['Frequency'], rfm['Monetary'], c=rfm['Cluster'], cmap='viridis', alpha=0.6)
plt.colorbar(label='Cluster')
plt.xlabel('Frequency (number of transactions)')
plt.ylabel('Monetary (total spend)')
plt.title('Frequency vs. Monetary with Cluster Coloring')
plt.show()


# Segmenting promo into Low, Medium, and High
data['Discount Category'] = pd.cut(data['Discount Percentage'], 
                                   bins=[-float('inf'), 10, 30, float('inf')],
                                   labels=['Low', 'Medium', 'High'])

# Analyze the effects of discount categories on sales metrics
discount_analysis = data.groupby('Discount Category').agg(
    Average_Sales_Amount=pd.NamedAgg(column='Sales Amount', aggfunc='mean'),
    Average_Sales_Quantity=pd.NamedAgg(column='Sales Quantity', aggfunc='mean'),
    Average_Sales_Margin=pd.NamedAgg(column='Sales Margin Amount', aggfunc='mean')
).reset_index()

print("Discount Analysis")
print(discount_analysis)



# Determine the top 10 items by total sales amount
top_items = data.groupby('Item')['Sales Amount'].sum().nlargest(10)

# Filter the dataset to include only the top items
top_items_data = data[data['Item'].isin(top_items.index)]

# Prepare data for plotting
plot_data = top_items_data.groupby('Item').agg(
    Sales_Amount=pd.NamedAgg(column='Sales Amount', aggfunc='sum'),
    Average_Sales_Margin=pd.NamedAgg(column='Sales Margin Amount', aggfunc='mean'),
    Average_Discount_Percentage=pd.NamedAgg(column='Discount Percentage', aggfunc='mean'),
    Total_Sales_Quantity=pd.NamedAgg(column='Sales Quantity', aggfunc='sum')
).reset_index()

# Create visualizations
fig, ax = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Performance Metrics for Top Items')

sns.barplot(x='Sales_Amount', y='Item', data=plot_data, ax=ax[0, 0])
ax[0, 0].set_title('Total Sales Amount')
ax[0, 0].set_xlabel('Sales Amount ($)')
ax[0, 0].set_ylabel('Item')

sns.barplot(x='Average_Sales_Margin', y='Item', data=plot_data, ax=ax[0, 1])
ax[0, 1].set_title('Average Sales Margin')
ax[0, 1].set_xlabel('Sales Margin ($)')
ax[0, 1].set_ylabel('')

sns.barplot(x='Average_Discount_Percentage', y='Item', data=plot_data, ax=ax[1, 0])
ax[1, 0].set_title('Average Discount Percentage')
ax[1, 0].set_xlabel('Discount Percentage (%)')
ax[1, 0].set_ylabel('Item')

sns.barplot(x='Total_Sales_Quantity', y='Item', data=plot_data, ax=ax[1, 1])
ax[1, 1].set_title('Total Sales Quantity')
ax[1, 1].set_xlabel('Quantity')
ax[1, 1].set_ylabel('')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Define thresholds for low, medium, and high promotions
low_threshold = data['Discount Amount'].quantile(0.33)
medium_threshold = data['Discount Amount'].quantile(0.66)


# Categorize each sale into promo categories
data['Promo Category'] = pd.cut(data['Discount Amount'], 
                                bins=[0, low_threshold, medium_threshold, float('inf')], 
                                labels=['Low', 'Medium', 'High'],
                                include_lowest=True)


# Aggregate data to see sales amount, quantity, etc., by item and promo category
promo_item_summary = data.groupby(['Item', 'Promo Category']).agg({
    'Sales Amount': 'sum',
    'Sales Quantity': 'sum',
    'Sales Margin Amount': 'mean'
}).reset_index()


# Plotting top items' sales by promo category
top_items = promo_item_summary['Item'].value_counts().index[:10]  # Top 10 items by occurrence

# Filter data for visualization
top_items_data = promo_item_summary[promo_item_summary['Item'].isin(top_items)]

# Sales Amount by Promo Category
plt.figure(figsize=(14, 7))
sns.barplot(x='Item', y='Sales Amount', hue='Promo Category', data=top_items_data)
plt.title('Sales Amount by Promotional Category for Top Items')
plt.xticks(rotation=45)
plt.legend(title='Promo Category')
plt.show()

# Sales Quantity by Promo Category
plt.figure(figsize=(14, 7))
sns.barplot(x='Item', y='Sales Quantity', hue='Promo Category', data=top_items_data)
plt.title('Sales Quantity by Promotional Category for Top Items')
plt.xticks(rotation=45)
plt.legend(title='Promo Category')
plt.show()

print(top_items_data)