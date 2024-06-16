# Using-Data-Analysis-to-build-an-effective-pricing-strategy
This project leverages advanced data analytics to develop effective pricing strategies aimed at enhancing revenue generation and market penetration in the retail sector. 

# INTRODUCTION
An effective pricing strategy involves setting the right price for a product or service that maximizes revenue while also considering market demand, competition, and customer behavior.


# SCOPE OF THE PROJECT
The project utilizes sales data with over 65000 datasets, specifically focusing on transactions related to food products. This dataset encompasses details such as sales amounts, quantities, discounts, customer information, and product attributes.

# PROGRAMMING LANGUAGE
Python will be the chosen programming language to analyze the dataset. Python is considered the best choice for analyzing data. Python can quickly create and manage data structures, allowing you to analyze and manipulate complex data sets. 

# ANALYSIS OBJECTIVES
1.Data Exploration and Preparation
2. Baseline Sales using Time Series Decomposition
3. Cannibalization
4. Promotional Effectiveness Analysis
5. Regional Analysis
6. Advanced Modelling Techniques
 - Regression Modelling
 - Customer Segmentation (K-Means Clustering)
7. Conclusion 


# 1. DATA EXPLORATION AND PREPARATION
- Loaded data from CSV file
- Examined data
- Handled missing values

![image](https://github.com/mojutoju/Using-Data-Analysis-to-build-an-effective-pricing-strategy/assets/52916369/22152479-f741-4399-94c3-f816b1d91e1d)


# 2. BASELINE SALES
Baseline sales are the expected sales of a product under normal market conditions without any promotional activities.

# How baseline sales was calculated
- Analyzed dates in the data are converted for easier handling, organizing sales figures by week.
- For each product, all the sales for each week are combined to see the total sales volume and amount on a weekly basis.
- Focus is placed on specific products with more observations in the data.

# Using a method called time series decomposition, we split the weekly sales data into three parts:
- Actual Sales: The real numbers we recorded.
- Baseline Sales: The underlying trend, showing how sales are doing over time without short-term ups and downs.
- Seasonal Patterns: Regular changes that happen at certain times of the year, like holiday boosts.
- Visualized the Observed vs Baseline Sales for the selected products

![image](https://github.com/mojutoju/Using-Data-Analysis-to-build-an-effective-pricing-strategy/assets/52916369/2b734bbc-4050-4e87-bf1a-052cc7fdd760)

# Observed vs. Baseline (Trend) Summary:
- Thresher Spicy Mints: Sales are volatile but generally follow a stable baseline, indicating predictable demand patterns.
- Even Better Whole Milk: High sales variability does not align well with a declining baseline, suggesting a gradual decrease in demand.
- Best Choice Potato Chips: Sharp peaks and valleys in sales are visible, but the flat baseline indicates consistent underlying demand.
- Fast Corn Chips: Sales closely match a slowly increasing baseline, suggesting slight growth in demand.


![image](https://github.com/mojutoju/Using-Data-Analysis-to-build-an-effective-pricing-strategy/assets/52916369/a55e9c24-f77f-478f-ad29-b32930d08ab9)

# Seasonality Summary
- Thresher Spicy Mints and Fast Corn Chips: Both show strong, regular seasonal patterns, likely influenced by specific annual events.
- Even Better Whole Milk: Exhibits minor seasonal variation, possibly due to weather-related consumption changes.
- Best Choice Potato Chips: Significant and regular seasonal peaks suggest impactful seasonal or promotional drivers.

# 3. Cannibalization Analysis
Cannibalization refers to a situation where the sales of one product or service adversely impact the sales of another existing product or service within the same category. This can occur regardless of whether the affected products are new or existing.

Cannibalization often happens when promotions, discounts, advertisements, or even the introduction of new features in one item draw potential or existing customers away from another similar product. 

Consequently, this leads to a redistribution of sales within the product lineup rather than an increase in overall revenue. Such shifts can also result from strategic changes in product positioning, pricing adjustments, or changes in consumer preferences influenced by the product offerings.

# ![image](https://github.com/mojutoju/Using-Data-Analysis-to-build-an-effective-pricing-strategy/assets/52916369/41f0f78c-b0b8-41db-9ae7-e959e7de80e1)


![image](https://github.com/mojutoju/Using-Data-Analysis-to-build-an-effective-pricing-strategy/assets/52916369/137eafca-da6e-40f0-803f-ac6c91438124)

- BBB Best Apple Preserves saw significant sales spikes, such as in March 2017 ($153,662.35) and September 2017 ($182,137.53). This is due to the promotion implemented in the product
- BBB Best Grape Jelly had much lower peaks, suggesting that promotions for Apple Preserves may have drawn customers away from Grape Jelly, leading to cannibalization within the BBB Best product line.

![image](https://github.com/mojutoju/Using-Data-Analysis-to-build-an-effective-pricing-strategy/assets/52916369/7d64121f-210e-4608-8c01-5b39ac291cd9)

- Applause Canned Mixed Fruit:
Consistent sales with peaks like May 2017 ($2,446.97) could result from targeted promotions.

- Applause Canned Peaches 
Only recorded sales in September 2017 ($1,400.50), indicating that Mixed Fruit promotions might have cannibalized Peaches sales, as customers preferred the variety promoted.

![image](https://github.com/mojutoju/Using-Data-Analysis-to-build-an-effective-pricing-strategy/assets/52916369/813a2f4c-4f20-4478-86bc-6e189755b54b)

- Golden Frozen Corn vs. Golden Frozen Broccoli:
Significant sales peaks for Golden Frozen Corn, such as in January 2017 ($125,561.23), shows effective promotions.

- Lower but notable peaks for Golden Frozen Broccoli indicate that while both products were promoted, Corn might have been more heavily advertised or discounted, leading to higher sales and cannibalization of Broccoli.


# 4. PROMOTIONAL EFFECTIVENESS ANALYSIS
Insights into how different levels of promotions affect sales and quantities sold.

![image](https://github.com/mojutoju/Using-Data-Analysis-to-build-an-effective-pricing-strategy/assets/52916369/3df4510e-9b15-4af8-90c1-02a9b92d6d12)

- High-intensity promotions lead to the highest average sales (5936.03) and transaction count (22,195), but also show high sales variability (standard deviation: 18838.26). 
- Medium promotions have the lowest sales (559.63) and quantity sold (7.52), while low promotions are in between with average sales of 1437.07. 
- Despite the variability, high promotions outperform medium and low in both sales and transactions.

# 5. REGIONAL ANALYSIS
![image](https://github.com/mojutoju/Using-Data-Analysis-to-build-an-effective-pricing-strategy/assets/52916369/2b6620cd-4afb-4e6f-9b21-5253c980663e)

"Seattle" and "San Jose" appear to be the top-performing regions, with significantly higher sales amounts compared to other regions.

# 6. ADVANCED MODELLING
Regression Modelling

**Why Regression Model?**
- Understanding Impact: Regression models are used to assess how sales are influenced by multiple factors, including promotions, discounts, and seasonal variations.
- Continuous Dependent Variable: The main variable of interest, Sales Amount, is continuous, making regression a suitable choice for prediction and analysis.
- Multiple Variables: The model handles both continuous (e.g., Discount Amount, Sales Quantity) and categorical variables (e.g., Month, DayOfWeek, Is_Promotion), allowing it to explore the effects of various predictors simultaneously.
- Predictive Capability: Regression not only provides insights into the relationships among variables but also enables forecasting of sales based on current data trends.


![image](https://github.com/mojutoju/Using-Data-Analysis-to-build-an-effective-pricing-strategy/assets/52916369/9ce72f20-c81e-48fd-9a86-5df085f2623e)

![image](https://github.com/mojutoju/Using-Data-Analysis-to-build-an-effective-pricing-strategy/assets/52916369/524281d6-c3c2-4251-8028-4538bc4450bf)

![image](https://github.com/mojutoju/Using-Data-Analysis-to-build-an-effective-pricing-strategy/assets/52916369/b62fe56a-7d4e-422e-92a4-de803ba65e7c)

# 7. Understanding Customer Purchase Behavior
RFM (Recency, Frequency, Monetary) analysis is to categorize customers based on their purchasing behaviors, which helps in understanding different customer segments and tailoring marketing strategies accordingly.

- Recency (R): Measures how recently a customer has made a purchase. A lower recency value means the customer purchased more recently, which is generally more favorable as it indicates ongoing engagement.
- Frequency (F): Counts how often a customer makes a purchase within a given time frame. Higher frequency indicates greater customer loyalty and repeated engagement.
- Monetary (M): Represents the total money spent by the customer. Higher monetary values suggest higher profitability per customer.

![image](https://github.com/mojutoju/Using-Data-Analysis-to-build-an-effective-pricing-strategy/assets/52916369/87fed4f0-c5c2-49cd-bb09-9fb3a6299ccd)

![image](https://github.com/mojutoju/Using-Data-Analysis-to-build-an-effective-pricing-strategy/assets/52916369/6d1a495a-3f3c-432e-aab9-692d862a6fc9)

# Conclusion
The analysis has effectively analyzed baseline sales, cannibalization, promotional effectiveness, utilized advanced modelling techniques like regression model and segmented customer base using RFM metrics, identifying key buying patterns and seasonal trends that influence sales performance. This insight allows us to tailor our marketing efforts and adjust our pricing strategy dynamically, enhancing customer engagement and maximizing profitability. 

Also pinpointed the effectiveness of various promotional strategies across different product lines, providing a clear direction for optimizing our discount levels to avoid cannibalization and strengthen our overall market position. Implementing these strategies will ensure we build an effective pricing strategy that adapts to customer preferences and market dynamics, driving sustained growth.

















