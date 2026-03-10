import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data_path = "Dataset (1).csv"
df = pd.read_csv(data_path, encoding='latin-1')

# 1. Geographical Exploration (Lat/Long Distribution)
plt.figure(figsize=(12, 8))
sns.scatterplot(x='Longitude', y='Latitude', data=df, hue='Country Code', palette='viridis', alpha=0.5)
plt.title('Geographical Distribution of Restaurants')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.savefig('restaurant_distribution.png')
print("Restaurant distribution map saved as restaurant_distribution.png")

# Handle potential BOM in column names
df.columns = df.columns.str.replace('ï»¿', '')

# 2. City and Locality Analysis (Concentration)
city_counts = df['City'].value_counts().nlargest(10)
plt.figure(figsize=(12, 6))
sns.barplot(x=city_counts.values, y=city_counts.index, hue=city_counts.index, palette='magma', legend=False)
plt.title('Top 10 Cities by Number of Restaurants')
plt.xlabel('Number of Restaurants')
plt.ylabel('City')
plt.savefig('city_concentration.png')
print("City concentration plot saved as city_concentration.png")

# 3. Statistical Analysis by City
# Average rating and price range by city
city_stats = df.groupby('City').agg({
    'Aggregate rating': 'mean',
    'Price range': 'mean',
    'Restaurant ID': 'count'
}).rename(columns={'Restaurant ID': 'Restaurant Count'})

# Sort by rating to find interesting patterns (cities with > 10 restaurants)
high_rated_cities = city_stats[city_stats['Restaurant Count'] > 10].sort_values(by='Aggregate rating', ascending=False).head(10)

print("\nTop 10 Cities with Highest Average Ratings (>10 restaurants):")
print(high_rated_cities[['Aggregate rating', 'Price range']])

# 4. Insights Generation
print("\n--- Insights ---")
# Finding relationship between price and rating by city
correlation = df[['Price range', 'Aggregate rating']].corr().iloc[0, 1]
print(f"Correlation between Price Range and Aggregate Rating: {correlation:.4f}")

# Identifying areas with high price but low rating or vice versa
print("\nCities with high average price range and their ratings:")
print(city_stats.sort_values(by='Price range', ascending=False).head(5))
