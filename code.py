
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

df = pd.read_csv('/content/messy_sales_data.csv')
print("Initial Dataset Shape:", df.shape)

df = df.drop_duplicates()
print("Shape after dropping duplicates:", df.shape)

df = df[df['Quantity'] > 0]
print("Shape after keeping positive quantities:", df.shape)

def clean_price(val):
    if pd.isna(val):
        return np.nan
    val_str = str(val).replace('Rs.', '').replace('$', '').replace(',', '').strip()
    try:
        return float(val_str)
    except:
        return np.nan

df['Unit Price'] = df['Unit Price'].apply(clean_price)

def clean_age(val):
    if pd.isna(val):
        return np.nan
    val_str = re.sub(r'\D', '', str(val))
    if val_str == '':
        return np.nan
    return float(val_str)

df['Age'] = df['Age'].apply(clean_age)

df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')

df['Unit Price'] = df['Unit Price'].fillna(df['Unit Price'].median())
df['Age'] = df['Age'].fillna(df['Age'].median())

df['Gender'] = df['Gender'].fillna('Unknown')
df['Status'] = df['Status'].fillna('Unknown')
df['Rating'] = df['Rating'].fillna('No Rating')

df['Department'] = df['Department'].astype(str).str.strip().str.title()

df['Department'] = df['Department'].replace({
    'Hr': 'HR',
    'It': 'IT',
    'Marketng': 'Marketing'
})

print("Final Cleaned Dataset Shape:", df.shape)

plt.figure(figsize=(8, 5))
sns.histplot(df['Quantity'], bins=10, kde=True, color='blue')
plt.title('Histogram of Order Quantity', fontsize=14, fontweight='bold')
plt.xlabel('Quantity')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 5))
sns.barplot(data=df, x='Department', y='Quantity', errorbar=None, palette='viridis', hue='Department', legend=False)
plt.title('Average Quantity per Department', fontsize=14, fontweight='bold')
plt.xlabel('Department')
plt.ylabel('Average Quantity')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(7, 7))
clean_counts = df['Department'].value_counts()
plt.pie(clean_counts, labels=clean_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
plt.title('Cleaned Proportion of Departments', fontsize=14, fontweight='bold')
plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='Order ID', y='Quantity', alpha=0.7, color='purple')
plt.title('Scatter Plot: Order ID vs Quantity', fontsize=14, fontweight='bold')
plt.xlabel('Order ID')
plt.ylabel('Quantity')
plt.show()

plt.figure(figsize=(10, 5))
df_sorted = df.dropna(subset=['Order Date']).sort_values('Order Date')
sns.lineplot(data=df_sorted, x='Order Date', y='Quantity', color='green', linewidth=1.5)
plt.title('Trend of Quantity Over Time', fontsize=14, fontweight='bold')
plt.xlabel('Order Date')
plt.ylabel('Quantity')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(y=df['Quantity'], color='orange')
plt.title('Box Plot of Quantity (Outlier Detection)', fontsize=14, fontweight='bold')
plt.ylabel('Quantity')
plt.show()

plt.figure(figsize=(8, 6))
numeric_df = df.select_dtypes(include=['number'])
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap', fontsize=14, fontweight='bold')
plt.show()

plt.figure(figsize=(9, 5))
sns.countplot(data=df, x='Department', order=df['Department'].value_counts().index, palette='Set2', hue='Department', legend=False)
plt.title('Count of Records per Department', fontsize=14, fontweight='bold')
plt.xlabel('Department')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(9, 5))
sns.violinplot(data=df, x='Department', y='Quantity', palette='muted', hue='Department', legend=False)
plt.title('Violin Plot of Quantity by Department', fontsize=14, fontweight='bold')
plt.xlabel('Department')
plt.ylabel('Quantity')
plt.show()
