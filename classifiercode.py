import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import sklearn.preprocessing
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('/Users/sebastien/PycharmProjects/classificationProject/penguins_lter.csv')

# Step 1 : inspect data

# Inspect data by printing head
print(df.head())

# Inspect numerical and categorical data
print(df.info())

# Step 2 : clean data, make classification data numeric
# drop na values

# Drop the study name, it's only an identifier, not a feature
# Drop sample number
df = df.drop(['studyName', 'Sample Number'], axis=1)


# Transform target variable (labels) to numerical data
species_list = df['Species'].unique().tolist()
df['Species'] = np.array([species_list.index(species) for species in df['Species']])

# There is no variation in the values of the region
# print(df['Region'].value_counts())
# so remove this feature
df = df.drop(['Region'], axis=1)

# Perform one-hot encoding for the island feature
dummies = pd.get_dummies(df['Island'])
df = df.join(dummies)
df = df.drop(['Island'], axis=1)

# Stage
# print(df['Stage'].value_counts())
# Drop stage feature because the value is constant (no contribution to variance)
df = df.drop(['Stage'], axis=1)

# What to do with the ID's? (drop?)
# print(df['Individual ID'].value_counts())
df = df.drop(['Individual ID'], axis=1)

# print(df['Clutch Completion'].unique().tolist())
clutch_dict = {'Yes': 1, 'No': 0}
df['Clutch Completion'] = df['Clutch Completion'].map(clutch_dict)
# print(df['Clutch Completion'].unique().tolist())

# Encode date-time variables
df['Date Egg'] = pd.to_datetime(df['Date Egg'], format="mixed")
df['Year Egg'] = df['Date Egg'].dt.year
df['Month Egg'] = df['Date Egg'].dt.month
df['Day Egg'] = df['Date Egg'].dt.day
df = df.drop(['Date Egg'], axis=1)

# No original categorical features have that many unique values
# that binary or hash encoding is necessary for that feature.

# Make sex feature numerical
sex_dict = {'MALE': 1, 'FEMALE': 0}
df['Sex'] = df['Sex'].map(sex_dict)

# Remove comments feature
df = df.drop(['Comments'], axis=1)
df = df.drop(['Delta 15 N (o/oo)'], axis=1)
df = df.drop(['Delta 13 C (o/oo)'], axis=1)

# Standardise the numerical data
# rename columns that are too long
df.rename(columns={
    'Culmen Length (mm)': 'Culmen Length',
    'Culmen Depth (mm)': 'Culmen Depth',
    'Flipper Length (mm)': 'Flipper Length',
    'Body Mass (g)': 'Body Mass'
}, inplace=True)

# Before standardisation : treat missing data (NaN)
df.dropna(inplace=True)

# Standardise all numerical features
scaler = StandardScaler()
columns_to_standard_scale = ['Culmen Length', 'Culmen Depth', 'Flipper Length', 'Body Mass']
df[columns_to_standard_scale] = scaler.fit_transform(df[columns_to_standard_scale])

# Test the normalisation
print('The mean is correctly : '+str(np.mean(df['Culmen Length'])))
print('The std is correctly : '+str(np.std(df['Flipper Length'])))


print(df.head())
print(df.info())

# Step 3 : inspect correlation matrix, throw away highly correlated features

# Step : split the data in features and labels
y = df['Species']
X = df.drop(['Species'], axis=1)

correlation_matrix = X.corr()

# print the heat map of the correlation matrix
ax = plt.axes()
sns.heatmap(correlation_matrix, cmap='Greens', ax=ax)
ax.set_title('Correlation matrix heatmap')
plt.show()
