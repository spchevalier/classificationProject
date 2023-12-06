import matplotlib
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
# from sklearn.model_selection import train_test_split
# from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('/Users/sebastien/PycharmProjects/classificationProject/penguins_lter.csv')

# Step 0.1 : INSPECT DATA

# Inspect data by printing head and summarizing all the columns
print(df.head())
print(df.info())

# Step 0.2 : CLEAN DATA

# Drop the study name, it's only an identifier, not a feature
# Drop sample number
df = df.drop(['studyName', 'Sample Number'], axis=1)

# There is no variation in the values of the region
# print(df['Region'].value_counts())
# so remove this feature
df = df.drop(['Region'], axis=1)

# Perform one-hot encoding for the island feature
dummies = pd.get_dummies(df['Island'])
df = df.join(dummies)
df = df.drop(['Island'], axis=1)

# Drop stage feature because the value is constant (no contribution to variance)
# print(df['Stage'].value_counts())
df = df.drop(['Stage'], axis=1)

# Drop the individual ID column
# print(df['Individual ID'].value_counts())
df = df.drop(['Individual ID'], axis=1)

# One-hot encode the clutch completion feature
# print(df['Clutch Completion'].unique().tolist())
clutch_dict = {'Yes': 1, 'No': 0}
df['Clutch Completion'] = df['Clutch Completion'].map(clutch_dict)

# Encode the date egg feature as date-time variables
df['Date Egg'] = pd.to_datetime(df['Date Egg'], format="mixed")
df['Year Egg'] = df['Date Egg'].dt.year
df['Month Egg'] = df['Date Egg'].dt.month
df['Day Egg'] = df['Date Egg'].dt.day
df = df.drop(['Date Egg'], axis=1)

# Remark : There are no original categorical features have that a large number of unique values
# such that binary or hash encoding is required.

# Make sex feature numerical
sex_dict = {'MALE': 1, 'FEMALE': 0}
df['Sex'] = df['Sex'].map(sex_dict)

# Remove comments and delta feature
df = df.drop(['Comments'], axis=1)
df = df.drop(['Delta 15 N (o/oo)'], axis=1)
df = df.drop(['Delta 13 C (o/oo)'], axis=1)

# Standardise the numerical data
# Rename columns that are too long
df.rename(columns={
    'Culmen Length (mm)': 'Culmen Length',
    'Culmen Depth (mm)': 'Culmen Depth',
    'Flipper Length (mm)': 'Flipper Length',
    'Body Mass (g)': 'Body Mass'
}, inplace=True)

# Before standardisation : treat missing data (NaN)
df = df.dropna()
df.reset_index(inplace=True)

# Standardise all numerical features
scaler = StandardScaler()
numerical_data_columns = ['Culmen Length', 'Culmen Depth', 'Flipper Length', 'Body Mass']
df[numerical_data_columns] = scaler.fit_transform(df[numerical_data_columns])

# Test the normalisation
for column in numerical_data_columns:
    print("Normalised mean of " + str(column) + "is : " + str(np.mean(df[column])))
    print("Normalised standard deviation of " + str(column) + " is : " + str(np.std(df[column])))
    print("Max of " + str(column) + " is : " + str(np.max(df[column])))
    print("Min of " + str(column) + " is : " + str(np.min(df[column])))

# Shorten name of labels and transform target variable (labels) to numerical data
names_dict = {'Adelie Penguin (Pygoscelis adeliae)': 'Adelie',
              'Chinstrap penguin (Pygoscelis antarctica)': 'Chinstrap',
              'Gentoo penguin (Pygoscelis papua)': 'Gentoo'}
names_number_dict = {'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2}
df['Species'] = df['Species'].map(names_dict)
df['Species number'] = df['Species'].map(names_number_dict)


# Inspect the data again after cleanup
print(df.head())
print(df.info())

# Split the data in features and labels
y_text = df['Species']
y = df['Species number']
X = df.drop(['Species', 'Species number'], axis=1)

# Isolate the numerical data
X_numerical = X[numerical_data_columns]

# Inspect the correlations between the features by printing a heatmap of the correlation matrix
correlation_matrix = X.corr()
ax = plt.axes()
sns.heatmap(correlation_matrix, vmin=-1, vmax=1, cmap='Greens', ax=ax)
ax.set_title('Correlation matrix heatmap')
plt.show()

# From the correlation matrix we can see that there is a stronger correlation between flipper length and
# body mass.


# FIRST : UNSUPERVISED FINDINGS (Being completely unaware about the true labels/classification,
# how many groups would you say there are?)

# Perform principal component analysis (PCA) for dimensional reduction.
# Because there are 4 numerical features (4-dimensional feature space),
# the number of eigenvectors/components will also be 4.
# The corresponding eigenvalues are the fractions of the variance of the data
# that is explained by / contained in the component.
pca = PCA()
pca.fit(X_numerical)
components = pca.components_
components_df = pd.DataFrame(components).transpose()
components_df.index = components_df.columns
print("The components are : " + str(components_df))

eigvals = pca.explained_variance_ratio_
cum_ratio = np.cumsum(eigvals)
var_ratio = pd.DataFrame(eigvals).transpose()
print("The eigenvalues are : " + str(var_ratio))

fig, axs = plt.subplots(1, 2)
axs[0].plot(np.arange(1, len(eigvals) + 1), eigvals,
            marker='o', markeredgecolor='darkslategrey', markerfacecolor='teal', color='black')
axs[0].set_title('Information explanation')
axs[0].set_xlabel('Principal Axes / Components')
axs[0].set_ylabel('Information explained')
axs[0].set_facecolor('azure')
axs[1].plot(np.arange(1, len(eigvals) + 1), cum_ratio,
            marker='D', markeredgecolor='black', markerfacecolor='azure', color='white')
axs[1].set_title('Cumulative information')
axs[1].set_xlabel('Principal Axes / Components')
axs[1].set_facecolor('teal')
axs[1].set_xticks(np.arange(1, len(eigvals) + 1))
axs[1].hlines(y=0.95, xmin=1, xmax=4, linestyle='--', color='lavender')
fig.set_facecolor("whitesmoke")
plt.show()
plt.clf()

# From the plot of cumulative information gain we see that three components already
# explain more than 95% of the variance. This allows for dropping 1 PCA, in this case also lowering the dimensionality
# of the feature space with 1.

# Try to perform dimensional reduction by only keeping a few principal components.
pca_restricted = PCA(n_components=3)
data_after_pca = pd.DataFrame(pca_restricted.fit_transform(X_numerical))
data_after_pca.columns = ['PCA1', 'PCA2', 'PCA3']
labeled_pca_data = data_after_pca.copy()
labeled_pca_data['labels'] = y_text
label_to_color_map = {'Adelie': 'blue', 'Chinstrap': 'red', 'Gentoo': 'green'}
fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection="3d"))
for unique_label in np.unique(labeled_pca_data['labels']):
    matching_data = labeled_pca_data.loc[labeled_pca_data["labels"] == unique_label, ['PCA1', 'PCA2', 'PCA3']]
    ax.scatter(matching_data['PCA1'], matching_data['PCA2'], matching_data['PCA3'],
               c=label_to_color_map.get(unique_label), label=unique_label, alpha=0.5)
ax.legend()
ax.set_facecolor("oldlace")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
fig.set_facecolor("wheat")
plt.title("Data in terms of the principal components")
plt.show()
plt.clf()

# Approximate the minimum number of clusters with sufficiently low inertia
inertias = []
max_groups = 11
num_groups = list(range(1, max_groups))
for k in num_groups:
    model = KMeans(n_clusters=k)
    model.fit(X, y_text)
    inertias.append(model.inertia_)

fig, ax = plt.subplots(1, 1)
ax.plot(num_groups, inertias, color='black',
        marker='s', markerfacecolor='darkmagenta', markeredgecolor='indigo')
ax.set_ylabel(r'Inertia $L$')
ax.set_xlabel(r'Number of groups $N$')
ax.set_xticks(np.arange(1, max_groups))
ax.set_facecolor('whitesmoke')
fig.set_facecolor('thistle')
plt.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
plt.title('Cluster determination')
plt.show()

# From the graph, using the elbow method, one can say that three clusters is a reasonable estimation of the
# number of groups.

# Perform KMeans algorithm for 3 clusters
model = KMeans(n_clusters=3)
model.fit(data_after_pca)
predictions = model.predict(data_after_pca)
pred_num_to_species = {0:'Adelie', 1:'Chinstrap', 2:'Gentoo'}
plot_data_with_predicted_labels = data_after_pca.copy()
plot_data_with_predicted_labels['predicted_labels'] = predictions
centers = model.cluster_centers_
print(centers)

fig, axs = plt.subplots(1, 1)
colors = ['b', 'r', 'g']
cmap_peng = matplotlib.colors.ListedColormap(colors)
numerical_label_to_label = {0: 'Adelie', 1: 'Gentoo', 2: 'XXX'}
axs.set_xlabel("PCA1")
axs.set_ylabel("PCA2")
axs.scatter(plot_data_with_predicted_labels['PCA1'], plot_data_with_predicted_labels['PCA2'],
            c=plot_data_with_predicted_labels['predicted_labels'], cmap=cmap_peng,
            alpha=0.5)
axs.scatter(centers[:, 0], centers[:, 1], marker='+', color='black')
axs.legend(labels=plot_data_with_predicted_labels['predicted_labels'])
axs.set_facecolor('gainsboro')
plt.grid(True, linestyle='--', linewidth=0.5, color='white', alpha=0.5)
plt.show()

# remark : print the centers of each cluster (after convergence)

# Do this clustering based on the principal components
model_pca = KMeans(n_clusters=3)
model_pca.fit(data_after_pca)

# Perform K-neighrest neighbours


# centroids_x = np.random.uniform(min(x),max(x),k)


# Look whether there might be other groups?


# SECOND : SUPERVISED LEARNING

# Compare this with the actual labels

# Step 3 : Feature selection
