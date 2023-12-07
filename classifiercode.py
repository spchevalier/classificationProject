import matplotlib
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
# from sklearn.model_selection import train_test_split
# from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
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
X = df.drop(['index', 'Species', 'Species number'], axis=1)

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
plt.title("Actual species")
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
random_seed = 100
model = KMeans(n_clusters=3, random_state=random_seed)
model.fit(data_after_pca)
numerical_predictions = model.predict(data_after_pca)
# predictions = numerical_predictions.apply(lambda x: 'Adelie' if x == 0 else ('Chinstrap' if x == 1 else 'Gentoo'))
pred_num_to_species = {0: 'Gentoo', 1: 'Adelie', 2: 'Chinstrap'}
centers = model.cluster_centers_
predicted_data = data_after_pca.copy()
predicted_data['labels'] = numerical_predictions
predicted_data['labels'] = predicted_data['labels'].map(pred_num_to_species)
label_to_color_map = {'Adelie': 'blue', 'Chinstrap': 'red', 'Gentoo': 'green'}
fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection="3d"))
for unique_label in np.unique(predicted_data['labels']):
    matching_data = predicted_data.loc[predicted_data["labels"] == unique_label, ['PCA1', 'PCA2', 'PCA3']]
    ax.scatter(matching_data['PCA1'], matching_data['PCA2'], matching_data['PCA3'],
               c=label_to_color_map.get(unique_label), label=unique_label, alpha=0.5)
ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], marker='+', c='black', s=50, alpha=1)
ax.legend()
subtitle_text = 'Based on KMeans clustering'
subtitle_location = (0.5, 0.95)
ax.annotate(subtitle_text, subtitle_location, xycoords='axes fraction',
            ha='center', va='center', fontsize=12, color='gray')
ax.set_facecolor("oldlace")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
fig.set_facecolor("wheat")
plt.title("Predicted species without label data")
plt.show()
plt.clf()

# Make plot of comparison
# if correct : green dot
# if wrong : red dot
all_data = predicted_data.copy()
all_data['actual_labels'] = labeled_pca_data['labels']
fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection="3d"))
matching_data = all_data.loc[all_data["labels"] == all_data["actual_labels"], ['PCA1', 'PCA2', 'PCA3']]
non_matching_data = all_data.loc[all_data["labels"] != all_data["actual_labels"], ['PCA1', 'PCA2', 'PCA3']]
ax.scatter(matching_data['PCA1'], matching_data['PCA2'], matching_data['PCA3'],
           c='green', label='correct', alpha=0.5)
ax.scatter(non_matching_data['PCA1'], non_matching_data['PCA2'], non_matching_data['PCA3'],
           c='red', label='incorrect', alpha=0.5)
ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], marker='+', c='black', s=50, alpha=1)
ax.legend()
subtitle_text = 'Based on KMeans clustering'
subtitle_location = (0.5, 0.95)
ax.annotate(subtitle_text, subtitle_location, xycoords='axes fraction',
            ha='center', va='center', fontsize=12, color='gray')
ax.set_facecolor("oldlace")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
fig.set_facecolor("wheat")
plt.title("Prediction comparison")
plt.show()
plt.clf()

# Give here some fractions of correct vs incorrect predictions
frac_correct = len(matching_data) / len(all_data)
frac_incorrect = len(non_matching_data) / len(all_data)
print("The fraction of points correctly predicted by KMeans is " + str(frac_correct))
print("The fraction of points wrongly predicted by KMeans is " + str(frac_incorrect))

# SECOND : SUPERVISED LEARNING

X_train, X_test, y_train, y_test = train_test_split(X_numerical, y_text, test_size=0.2, random_state=random_seed)

# Do k-nearest neighbours
max_neighbours = 200
x_tick_interval = 10
accuracies = []
neighb_range = range(1, max_neighbours + 1)
for k in neighb_range:
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(X_train, y_train)
    score = classifier.score(X_test, y_test)
    accuracies.append(score)
print(accuracies)
fig, ax = plt.subplots(1, 1)
ax.plot(neighb_range, accuracies, c='darkslategrey', linewidth=1, alpha=0.5, label="numerical data")
ax.set_facecolor("ivory")
ax.set_xlabel(r'number of nearest neighbours $k$')
ax.set_ylabel(r'model accuracy')
ax.set_xticks(np.arange(0, max_neighbours + 1, x_tick_interval))
ax.set_yticks(np.arange(0, 1.1, 0.025))
plt.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
fig.set_facecolor("palegoldenrod")
plt.title(r'KMeans accuracy in terms of $k$')
# plt.show()
# plt.clf()

print("Using a KMeans model with k=2,3,...9 predicts 100% of labels correctly")

# What if we only used the principal components?
X_train, X_test, y_train, y_test = train_test_split(data_after_pca, y_text, test_size=0.2, random_state=random_seed)

# Perform k-nearest neighbours only based on the principal (numerical) component
max_neighbours = 200
x_tick_interval = 10
accuracies = []
neighb_range = range(1, max_neighbours + 1)
for k in neighb_range:
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(X_train, y_train)
    score = classifier.score(X_test, y_test)
    accuracies.append(score)
print(accuracies)
# fig, ax = plt.subplots(1, 1)
ax.plot(neighb_range, accuracies, c='darkorange', linewidth=1, label="pca data")
# ax.set_facecolor("midnightblue")
# ax.set_xlabel(r'number of nearest neighbours $k$')
# ax.set_ylabel(r'model accuracy')
# ax.set_xticks(np.arange(0, max_neighbours+1, x_tick_interval))
# ax.set_yticks(np.arange(0, 1.1, 0.1))
# plt.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
# fig.set_facecolor("honeydew")
# plt.title(r'KMeans accuracy in terms of $k$ (using only PCA)')
# plt.show()
# plt.clf()

# What if we do the same exercise based on all the data?
X_train, X_test, y_train, y_test = train_test_split(X, y_text, test_size=0.2, random_state=random_seed)

# Perform k-nearest neighbours
max_neighbours = 200
x_tick_interval = 10
accuracies = []
neighb_range = range(1, max_neighbours + 1)
for k in neighb_range:
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(X_train, y_train)
    score = classifier.score(X_test, y_test)
    accuracies.append(score)
print(accuracies)
# fig, ax = plt.subplots(1, 1)
ax.plot(neighb_range, accuracies, c='mediumseagreen', linewidth=1, label="all data")
# ax.set_facecolor("midnightblue")
# ax.set_xlabel(r'number of nearest neighbours $k$')
# ax.set_ylabel(r'model accuracy')
# ax.set_xticks(np.arange(0, max_neighbours+1, x_tick_interval))
# ax.set_yticks(np.arange(0, 1.1, 0.1))
# plt.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
# fig.set_facecolor("honeydew")
# plt.title(r'KMeans accuracy in terms of $k$')
ax.legend()
plt.show()
plt.clf()

# Decision tree
# Fit/train a decision tree classifier and analyse performance based on the test data for the three kinds of data.
data_to_use = [X_numerical, data_after_pca, X]
X_numerical.name = 'numerical data'
data_after_pca.name = 'pca data'
X.name = 'all data'
color_map = {'numerical ': 'darkslategrey', 'pca data': 'darkorange', 'all data': 'mediumseagreen'}
fig, ax = plt.subplots(1, 1)
max_depth = 11
depths = range(1, max_depth)
best_depth_all_data = 0
best_score = 0
for data in data_to_use:
    X_train, X_test, y_train, y_test = train_test_split(data, y_text, test_size=0.2, random_state=random_seed)
    accuracies_tree = []
    for depth in depths:
        classifier = DecisionTreeClassifier(criterion="gini", max_depth=depth, random_state=random_seed)
        classifier.fit(X_train, y_train)
        score = classifier.score(X_test, y_test)
        if (data.name == 'all data') & (score > best_score):
            best_score = score
            best_depth_all_data = depth
        accuracies_tree.append(score)
    ax.plot(depths, accuracies_tree, c=color_map.get(data.name), label=str(data.name), linewidth=1, alpha=0.8)
ax.set_xlabel(r'depth $d$')
ax.set_ylabel(r'accuracy')
ax.set_xticks(depths)
ax.set_facecolor("honeydew")
ax.legend()
fig.set_facecolor("darkseagreen")
plt.title(r'Decision tree accuracy in terms of depth $d$')
plt.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
plt.show()
plt.clf()

# Print best decision tree
X_train, X_test, y_train, y_test = train_test_split(X, y_text, test_size=0.2, random_state=random_seed)

plt.figure(figsize=(14, 8))
dt = DecisionTreeClassifier(random_state=100, max_depth=best_depth_all_data)
dt.fit(X_train, y_train)
tree.plot_tree(dt, feature_names=X_train.columns,
               class_names=['Adelie', 'Chinstrap', 'Gentoo'],
               filled=True)
plt.show()

# Possible outlooks
# Try sequential forward selection (SFS)
