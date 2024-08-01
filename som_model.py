# %%
# Import python packages:
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom
from pylab import bone, pcolor, colorbar, plot, show

# Import dataset:
df = pd.read_csv(r'Credit_Card_Applications.csv')

# %%
# Check first 5 rows of data:
df.head()

# %% [markdown]
# Feature Scaling

# %%
# Split data to X and Y variables:
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Instaniate MinxMax Scaler:
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)

# %% [markdown]
# Training SOM

# %%
# Training the SOM:
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)

# Initalize weights for data (Step 3):
som.random_weights_init(X)

# Train Self Organizing Map Model: Set to 100 iterations
som.train_random(X, num_iteration=100)

# %% [markdown]
# Visualize Results: Mean Interneuron Distance (MID)
# * The Mean Interneuron Distance (MID) of a specific winning node is the average distance of all the neurons surrounding the winning node within a neighborhood defined by our sigma parameter. The sigma parameter represents the radius of this neighborhood. A higher MID indicates that the winning node is farther away from its neighbors within the neighborhood. Essentially, the higher the MID, the more likely the winning node is an outlier. This is how we will determine potential fraud: nodes with a high MID are flagged as possible fraudulent activities.

# %%
bone()
pcolor(som.distance_map().T)
colorbar()
# Add markers

# Create vector of markers:
markers = ['o', 's']

# Add colors for symbols
colors = ['r', 'g']

# Loop through all customers
for i, x in enumerate(X):
    # Get winning node
    w = som.winner(x)
    plot(
        w[0] + 0.5,
        w[1] + 0.5,
        markers[y[i]],
        markeredgecolor = colors[y[i]],
        markerfacecolor = 'None',
        markersize = 12,
        markeredgewidth = 3)

show()


# %% [markdown]
# Finding Customers Flagged As Fraud

# %%
# Get the mappings of the input data to their winning nodes in the SOM
mappings = som.win_map(X)

frauds = np.concatenate((mappings[(9,7)], mappings[(7,8)]), axis = 0)
# Inverse scaling
frauds = sc.inverse_transform(frauds)

# Convert the frauds array into a DataFrame
fraud_df = pd.DataFrame(frauds, columns=df.columns[:-1])  # Use appropriate column names from your dataset

# Display the DataFrame
fraud_df['CustomerID']


# %%


# %%



