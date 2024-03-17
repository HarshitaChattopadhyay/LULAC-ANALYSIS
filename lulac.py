import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('LCLU with GDP.csv')

df.head()

df.shape

df.columns

df.dtypes

df.isnull()

df.isnull().sum()

df_cleaned = df.dropna()
print(df_cleaned)

df_cleaned.isnull().any()

df_cleaned.isnull().sum()

df_cleaned.info()

df_cleaned.duplicated()

column1_contents = df['States/Union Territory']
print(column1_contents)

row_index = 2
row_contents = df.loc[row_index]
print(row_contents)

import matplotlib.pyplot as plt

# Provided data
states = ['Andhra Pradesh', 'Arunachal Pradesh', 'Assam', 'Bihar', 'Chhattisgarh', 'Goa', 'Gujarat', 'Haryana',
          'Himachal Pradesh', 'Jammu & Kashmir', 'Jharkhand', 'Karnataka', 'Kerala', 'Madhya Pradesh', 'Maharashtra',
          'Manipur', 'Meghalaya', 'Mizoram', 'Nagaland', 'Odisha', 'Punjab', 'Rajasthan', 'Sikkim', 'Tamil Nadu',
          'Telangana', 'Tripura', 'Uttar Pradesh', 'Uttarakhand', 'West Bengal', 'NCT Delhi', 'Puducherry',
          'All States and UTs']

# Create a separate subplot for each state
fig, axes = plt.subplots(len(states), 1, figsize=(10, 40), sharex=True)

# Plotting line plot for each state
for i, state in enumerate(states):
    axes[i].plot(range(1, 6), [2, 4, 1, 5, 3], marker='o', label='Some Data')  # Replace with your actual data
    axes[i].set_ylabel(state, rotation=0, ha='right')
    axes[i].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

plt.xlabel('X-axis Label')
plt.suptitle('Individual Line Plots for States', y=0.92)  # Adjust the title position
plt.tight_layout()

# Show the plot
plt.show()


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Provided data
states = ['Andhra Pradesh', 'Arunachal Pradesh', 'Assam', 'Bihar', 'Chhattisgarh', 'Goa', 'Gujarat', 'Haryana',
          'Himachal Pradesh', 'Jammu & Kashmir', 'Jharkhand', 'Karnataka', 'Kerala', 'Madhya Pradesh', 'Maharashtra',
          'Manipur', 'Meghalaya', 'Mizoram', 'Nagaland', 'Odisha', 'Punjab', 'Rajasthan', 'Sikkim', 'Tamil Nadu',
          'Telangana', 'Tripura', 'Uttar Pradesh', 'Uttarakhand', 'West Bengal', 'NCT Delhi', 'Puducherry',
          'All States and UTs']

# Create a DataFrame with random data for illustration
data = np.random.rand(len(states), len(states))

df = pd.DataFrame(data, columns=states, index=states)

# Plotting correlation matrix using seaborn
plt.figure(figsize=(15, 12))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix for States')
plt.show()


import matplotlib.pyplot as plt

# Provided data
states = ['Andhra Pradesh', 'Arunachal Pradesh', 'Assam', 'Bihar', 'Chhattisgarh', 'Goa', 'Gujarat', 'Haryana',
          'Himachal Pradesh', 'Jammu & Kashmir', 'Jharkhand', 'Karnataka', 'Kerala', 'Madhya Pradesh', 'Maharashtra',
          'Manipur', 'Meghalaya', 'Mizoram', 'Nagaland', 'Odisha', 'Punjab', 'Rajasthan', 'Sikkim', 'Tamil Nadu',
          'Telangana', 'Tripura', 'Uttar Pradesh', 'Uttarakhand', 'West Bengal', 'NCT Delhi', 'Puducherry',
          'All States and UTs']

# Create a separate subplot for each state
fig, axes = plt.subplots(len(states), 1, figsize=(10, 40), sharex=True)

# Plotting bar plot for each state
for i, state in enumerate(states):
    axes[i].bar(['Category 1', 'Category 2', 'Category 3', 'Category 4', 'Category 5'], [2, 4, 1, 5, 3], color='blue')
    axes[i].set_ylabel(state, rotation=0, ha='right')
    axes[i].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

plt.xlabel('Categories')
plt.suptitle('Individual Bar Plots for States', y=0.92)  # Adjust the title position
plt.tight_layout()

# Show the plot
plt.show()


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Provided data
states = ['Andhra Pradesh', 'Arunachal Pradesh', 'Assam', 'Bihar', 'Chhattisgarh', 'Goa', 'Gujarat', 'Haryana',
          'Himachal Pradesh', 'Jammu & Kashmir', 'Jharkhand', 'Karnataka', 'Kerala', 'Madhya Pradesh', 'Maharashtra',
          'Manipur', 'Meghalaya', 'Mizoram', 'Nagaland', 'Odisha', 'Punjab', 'Rajasthan', 'Sikkim', 'Tamil Nadu',
          'Telangana', 'Tripura', 'Uttar Pradesh', 'Uttarakhand', 'West Bengal', 'NCT Delhi', 'Puducherry',
          'All States and UTs']

# Create a DataFrame with random data for illustration
data = np.random.rand(len(states), len(states))

df = pd.DataFrame(data, columns=states, index=states)

# Plotting correlation matrix using seaborn
plt.figure(figsize=(15, 12))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix for States')
plt.show()


