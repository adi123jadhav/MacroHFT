import numpy as np

# Load data from a .npy file
data = np.load(r'C:\Users\KIIT\Downloads\MacroHFT\data\feature_list\trend_features.npy')

# Now 'data' contains the array that was saved in the file
print(data)

# Print the shape of the array
print(data.shape)

# Print the first few elements of the array
