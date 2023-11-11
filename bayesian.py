import numpy as np
import matplotlib.pyplot as plt
import theano . tensor as tens
from scipy import stats
import pymc3



# Function that computes the Euclidean distance equation using the
# numpy library
def eucledian(x,y):
    return  np.sqrt((( x - y ) **2).sum (axis=1))
############################### Make a map for warehouse #######################################

# number of measurement of each AP 
measurement = 211


# The dimensions of the warehouse
L = 8 # Length
W = 7 # Wide

# AP Position 
access_point_coordinates = {
    "Wifi_A": [0, 7],
    "Wifi_B": [0, 0],
    "Wifi_C": [8, 7],
    "Wifi_D": [8, 0]  # I assume you meant Wifi_D at [8, 0] instead of [8, 7]
}
# Create the anchor_pos array with access point coordinates
anchor_pos = np.array([access_point_coordinates["Wifi_A"], 
                       access_point_coordinates["Wifi_B"], 
                       access_point_coordinates["Wifi_C"], 
                       access_point_coordinates["Wifi_D"]])
num_AP = len(anchor_pos)

# Plot the access point positions
plt.figure(figsize=(8, 7))
plt.scatter(anchor_pos[:, 0], anchor_pos[:, 1], marker='o', label="Access Points")
plt.xlim(-1, L + 1)
plt.ylim(-1, W + 1)
plt.xlabel("X-coordinate")
plt.ylabel("Y-coordinate")
plt.title("Access Point Positions")
plt.legend()
plt.grid(True)
#target position 
target_pos = np.array ([1, 1])

distance = np.zeros((num_AP,1))

for i in range(0, num_AP):
    distance[i] = eucledian(target_pos, anchor_pos[i].reshape(1,-1))
    print(f"Distance from test point to Wifi_{chr(ord('A') + i)}: {distance[i][0]:.2f}")
plt.scatter(target_pos[0], target_pos[1], marker='x', color='red', label="Test Point")
plt.legend()
plt.show()





# # Plot the access point positions
# plt.figure(figsize=(8, 7))
# plt.scatter(anchor_pos[:, 0], anchor_pos[:, 1], marker='o', label="Access Points")
# plt.xlim(-1, L + 1)
# plt.ylim(-1, W + 1)
# plt.xlabel("X-coordinate")
# plt.ylabel("Y-coordinate")
# plt.title("Access Point Positions")
# plt.legend()
# plt.grid(True)
# plt.show()
