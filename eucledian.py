import math

# Assuming test and first_mean are 1D NumPy arrays or Python lists
def euclidean_distance(point1, point2):
    if len(point1) != len(point2):
        raise ValueError("Vectors must have the same length")
    
    squared_differences = [(a - b) ** 2 for a, b in zip(point1, point2)]
    distance = math.sqrt(sum(squared_differences))
    return distance

# Example usage
test = [1, 2, 3, 4]
first_mean = [5, 6, 7, 8]
distance = euclidean_distance(test, first_mean)
print(f"Euclidean Distance: {distance}")