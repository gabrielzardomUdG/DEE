import math
import pandas as pd
import numpy as np
import handmade_utils.utils as hmu
import matplotlib.pyplot as plt
from scipy.stats import norm

# Load data
df = pd.read_csv('source/data_science_student_marks.csv')
data = df['english_marks'].to_numpy()

# Find values
hmu.print_centered("Values",80,"*")
global_min_value : float = np.min(data)
global_max_value : float = np.max(data)
average : float = np.mean(data)
standard_deviation : float = np.std(data)
agustin_min_proportion_low_value : float = average - standard_deviation
agustin_max_proportion_low_value : float = average + standard_deviation
agustin_lowZ_first_form: float = (agustin_min_proportion_low_value - average)/standard_deviation
agustin_highZ_first_form: float = (agustin_max_proportion_low_value - average)/standard_deviation
agustin_min_proportion : float = 1 - ((1/2) ** standard_deviation)
agustin_min_proportion : float = 1 - ((1/2) ** standard_deviation)

print("Statistical Summary")
print("-" * 40)
print(f"Minimum value in array: {global_min_value:.2f}")
print(f"Maximum value in array: {global_max_value:.2f}")
print(f"Average value: {average:.2f}")
print(f"Standard deviation: {standard_deviation:.2f}")
print()

print("Agustin's Proportion Details")
print("-" * 40)
print(f"Min proportion range: {agustin_lowZ_first_form:.2f} to {agustin_highZ_first_form:.2f}")
print(f"Agustin minimum proportion: {agustin_min_proportion:.2f}")

p = 0.7
log_term = math.log(1 / (1 - p)) / math.log(2)
low_range = average - (standard_deviation * log_term)
high_range = average + (standard_deviation * log_term)

print(f"Agustin proportion at 70%: {low_range:.2f} to {high_range:.2f}")
# Curva normal teórica
x = np.linspace(global_min_value, global_max_value, 100)
y = norm.pdf(x, average, standard_deviation)

# Visualización
plt.figure(figsize=(8,5))
plt.hist(data, bins=10, density=True, alpha=0.6, color='lightblue', edgecolor='black', label='Observed Data')
plt.plot(x, y, 'r-', linewidth=2, label='Normal Curve')
plt.title('Score Distribution (English Marks)')
plt.xlabel('Score')
plt.ylabel('Frequency')
plt.legend()
plt.show()
