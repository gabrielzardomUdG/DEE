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
min_value = np.min(data)
max_value = np.max(data)
average = np.mean(data)
standard_deviation = np.std(data)

print(f"Min_value: {min_value}")
print(f"Max_value: {max_value}")
print(f"Theoretic_Normal: {average:.2f}")
print(f"Standard_Deviation: {standard_deviation:.2f}")

# Curva normal teórica
x = np.linspace(min_value, max_value, 100)
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
