import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io 

df = pd.read_csv("salary_dataset.csv")
plt.figure(figsize=(11, 9))
sns.set_theme(style="whitegrid")
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Feature Correlation Matrix')
plt.savefig('Feature Correlation Matrix.png', dpi=300)