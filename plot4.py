import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io 

df = pd.read_csv("salary_dataset.csv")

sns.scatterplot(data=df, x='ProjectsCompleted', y='Salary', hue='Certifications', palette='viridis', s=100)
plt.title('Salary vs Projects Completed')
plt.xlabel('Projects Completed')
plt.ylabel('Salary (in k MAD)')
plt.savefig('Salary vs Projects Completed.png', dpi=300)