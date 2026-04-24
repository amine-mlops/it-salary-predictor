import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io 

df = pd.read_csv("salary_dataset.csv")

sns.set_theme(style="whitegrid")
sns.boxplot(data=df, x='Certifications', y='Salary')
plt.xlabel('Certifications')
plt.ylabel('Salary (in k MAD)')
plt.title("Salary Distribution by Number of Certifications")
plt.savefig('Salary by Certification Count.png', dpi=300)