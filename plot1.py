import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io 

df = pd.read_csv("salary_dataset.csv")
sns.set_theme(style="whitegrid")
sns.regplot(data=df, x='Experience', y='Salary', scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
plt.xlabel('Experience (Years)')
plt.ylabel('Salary (in k MAD)')
plt.title('Impact of Experience on Salary')
plt.savefig('salaire_vs_experience.png', dpi=300)