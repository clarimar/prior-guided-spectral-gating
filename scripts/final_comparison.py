
"""
Tabela final de comparação para o paper
"""

import numpy as np
import pandas as pd

results = {
    'Method': [
        'PLS (10 components)',
        'BandSelectNet + ANOVA',
        'BandSelectNet + VIP',
        'BandSelectNet + RF',
        'BandSelectNet (random)',
        'Plain CNN'
    ],
    'Parameters': [10, 1765, 1765, 1765, 1765, 109221],
    'Test_RMSE': [0.2957, 0.3607, '?', '?', '?', 0.9094],
    'Test_R2': [0.9191, 0.8617, '?', '?', '?', 0.2348],
    'CV_std': ['-', 0.0363, '?', '?', '?', '-'],
    'Interpretable': ['VIP only', 'Yes', 'Yes', 'Yes', 'No', 'No']
}

df = pd.DataFrame(results)
print("\n" + "="*80)
print("FINAL COMPARISON TABLE")
print("="*80)
print(df.to_string(index=False))
print("\n" + "="*80)

# Save
df.to_csv('results/tecator/final_comparison.csv', index=False)
print("✅ Saved: results/tecator/final_comparison.csv")



