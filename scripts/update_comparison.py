import pandas as pd

results = {
    'Method': [
        'PLS (10 components)',
        'BandSelectNet + ANOVA (CV)',
        'BandSelectNet + ANOVA (test)',
        'BandSelectNet + VIP',
        'BandSelectNet + RF',
        'BandSelectNet (random)',
        'Plain CNN'
    ],
    'Parameters': [10, 1765, 1765, 1765, 1765, 1765, 109221],
    'Test_RMSE': [0.2957, '0.36±0.04', 0.2536, 0.3234, 0.2583, 0.3732, 0.9094],
    'Test_R2': [0.9191, '0.86±0.04', 0.9405, 0.9032, 0.9383, 0.8711, 0.2348],
    'Evaluation': ['Single', '5-fold CV', 'Single', 'Single', 'Single', 'Single', 'Single'],
    'Interpretable': ['VIP only', 'Yes', 'Yes', 'Yes', 'Yes', 'No', 'No']
}

df = pd.DataFrame(results)
print("\n" + "="*90)
print("FINAL COMPARISON TABLE - ALL METHODS")
print("="*90)
print(df.to_string(index=False))
print("\n" + "="*90)

# Save
df.to_csv('results/tecator/final_comparison_complete.csv', index=False)
print("✅ Saved: results/tecator/final_comparison_complete.csv")

