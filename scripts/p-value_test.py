from statsmodels.formula.api import ols
import statsmodels.api as sm
import pandas as pd
from scipy import stats

df1 = pd.read_csv(
    '/Volumes/SSD/Mestrado/Dissertacao/resources/bpca_unet_500_miou/bpca_unet_miou.csv'
)
df2 = pd.read_csv(
    '/Volumes/SSD/Mestrado/Dissertacao/resources/max_unet_500_miou/max_unet_miou.csv'
)

for col in df1.columns:
    print(col)
    print(stats.ttest_ind(df1[col], df2[col]))

df1['model'] = 'bpca_unet'
df2['model'] = 'max_unet'

df = pd.concat([df1, df2])

mod = ols('val_mean_iou ~ model', data=df).fit()
aov_table = sm.stats.anova_lm(mod, typ=2)
print(aov_table)

# perform Kolmogorov-Smirnov test
print(stats.kstest(
    df1['val_mean_iou'],
    df2['val_mean_iou']
))
