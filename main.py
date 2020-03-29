"""
Credits:
https://machinelearningmastery.com/how-to-code-the-students-t-test-from-scratch-in-python/
https://pythonfordatascience.org/paired-samples-t-test-python/
https://reneshbedre.github.io/blog/anova.html
https://machinelearningmastery.com/how-to-use-correlation-to-understand-the-relationship-between-variables/
https://pythonfordatascience.org/anova-python/

Fotios Lygerakis
PhD Student
Statistics Assignment
CSE-6963 "Advanced Topics in Human Robot Interaction"
University of Texas at Arlington
"""
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, ttest_rel, stats, pearsonr, spearmanr, kendalltau
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# choose part [1, 2, 3]
part = 3

if part == 1:
    # reading csv file
    data = pd.read_csv("Part1.csv", header=None)

    col1 = data[0].values
    col2 = data[1].values

    data[0].plot(kind='hist', title='histogram of 0')
    data[1].plot(kind='hist', title='Distribution of data')
    plt.savefig('distr.png')

    data[[0, 1]].plot(kind='box')
    plt.savefig('boxplot_outliers.png')
    # compare samples
    # Null hypothesis is that samples are the same
    stat, p = ttest_ind(col1, col2, equal_var=False)
    print('T-Test individual: t=%.3f, p=%.3f' % (stat, p))

    stat, p = ttest_rel(col1, col2)
    print('Paired T-Test: t=%.3f, p=%.3f' % (stat, p))

    # calculate Pearson's correlation
    corr, _ = pearsonr(col1, col2)
    print('Pearsons correlation: %.3f' % corr)

    corr, _ = spearmanr(col1, col2)
    print('Spearmans p: %.3f' % corr)

    corr, _ = kendalltau(col1, col2)
    print('Kendall\'s tau: %.3f' % corr)

elif part == 2:
    # reading csv file
    data = pd.read_csv("Part2.csv")

    # stats f_oneway functions takes the groups as input and returns F and P-value
    fvalue, pvalue = stats.f_oneway(data['Before'], data['After'])
    print(fvalue, pvalue)

    # Create a boxplot
    # data[["Before", "After"]].plot(kind='box')
    # plt.savefig('part2_box.png')

    data["Before"].plot(kind='hist', title='histogram of 0')
    data["After"].plot(kind='hist', title='Distribution of data')
    plt.savefig('part2_distr.png')

    stat, p = ttest_ind(data["Before"], data["After"], equal_var=False)
    print('T-Test individual: t=%.3f, p=%.3f' % (stat, p))
    print("Mean Before %.2f(%.2f)" % (np.mean(data["Before"]), np.std(data["Before"])))
    print("Mean After %.2f(%.2f)" % (np.mean(data["After"]), np.std(data["After"])))

elif part == 3:
    # reading csv file
    data = pd.read_csv("Part3.csv")
    print(data)
    print(stats.f_oneway(data["Family 1"], data["Family 2"], data["Family 3"], data["Family 4"]))

    # data.boxplot(column=['Family 1', 'Family 2', 'Family 3', 'Family 4'], grid=False)
    # plt.savefig('part3_box.png')
    # reshape the d dataframe suitable for statsmodels package
    d_melt = pd.melt(data.reset_index(), id_vars=['index'], value_vars=['Family 1', 'Family 2', 'Family 3', 'Family 4'])
    # replace column names
    d_melt.columns = ['index', 'treatments', 'value']
    # Ordinary Least Squares (OLS) model
    model = ols('value ~ C(treatments)', data=d_melt).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print(anova_table)

    m_comp = pairwise_tukeyhsd(endog=d_melt['value'], groups=d_melt['treatments'], alpha=0.05)
    print(m_comp)