from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from sksurv.datasets import load_breast_cancer
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, KFold
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


#Fucntion to plot the continuous graph made by the different alpha(penalty) for each gene(variates) in our analysis and find the point where the variate converges to 0
def plot_coefficients(coefs, n_highlight):
    _, ax=plt.subplots(figsize=(9,6))
    n_features=coefs.shape[0]
    alphas=coefs.columns
    print(n_features, alphas)
    for row in coefs.itertuples():
        ax.semilogx(alphas, row[1:], ".-", label=row.Index)

    alpha_min=alphas.min()
    top_coefs=coefs.loc[:, alpha_min].map(abs).sort_values().tail(n_highlight)
    for name in top_coefs.index:
        coef=coefs.loc[name, alpha_min]
        plt.text(alpha_min, coef, name + "   ", horizontalalignment="right", verticalalignment="center")

    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.grid(True)
    ax.set_xlabel("alpha")
    ax.set_ylabel("coefficient")

#For when we needed to combine two seperate dataframes(files)
def merge_frames(df1, df2):
    df=pd.merge(df1,df2, on=["Sample", "Status", "Time"])
    return(df)

#For when you want to remove certain samples that is present in one but absent in the other dataframe
def intersection(df1, df2):
    df1_sample=df1["Sample"].to_numpy();
    df2_sample=df2["Sample"].to_numpy();

    diff=[]

    for samp1 in df2_sample:
        if samp1 not in df1_sample:
            diff.append(samp1)

    for x in diff:
        indx=df2[df2["Sample"]==x].index
        df2.drop(indx, inplace=True)

    df=df2
    return(df)




#Reading the survival analysis file
df1=pd.read_csv("forRegression.csv")
df2=pd.read_csv("forRegressionTest.tsv", sep="\t")

#Intersection between two files
#df=intersection(df1, df2)

#Adding genes based on the sample, status and time
#df=merge_frames(df1,df2)


#Converting the integer to 0 and 1 to boolean for python
df["Status"]=df["Status"].astype(bool)
#data contains the time and status column and X will have all the mutation present or absent corresponding to each gene
data=df.iloc[0:, 1:3]
X=df.iloc[0:, 3:]

#storing the value used to store status and time in tuple
Y=data.to_records(index=False)

X=OneHotEncoder().fit_transform(X)

#Running the module for 50 randomly generated penalty values 
estimator=CoxnetSurvivalAnalysis(n_alphas=100, l1_ratio=1, alpha_min_ratio=0.01, max_iter=10000)
estimator.fit(X,Y)

#Making the dataframe for the coefficients of each genes corresponding to that alpha value
coefficients_lasso=pd.DataFrame(estimator.coef_, index=X.columns, columns=np.round(estimator.alphas_, 5))
alphas=estimator.alphas_

print(coefficients_lasso)

#Sending parameters to the function to plot the alpha vs coefficient graph for all the genes, with the 10 mostly divergent genes as hightlights
plot_coefficients(coefficients_lasso, n_highlight=10)
alphas=coefficients_lasso.columns

#Determination of the alpha value for which we want to make our cox model to obtain the coefficient values
coxnet_pipe=make_pipeline(StandardScaler(), CoxnetSurvivalAnalysis(l1_ratio=1, alpha_min_ratio=0.01, max_iter=10000))
warnings.simplefilter("ignore", ConvergenceWarning)
coxnet_pipe.fit(X,Y)

#Doing a 5-fold cross validation for each alpha value along with its standard deviation
cv=KFold(n_splits=5, shuffle=True, random_state=0)
gcv=GridSearchCV(make_pipeline(StandardScaler(), CoxnetSurvivalAnalysis(l1_ratio=1.0, alpha_min_ratio=0.01, max_iter=10000)), param_grid={"coxnetsurvivalanalysis__alphas":[[v] for v in alphas]}, cv=cv, error_score=0.5, n_jobs=4).fit(X,Y)

cv_results=pd.DataFrame(gcv.cv_results_)

alphas= cv_results.param_coxnetsurvivalanalysis__alphas.map(lambda x: x[0])
mean=cv_results.mean_test_score
std= cv_results.std_test_score

plt.figure(1)
fig, ax=plt.subplots(figsize=(9,6))
ax.plot(alphas, mean)
ax.fill_between(alphas, mean - std, mean + std, alpha=.15)
ax.set_xscale("log")
ax.set_ylabel("concordance index")
ax.set_xlabel("alpha")
ax.axvline(gcv.best_params_["coxnetsurvivalanalysis__alphas"][0], c="C1")
ax.axhline(0.5, color="grey", linestyle="--")
ax.grid(True)

#With the best alpha value found we print the corresponding coefficients for all the genes which are non zero
best_model=gcv.best_estimator_.named_steps["coxnetsurvivalanalysis"]
best_coefs=pd.DataFrame(best_model.coef_, index=X.columns, columns=["coefficient"])
print(best_coefs)
#best_coefs.to_csv("../Data_Images/6_genes_reduced_samples_coef.csv", sep="\t")

non_zero=np.sum(best_coefs.iloc[:, 0] !=0)
print("Number or non-zero coefficients: {}".format(non_zero))

#Plot for coefficient for each gene 
plt.figure(2)
non_zero_coefs=best_coefs.query("coefficient != 0")
coef_order=non_zero_coefs.abs().sort_values("coefficient").index

_, ax=plt.subplots(figsize=(6,8))
non_zero_coefs.loc[coef_order].plot.barh(ax=ax, legend=False)
ax.set_xlabel("coefficient")
ax.grid(True)
plt.show()
