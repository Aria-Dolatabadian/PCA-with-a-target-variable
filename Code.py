from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from bioinfokit.analys import get_data
from bioinfokit.visuz import cluster
import pandas as pd
df = pd.read_csv('iris.csv')
print(df)
X = df.iloc[:,0:4]
target = df['class'].to_numpy()
X.head(2)
X_st =  StandardScaler().fit_transform(X)
pca_out = PCA().fit(X_st)
# component loadings
loadings = pca_out.components_
loadings
# get eigenvalues (variance explained by each PC)
pca_out.explained_variance_
# get biplot
pca_scores = PCA().fit_transform(X_st)
cluster.biplot(cscore=pca_scores, loadings=loadings, labels=X.columns.values, var1=round(pca_out.explained_variance_ratio_[0]*100, 2),
    var2=round(pca_out.explained_variance_ratio_[1]*100, 2), colorlist=target)

#See WD to find the result 
