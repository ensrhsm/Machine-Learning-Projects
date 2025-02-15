from sklearn.datasets import load_iris
import pandas as pd


iris = load_iris()

data = iris.data
feature_names = iris.feature_names
y = iris.target

df = pd.DataFrame(data,columns = feature_names)
df["sinif"] = y

x = data

# PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 2, whiten= True )  # whitten = normalize
pca.fit(x) # 4 boyutlu datayı 2 boyuta düşürecek modeli elde ediyoruz.(matematik)

x_pca = pca.transform(x)

print("variance ratio: ", pca.explained_variance_ratio_) # ne kadarı p1 ne kadarı p2 onu öğrenmek için

print("sum: ",sum(pca.explained_variance_ratio_)) # verilerin ne kadarını koruyabilmişiz varyansı ne kadar koruyabilmişiz onu öğrenmek için

# 2D görselleştirme

df["p1"] = x_pca[:,0]
df["p2"] = x_pca[:,1]

color = ["red","green","blue"]

import matplotlib.pyplot as plt
for each in range(3):
    plt.scatter(df.p1[df.sinif == each],df.p2[df.sinif == each],color = color[each],label = iris.target_names[each])
    
plt.legend()
plt.xlabel("p1")
plt.ylabel("p2")
plt.show()