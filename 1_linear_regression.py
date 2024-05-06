# import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#import data
df = pd.read_csv("linear-regression-dataset.csv", sep = ";") # araları noktalı virgülle ayır

#plot plot
plt.scatter(df.deneyim, df.maas)
plt.xlabel("deneyim")
plt.ylabel("maas")
plt.show()

# sklearn library
from sklearn.linear_model import LinearRegression

linear_reg = LinearRegression()

# pandası numpy arraye çeviriyoruz
x = df.deneyim.values.reshape(-1,1) #reshape etmemizin amacı boyutu (14,) olarak yazan x.shapei (14, 1) olacak şekile çevirmek..
y = df.maas.values.reshape(-1,1)    #çünkü sklearn kütüphanesi bu şekiklde tanımlanıyor.

linear_reg.fit(x,y) # line'ımızı fit ettik

b0 = linear_reg.predict([[0]]) # y eksenini kestiği nokta
print("b0: ",b0)

b0_ = linear_reg.intercept_ # y eksenini kestiği nokta
print("b0_: ",b0_)

b1 = linear_reg.coef_
print("b1: ",b1)  # eğim

# maas = 1663 + 1138 * deneyim

maas_yeni = 1663 + 1138*11
print(maas_yeni)

print(linear_reg.predict([[11]]))

# visualize line
array = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]).reshape(-1, 1)  #predict etmek istediğim deneyimim

plt.scatter(x, y)
plt.show()

y_head = linear_reg.predict(array) # predict ettiğim maaşım

plt.plot(array, y_head, color = "red")

print(linear_reg.predict([[100]]))

















