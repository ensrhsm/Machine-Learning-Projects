import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("polynomial-regression.csv", sep = ";")

y = df.araba_max_hiz.values.reshape(-1,1)
x = df.araba_fiyat.values.reshape(-1,1)

plt.scatter(x,y, label = "Veri Noktaları")
plt.ylabel("araba_max_hiz")
plt.xlabel("araba_fiyat")
plt.show()

# linear regression = y = b0 + b1*x
# multiple linear regression = y = b0 + b1*x1 + b2*x2 + .... + bn*xn

from sklearn.linear_model import LinearRegression

linear_regression = LinearRegression()

linear_regression.fit(x, y)

y_head = linear_regression.predict(x)

plt.plot(x, y_head, color = "red", label = "linear")
plt.show()

print("10 milyon TL'lik araba hizi tahmini(linear regression): ",linear_regression.predict([[10000]]))

# polynomial regression = y = b0 + b1*x + b2*x^2 + b3*x^3 + .... + bn*x^n

from sklearn.preprocessing import PolynomialFeatures
polynomial_regression = PolynomialFeatures(degree = 4) # degree = n sayısı

x_polynomial = polynomial_regression.fit_transform(x) # x^2 yi oluşturuyoruz.

linear_regression2 = LinearRegression()
linear_regression2.fit(x_polynomial, y)

y_head2 = linear_regression2.predict(x_polynomial)

plt.plot(x, y_head2, color = "green", label = "polynomial")
plt.legend()
plt.show()



























