# Rodolfo Isaac Verdín Monzón
# 10/10/20
# Regresión logistica


"""
El siguiente script es un algoritmo de clasificación, este acepta
como entrada valores reales y continuos, de igual este arroja salidas
continuas entre 1-0, el scrip esta dividido en las siguientes secciones:

1- Se definine la función de activación de la red neuronal, la función 
es una sigmoide  sigma(z) = 1 / (1 + e ^-z), esta es una función asintotica, 
el cual nunca va a llegar a tocar los valores de 0 u 1, solo aproximarse.

2- Definimos los valores de entradas X, Asi como su valor target 
este valor target es el de distintas compuertas logicas.

3- se declaran la taza de aprendizaje, los pesos (en este caso son aleatorios)
y el numero de epocas(epochs), un numero de epocas alto ayuda a que el valor de salida
se aproxime al del target.

4- se Concatena el bias a X

5 - Se inicia el algoritmo de regla para la regresión logistica un ciclo for 
respecto al numero de epocas, 1- se declara la funcio (z) la cual es producto punto
de las entrada y los pesos z = x * w
    2- la salida z es variable de la salida de la red Y representada por la sigmoide
       y = sigma(z)
    3- con ayuda del gradiente desecnciendete se actualizan los nuevos pesos
    4- actualización de los nuevos pesos
    5- función costo, en este caso se considera un logaritmo dada la función sigmoide.

6- Graficación
   1- epocas vs costo
   2 - boudaries target vs predicted


"""


import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import time

#Definir función de activación de tipo sigmoide

def sigmoide(z): 
    return 1./(1 + np.exp(-z)) 

#Entrada y target del perceptron

x = np.array([[0,0], [0,1],[1,0],[1,1]])
t = np.array([[1],[0],[0],[0]])

# Compuertas sustituir target por t
com_or = np.array([[0],[1],[1],[1]])
com_nor = np.array([[0],[1],[1],[1]])
com_and = np.array([[0],[1],[1],[1]])
com_nand = np.array([[0],[1],[1],[1]])

target = com_nor

n = len(t)
costo = []

#taza de aprendizaje
alpha = 0.5
#pesos aleatorios
w_i = np.random.rand(3,1)
epochs = 200

tic = time.time()

#añadiendo las columnas del bias  a x
bias = -np.ones((x.shape[0], 1))
x_1 = np.concatenate([x,bias], axis = 1)


for i in range(epochs):
  z = np.dot(x_1,w_i) 
  # print(a)
  yp = sigmoide(z)


  #regla de regresión logistica
  dw = ((t - yp).T.dot(x_1)).mean(axis=0)
  toc = time.time()
  wn = w_i.T + (alpha*dw)
  w_i = wn.T
  epoch = i
  costo.append(np.mean(-((t*np.log(yp))+(1-t)*np.log(1-yp)),axis=0))
  if costo[i] == 0:
    break

#grafica epocas vs costo 
print(epoch)
plt.plot(range(epoch+1), costo)
plt.title("Grafica de error")
plt.xlabel("epochs")
plt.ylabel("costo")
plt.show()

#Boundaries 
b, w1, w2 = w_i
ejex = -b / w1
ejey = -b / w2

d = ejey
c = -ejey / ejex

line_x_coords = np.array([0, ejex])
line_y_coords = c * line_x_coords + d

plt.plot(line_x_coords, line_y_coords)
plt.scatter(*x_1[:, 1:].T, c=t, s=75)
plt.title("Decision boundaries")
plt.show()

  
# Target vs Predicted
print('Perceptron rule')
print('target:    predicted:')
for j in range(len(yp)):
  res = str(t[j]) + '--------' + str(yp[j])
  print(res)

print('')
print('required time:{:.5f} s'.format(toc - tic))
print('required epochs: {}'.format(epoch))
