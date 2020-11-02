
# Rodolfo Isaac Verdín Monzón
# 02/11/20
# MLP multi layer perceptron

"""
El sguiente script es un algoritmo MLP, este tiene como caracteristica que cada neurona
usa una función de activación de tipo sigmoide, Este cuenta con tres layers teniendo solamente un nodo en la salida
este cuenta con dos entradas y un bias como tercera, en el segundo layer se cuentan con tres nodos un bias y dos neuronas
Para los pesos se cuentan con 6 para la capa oculta y 3 para la capa de salida; En total son 9 pesos.
"""

import numpy as np
import matplotlib.pyplot as plt
import time

#Función de activación de tipo sigmoide
def sigmoide(x):
  return 1/(1 + np.exp(-x))

 #Derivada de función sigmoide; para gradiente descendiente.
def sigmoide_deriv(x):
  return sigmoide(x)*(1-sigmoide(x)) 

# Grafica de frontera de decisión 
def heaviside(x):
    return (x >= 0).astype(x.dtype)

def mlp_xor(x1, x2, activacion=heaviside):
    return activacion(
        -activacion(x1 + x2 - 1.4) + activacion(x1 + x2 - 0.1) - 0.1)

#matriz de entradas
X_1 = np.array([[0,0],  
            [0,1],
            [1,0],
            [1,1]])
x1 = X[:, 0]
x2 = X[:, 1]

#target
t = np.array([[0],[1],[1],[0]])  #compuerta and

# t = np.array([[0],[1],[1],[1]]) #compuerta or
# t = np.array([[0],[1],[1],[0]]) #compuerta xor
# t = np.array([[1],[1],[1],[0]]) #compuerta nand 
# t = np.array([[1],[0],[0],[0]]) #compuerta nor

costs = []

# Learning rate y numero de epocas 
alpha = 0.5
epochs = 3000

#Iniciar pesos aleatorios 
# 6 para la capa oculta
# 3 para la capa de salida; En total son 9 pesos 
wi = np.random.rand(3,2)
wj = np.random.rand(3,1)

tic = time.time()

# Concatenación del bias con la matriz de entradas 
bias1 = -np.ones((X_1.shape[0], 1))
X = np.concatenate([X_1,bias1], axis = 1)
m = len(X)

#Feed fordward
for i in range(epochs):
  z1, a1, z2, a2 = forward(X, wi, wj)

  #backpropagation ( actualización de pesos o nuevos pesos)
  delta2, Delta1, Delta2 = backprop(z2, X, a1, a2, t)
  wi -= alpha*(1/m)*Delta1
  wj -= alpha*(1/m)*Delta2

  #error de salida 
  c =  c = np.mean(np.abs(delta2))
  costs.append(c)
 
#feed-Forward pesos de entrada
def forward(x,w1,w2,predict = False): 
  z1 = np.dot(x,w1)
  a1 = sigmoide(z1)
  

  # Nuevo bias ( este bias pertenece al segundo layer)
  bias = np.ones((len(a1),1)) 

# feed-Forward pesos de salida ( En estas sección se realiza un rpducto punto de los pesos de salida con la nueva entreda las neuronas de segundo layer)
  a1 = np.concatenate((bias,a1),axis=1)  
  z2 = np.dot(a1,w2)
  a2 = sigmoide(z2)
  if predict:
    return a2
  return z1 , a1, z2, a2

#backpropagation function ( se calculan los Deltas a partir del metodo de gradiente descendiente)
def backprop(z2,z0,a1,a2,y):

  delta2 = a2 - y 
  Delta2 = np.dot(a1.T,delta2)
  delta1 = (delta2.dot(wj[1:,:].T))*sigmoide_deriv(z1)
  Delta1 = np.dot(z0.T,delta1)
  return delta2, Delta1, Delta2

toc = time.time()  

"""
SECCIÓN DE GRAFICAS Y TABLAS

1- TARGET VS PREDICTED : muetras los valores de entrada y el target elejido por ultimo se muestran los valores finales de la red respecto al target.
2- FRONTERAS DE DECISIÓN: en esta sección se realizó un meshgrid  y graficaron los valores de target.
3- GRAFICA DE ERROR: grafica el error respecto al numero de epocas.
"""
  # Target vs Predicted
print('Resultados del MLP ')
print('entradas: target: predicted:')
for j in range(len(a2)):
  res = str(X_1[j])+'-----'+str(t[j]) +'-----'+ str(a2[j])              
  print(res)

#Gradica de error
plt.plot(costs)
plt.title("Grafica de error")
plt.xlabel("epochs")
plt.ylabel("costo")
plt.show  

#Fronteras de decisión
x1s = np.linspace(-1, 1.5, 10)
x2s = np.linspace(-1, 1.5, 10)
x1, x2 = np.meshgrid(x1s, x2s)
y2 = mlp_xor(x1, x2, activacion=sigmoide)
plt.figure(figsize=(10,4))
plt.subplot(122)
plt.contourf(x1, x2, y2)
plt.scatter(X_1[:, 0],X_1[:, 1], c=t, s=80)
plt.title("Fronteras de decisión", fontsize=14)
plt.grid(False)
plt.colorbar() 
plt.show()

  
print('')
print('required time:{:.5f} s'.format(toc - tic))
print('required epochs: {}'.format(epochs))

