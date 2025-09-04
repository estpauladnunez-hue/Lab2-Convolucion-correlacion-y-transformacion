# Lab2-Convolucion-correlacion-y-transformacion

# INTRDODUCCÍON 


En este laboratorio trabajamos con señales discretas obtenidas a partir de los dígitos de las cédulas y los códigos estudiantiles, con el fin de representarlas gráficamente y posteriormente realizar la operación de convolución entre ambas. Esta primera parte  permitió observar cómo, al combinar dos secuencias, se genera una nueva señal que refleja la interacción de los datos de entrada.

En la segunda parte del trabajo generamos las  señales sinusoidales en Python y aplicamos la correlación cruzada para analizar su grado de similitud en función de un desplazamiento. Con esto pudimos comprobar cómo esta herramienta permite identificar relaciones entre señales periódicas y cómo se manifiestan en su representación gráfica.

De esta manera, a través de la práctica logramos aplicar los conceptos de convolución y correlación cruzada vistos en clase, comprendiendo mejor su utilidad y su interpretación dentro del procesamiento de las señales.

# RESUMEN 


En este laboratorio estudiamos dos operaciones fundamentales del procesamiento digital de señales: la convolución y la correlación cruzada. En la Parte A, se construyeron señales discretas a partir de los dígitos de la cédula y el código estudiantil de cada integrante, las cuales se representaron gráficamente y se combinaron mediante la convolución, obteniendo una nueva señal resultante. En la Parte B, se analizaron dos señales sinusoidales generadas matemáticamente, aplicando la correlación cruzada para determinar su grado de similitud. Los resultados obtenidos muestran cómo estas operaciones permiten caracterizar la interacción entre señales y medir relaciones temporales, destacando su importancia en aplicaciones como comunicaciones, filtrado digital y análisis de sistemas.

# MARCO TEORICO

Señales discretas

Una señal discreta es una secuencia de valores definida en instantes específicos de tiempo, generalmente representada por índices enteros. Son fundamentales en el procesamiento digital, pues permiten manipular información en sistemas computacionales. En este laboratorio, los dígitos de cédulas y códigos estudiantiles se emplearon como señales discretas de entrada.

Convolución discreta

La convolución es una operación que describe la salida de un sistema lineal e invariante en el tiempo (LTI) dado un estímulo de entrada. Matemáticamente, para dos señales X[n] y H[n] que se define como:



<img width="220" height="85" alt="image" src="https://github.com/user-attachments/assets/0df4dfe5-16dc-4774-89da-02f0522020a6" />

Imagen [1] Ecuación denominada Convolución entre las señales discretas.

En la práctica, esta operación combina dos secuencias y refleja cómo una de ellas se ve modificada al pasar por un sistema descrito por la otra. En este laboratorio, la convolución permitió obtener una nueva señal que representa la unión entre cédula y código.

Correlación cruzada

La correlación cruzada es una medida de similitud entre dos señales en función de un desplazamiento temporal. Se define como:

<img width="1287" height="302" alt="image" src="https://github.com/user-attachments/assets/7f7db1c4-2160-4f47-9f73-36304e464199" />

Imagen [2] Correlación cruzada

En comunicaciones, se utiliza para sincronizar señales, detectar patrones y estimar retardos. En este laboratorio se aplicó a dos señales sinusoidales para identificar sus relaciones temporales.

Señales sinusoidales

En la guía se definieron dos señales periódicas:
<img width="736" height="78" alt="image" src="https://github.com/user-attachments/assets/7bcf984f-711c-4d8f-9f6e-ca41f09b36ce" />

Imagen[3] Señales sinusoidales
​
T=1.25ms. Estas funciones son ortogonales en un periodo completo, lo que implica que su correlación será nula salvo en ciertos puntos, propiedad clave en telecomunicaciones y análisis espectral.
​

# PARTE A: Señales discretas y convolucion.
Para la construccion de las señales iniciales se usaron los digitos de la cedula y del codigo estudiantil como base, se tomaron estos dos datos de cada integrante para representar una señal con cada dato, es decir una señal para la cedula y una señal para el codigo, en donde para esto la parte de nuestro codigo implementado fue:
```python
import numpy as np
import matplotlib.pyplot as plt
#------------------- PARTE A - SEÑALES DISCRETAS----------------------
#          Códigos, Cédulas y Unión por Convolución

# Datos de los integrantes:
integrantes = [
    {
        "nombre": "Paula Núñez",
        "codigo": [5, 6, 0, 0, 7, 2, 0],        # Código separado en dígitos
        "cedula": [1, 0, 5, 3, 3, 2, 2, 1, 7, 6]  # Cédula separada en dígitos
    },
    {
        "nombre": "Kevin Ducuara",
        "codigo": [5, 6, 0, 0, 7, 1, 8],
        "cedula": [1, 0, 7, 5, 6, 8, 7, 9, 3, 4]
    },
    {
        "nombre": "Ana María Díaz",
        "codigo": [5, 6, 0, 0, 5, 8, 9],
        "cedula": [1, 0, 1, 3, 2, 5, 9, 2, 9, 1]
    }
]
```
Estas señales se graficaron individualmente para visualizar como varian sus valores, lo que nos permitio identificar patrones y distribuciones presentes en los datos personales de cada integrante, implementando esta parte de nuestro codigo:
```python
# Graficar resultados por integrante
for integ in integrantes:
    cod = np.array(integ["codigo"])
    ced = np.array(integ["cedula"])

    # Graficas individuales
    plt.figure(figsize=(12, 5))
    plt.suptitle(f"Parte A - {integ['nombre']}", fontsize=15, fontweight="bold")

    # Cédula
    plt.subplot(2, 1, 1)
    plt.stem(range(len(ced)), ced, basefmt="k", linefmt="g-", markerfmt="go")
    plt.title(f"Cédula: {integ['cedula']}", fontsize=12)
    plt.ylabel("Valor")
    plt.xticks(range(len(ced)))  # Mostrar índices exactos
    plt.grid(True)

    # Código
    plt.subplot(2, 1, 2)
    plt.stem(range(len(cod)), cod, basefmt="k", linefmt="b-", markerfmt="bo")
    plt.title(f"Código: {integ['codigo']}", fontsize=12)
    plt.xlabel("Posición del dígito")
    plt.ylabel("Valor")
    plt.xticks(range(len(cod)))
    plt.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
```
Para cada integrante obtuvimos dos señales, la primera corresponde a la señal de la cedula y la segunda corresponde a la señal del codigo:
<img width="1189" height="494" alt="image" src="https://github.com/user-attachments/assets/f236e0c2-5ba6-4cdf-9226-47fd80d8a89f" />
                [1]Graficas de cedula y codigo-1er integrante.

<img width="1189" height="494" alt="image" src="https://github.com/user-attachments/assets/b202c676-2714-4218-9d51-89b214bd68eb" />
                [2]Graficas de cedula y codigo-2do integrante.

<img width="1189" height="494" alt="image" src="https://github.com/user-attachments/assets/60655c62-1c79-46ed-a549-fe78086cea67" />
                [3]Graficas de cedula y codigo-3er integrante.

Una vez definidas estas dos señales para cada integrante, se realizo la convolucion entre ambas, este proceso combina la informacion de la cedula y el codigo, generando una nueva señal resultante con 16 puntos, la cual refleja el efecto conjunto de ambas secuencias, gracias a la implementacion de esta parte de nuestro codigo:
```python
#Union= convolución entre cédula y código
    union = np.convolve(cod, ced)

    # Grafica de la Unión de Cédula y Código
    plt.figure(figsize=(10, 5))
    plt.title(f"Parte A - {integ['nombre']} | Unión (Convolución)", fontsize=15, fontweight="bold")
    plt.stem(range(len(union)), union, basefmt="k", linefmt="r-", markerfmt="ro")
    plt.xlabel("n (muestras)")
    plt.ylabel("Amplitud")
    plt.xticks(range(len(union)))  # Mostrar todos los puntos claramente
    plt.grid(True)
    plt.show()
```
Para cada integrante obtuvimos una tercera grafica mostrando la señal final obtenida tras la convolucion de cada integrante:
<img width="842" height="474" alt="image" src="https://github.com/user-attachments/assets/7ae12003-5dfe-4737-8341-6fbc2075d389" />
                 [4]Union (convolucion) datos del 1er integrante.

<img width="850" height="474" alt="image" src="https://github.com/user-attachments/assets/e6a32589-71ae-463f-9549-c8623d90e92d" />
                 [5]Union (convolucion) datos del 2do integrante.

<img width="850" height="474" alt="image" src="https://github.com/user-attachments/assets/c5b4d4e5-25cf-4518-a8cd-c1e5c879e3e9" />
                 [6]Union (convolucion) datos del 3er integrante.

# PARTE B: Correlacion Cruzada.
En esta segunda parte del laboratorio, trabajamos con dos señales sinusoidales generadas matemáticamente para estudiar su relación y similitud utilizando el concepto de correlación cruzada.
