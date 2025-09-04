# Lab2-Convolucion-correlacion-y-transformacion

# INTRDODUCCÍON 

El análisis y procesamiento de señales discretas constituye una herramienta fundamental dentro de la ingeniería, ya que permite estudiar el comportamiento de datos en el dominio digital y extraer información relevante para su aplicación en diversos campos. En este laboratorio se abordaron dos operaciones esenciales en el tratamiento de señales: la convolución y la correlación cruzada.

La primera de ellas, la convolución, es clave para comprender cómo interactúan dos secuencias al combinarse, lo cual resulta indispensable en el diseño y análisis de sistemas lineales e invariantes en el tiempo. Por su parte, la correlación cruzada es un procedimiento que permite medir el grado de similitud entre dos señales en función de un desplazamiento temporal, lo que la hace especialmente útil en aplicaciones de detección, comparación y análisis de señales periódicas.

A través de esta práctica, se buscó reforzar los conceptos vistos en clase, trasladándolos a un entorno de programación en Python que facilitó la construcción, visualización e interpretación de las señales, brindando una perspectiva más clara de su utilidad en el procesamiento digital de señales.

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
                [4]Graficas de cedula y codigo-1er integrante.

# Análisis de la Gráfica – Parte A: Paula Núñez

En la primera figura se observa la señal construida a partir de los dígitos de la cédula:
[1,0,5,3,3,2,2,1,7,6]. Esta secuencia presenta una variación de valores entre 0 y 7, lo que refleja una distribución dispersa de amplitudes. Se nota que los ceros y valores bajos se alternan con picos más altos (como en las posiciones 2, 8 y 9), generando una señal no periódica y sin un patrón repetitivo claro, lo cual es característico ya que los datos provienen de un número personal.

En la segunda figura se representa la señal del código estudiantil :
[5,6,0,0,7,2,0]. Aquí se observa una secuencia más corta que la de la cédula, con valores que oscilan entre 0 y 7. La presencia de varios ceros genera intervalos de reposo, intercalados con picos notables en las posiciones 0, 1 y 4. Esto provoca que la señal tenga un comportamiento más irregular y con saltos bruscos entre valores consecutivos.

Comparando ambas señales, se puede decir que la de la cédula es más extensa y con mayor variedad de amplitudes, mientras que la del código es más breve y con valores concentrados en pocos puntos destacados. Este contraste será importante al momento de realizar la convolución, ya que el resultado reflejará cómo los picos del código se expanden sobre la señal de la cédula, generando una nueva secuencia que combina ambas características.


<img width="1189" height="494" alt="image" src="https://github.com/user-attachments/assets/b202c676-2714-4218-9d51-89b214bd68eb" />
                [5]Graficas de cedula y codigo-2do integrante.
                
# Análisis de la Gráfica – Parte A: Kevin 

En la primera figura se observa la señal construida a partir de los dígitos de la cédula [1, 0, 7, 5, 6, 8, 7, 9, 3, 4]. Esta secuencia presenta una variación amplia de valores entre 0 y 9, lo que refleja una distribución dispersa de amplitudes. Se aprecia que algunos dígitos corresponden a valores bajos, como el 0 en la posición 1 y el 3 en la posición 8, mientras que otros alcanzan picos altos como el 9 en la posición 7 y el 8 en la posición 5. Esta variabilidad genera una señal irregular, sin un patrón periódico claro, lo cual es coherente con que los datos provienen de un número personal.

En la segunda figura se representa la señal del código estudiantil [5, 6, 0, 0, 7, 1, 8]. Aquí se observa una secuencia más corta en comparación con la de la cédula, con valores que oscilan entre 0 y 8. Se destaca la presencia de dos ceros consecutivos en las posiciones 2 y 3, lo que genera un intervalo de reposo, mientras que en la última posición aparece el valor máximo de 8, que produce un pico al final de la señal. Este comportamiento evidencia un contraste marcado entre los valores bajos y los altos, generando saltos bruscos entre dígitos consecutivos.

Comparando ambas señales, se puede decir que la de la cédula es más extensa y presenta mayor variedad de amplitudes distribuidas de manera irregular, mientras que la del código estudiantil es más breve y concentra los valores más significativos en ciertos puntos específicos. Este contraste entre extensión y variabilidad permite diferenciar el comportamiento de cada secuencia y resulta útil para el análisis de señales discretas.

<img width="1189" height="494" alt="image" src="https://github.com/user-attachments/assets/60655c62-1c79-46ed-a549-fe78086cea67" />
                [6]Graficas de cedula y codigo-3er integrante.

                
# Análisis de la Gráfica – Parte A: Ana Maria Diaz 

En la primera figura se observa la señal construida a partir de los dígitos de la cédula [1, 0, 1, 3, 2, 5, 9, 2, 9, 1]. Esta secuencia muestra valores que oscilan entre 0 y 9, con una distribución irregular de amplitudes. Se identifican valores bajos como los ceros en las posiciones 1 y 2, y también picos altos, especialmente en las posiciones 6 y 8, donde aparece el valor máximo de 9. Esto genera una señal que combina tramos de baja amplitud con variaciones bruscas hacia valores elevados, sin un patrón periódico definido, lo que refleja la naturaleza aleatoria de los datos al provenir de un número personal.

En la segunda figura se representa la señal del código estudiantil [5, 6, 0, 0, 5, 8, 9]. A diferencia de la cédula, esta secuencia es más corta, pero presenta valores igualmente contrastantes. Se observan dos ceros consecutivos en las posiciones 2 y 3, que generan un intervalo de reposo, y picos altos en las posiciones finales con valores de 8 y 9, lo cual da un cierre abrupto a la señal. La disposición de ceros intercalados con valores altos resalta un comportamiento irregular y con saltos pronunciados.

Comparando ambas señales, se puede afirmar que la de la cédula es más extensa y variada, con alternancia entre valores bajos y picos altos que generan una señal irregular y dispersa. En cambio, la del código estudiantil es más breve, pero concentra las amplitudes más significativas en sus últimos puntos. Este contraste evidencia que la cédula distribuye la variabilidad a lo largo de más posiciones, mientras que el código la concentra en pocos instantes destacados.

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
                 [7]Union (convolucion) datos del 1er integrante.
                 
 En la gráfica se observa el resultado de la convolución, la cual genera una señal más larga que las secuencias originales, extendiéndose desde la muestra 0 hasta la 15. La amplitud alcanza un máximo cercano a 90 en la posición 9, mostrando cómo la superposición de los valores de entrada refuerza ciertos puntos. La señal presenta un crecimiento progresivo hasta el pico máximo, seguido de un descenso gradual, lo que refleja el comportamiento característico de la convolución: acumulación inicial, punto de máxima coincidencia y luego disminución al agotarse las superposiciones.

<img width="850" height="474" alt="image" src="https://github.com/user-attachments/assets/e6a32589-71ae-463f-9549-c8623d90e92d" />
                 [8]Union (convolucion) datos del 2do integrante.

En la gráfica se aprecia el resultado de la convolución, que se extiende desde la muestra 0 hasta la 15. La señal muestra un crecimiento progresivo en amplitud hasta alcanzar su pico máximo cercano a 160 en la posición 8, lo que indica el punto de mayor coincidencia entre las secuencias originales. Después de este valor, la amplitud comienza a descender de manera gradual, reflejando la disminución de las superposiciones. El comportamiento general evidencia una forma triangular asimétrica, típica de la convolución, donde la energía se concentra en el centro de la señal.

<img width="850" height="474" alt="image" src="https://github.com/user-attachments/assets/c5b4d4e5-25cf-4518-a8cd-c1e5c879e3e9" />
                 [9]Union (convolucion) datos del 3er integrante.

En la gráfica se observa el resultado de la convolución, que abarca desde la muestra 0 hasta la 15. La señal inicia con valores bajos y va aumentando progresivamente hasta alcanzar su máximo cercano a 120 en la posición 9, lo que corresponde al punto de mayor solapamiento entre las secuencias. Posteriormente, la amplitud disminuye de manera gradual hasta llegar nuevamente a valores bajos. Este comportamiento refleja el patrón típico de la convolución: crecimiento inicial, un pico central marcado y un descenso simétrico hacia el final de la señal.

# PARTE B: Correlacion Cruzada.
En esta segunda parte del laboratorio, trabajamos con dos señales sinusoidales generadas matemáticamente para estudiar su relación y similitud utilizando el concepto de correlación cruzada.

Para generar las señales, definimos dos funciones periódicas discretas: 
x1​[n]=cos(w⋅n)
x2​[n]=sin(w⋅n)
donde w=2πf⋅Ts, con f=100Hz como frecuencia de la señal y Ts=1.25ms como periodo de muestreo. Para lo cual implementamos esta parte de nuestro codigo, generando nueve muestras para cada señal:

```python
# Definicion de las señales
Ts = 1.25e-3      # Periodo de muestreo (1.25 ms)
f = 100           # Frecuencia en Hz
n = np.arange(0, 9)
w = 2 * np.pi * f * Ts

x1 = np.cos(w * n)
x2 = np.sin(w * n)

# Grafica de las Señales originales
plt.figure(figsize=(10, 5))
plt.suptitle("Parte B - Señales Originales", fontsize=14)

plt.subplot(2, 1, 1)
plt.stem(n, x1, basefmt="k")
plt.title("x1[n] = cos(w*n)")
plt.ylabel("Amplitud")
plt.grid()

plt.subplot(2, 1, 2)
plt.stem(n, x2, basefmt="k")
plt.title("x2[n] = sin(w*n)")
plt.xlabel("n (muestras)")
plt.ylabel("Amplitud")
plt.grid()

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
```
Luego, al graficarlas pudimos observar que ambas son señales periódicas, pero presentan un desfase de 90° debido a la diferencia entre las funciones seno y coseno. Esto significa que cuando X1[n] alcanza su valor maximo, X2[n] se encuentra en cero, y viceversa.

<img width="800" height="563" alt="image" src="https://github.com/user-attachments/assets/3b6f941f-139a-454f-8676-27f20dc77625" />
                     [10] X1[n] =cos (π/4n)
muestra un coseno discreto que inicia en su maxima amplitud positiva, lo cual es caracteristico del coseno, presentando una variacion periodica drecreciendo hacia valores negativos, para finalmente retornar a su amplitud inicial.


<img width="800" height="563" alt="image" src="https://github.com/user-attachments/assets/5bf5e23a-046c-4bbb-8c76-89815506e177" />

                     [11] X2 [n] =sin (π/4n)
esta imagen muestra un seno discreto , comenzando en cero, alcanza un valor maximo positivo en n=2, y posteriormente disminuye atravesando valores negativos antes de regresar a cero.
Ambas señales tienen la misma frecuencia angular y, por lo tanto, la misma periodicidad, pero presentan un desfase de π/2 radianes. Este desfase se aprecia porque los máximos y mínimos de X1[n] ocurren medio periodo antes que los de X2 [n]. La amplitud en ambas señales está limitada entre -1 y 1, como corresponde a funciones trigonométricas puras sin escalamiento.

Posteriormente, calculamos la correlación cruzada entre ambas señales para determinar qué tan similares son en función del desplazamiento entre ellas.
Implementando esta parte de nuestro codigo, este cálculo se realizó de dos maneras:
```python
# Mostrar en consola el máximo coeficiente de correlación
max_corr = np.max(r12_norm)
k_max = lags[np.argmax(r12_norm)]
print(f"\nMáxima correlación normalizada = {max_corr:.3f} en k = {k_max}")

# Calculo de correlación cruzada
N = len(n)
lags = np.arange(-(N - 1), N)
r12 = np.zeros(len(lags))

for i, k in enumerate(lags):
    suma = 0
    for ni in range(N):
        nk = ni + k
        if 0 <= nk < N:
            suma += x1[ni] * x2[nk]
    r12[i] = suma

# Calculo de correlación normalizada
E1 = np.sum(x1**2)
E2 = np.sum(x2**2)
r12_norm = r12 / np.sqrt(E1 * E2)
```
<img width="363" height="57" alt="image" src="https://github.com/user-attachments/assets/ce8504bc-539e-43b8-a519-e3308e6e675f" />
[12] Vlor de correlacion maxima

Con este valor encontramos que la máxima correlación normalizada es muy baja, lo que confirma que las dos señales, aunque tienen la misma frecuencia, son ortogonales. Esto significa que no comparten fase y, por tanto, su similitud directa es mínima.

Finalmente, representamos gráficamente los resultados de la correlación cruzada en dos figuras, gracias a esta parte de nuestro codigo:
```python
# Grafica de las Correlaciones
plt.figure(figsize=(12, 6))
plt.suptitle("Parte B - Correlaciones entre x1[n] y x2[n]", fontsize=14)

plt.subplot(2, 1, 1)
plt.stem(lags, r12, basefmt="k")
plt.title("Correlación cruzada r12[k] (sin normalizar)")
plt.xlabel("Retardo k")
plt.ylabel("r12[k]")
plt.grid()

plt.subplot(2, 1, 2)
plt.stem(lags, r12_norm, basefmt="k")
plt.title("Correlación cruzada normalizada")
plt.xlabel("Retardo k")
plt.ylabel("Coeficiente")
plt.grid()

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
```

Correlación cruzada no normalizada: 
Nos permitió observar la magnitud absoluta de la similitud entre las señales, considerando directamente sus valores sin escalamiento.
<img width="768" height="563" alt="image" src="https://github.com/user-attachments/assets/307132b5-4c06-4e9d-82d1-90995dabc543" />
                 [12] Correlacion cruzada.
Con esta observamos  que la correlacion maxima ocurre alrededor de k=2 y k=-7, donde le valor r12 alcanza su pico positivo, ademas en los retardos en -2 y 3, los valores minimos negativos indican maxima oposicion entre las señales, lo que quiere decir que mientras una señal esta en su maximo la otra esta en su minimo, y esta sorma simetrica es coherente con la naturaleza periodica y desfasada de seno y coseno.

Correlación cruzada normalizada:
Ajustó los valores de la correlación a un rango comprendido entre -1 y 1, facilitando la interpretación de los resultados.
<img width="787" height="563" alt="image" src="https://github.com/user-attachments/assets/afd15713-871a-41df-ab99-9fcb737cdb66" />
                [13] correlacion cruzada normalizada 

#PARTE C
Para esta parte usamos una señal de neuropatia lumbar electromiografica biomedica estocastica de banda ancha descargada de fhysionet

<img width="1280" height="501" alt="image" src="https://github.com/user-attachments/assets/eca0042b-97f6-4fd0-80e4-18fd4396ec8c" />

Imagen[14] señal emg en le tiempo

<img width="1280" height="484" alt="image" src="https://github.com/user-attachments/assets/2477a9b7-b2c1-47b2-a41c-872da333c366" />
Imgagen [15] FFT magnitud 

<img width="1280" height="486" alt="image" src="https://github.com/user-attachments/assets/3d69682f-2248-4b14-8aa6-9cdeca2262c5" />
Imagen [16]Densidad espectral del potencia 


Descripcion:
Señal EMG: biomédica, estocástica, de banda ancha. Se considera cuasi-estacionaria en ventanas cortas.

Informacion de registo :

Número de canales: 1
Canal analizado: 0
Frecuencia muestreo (Fs): 4000.0000 Hz
Muestras (N): 147858
Duración: 36.965 s

Estadisticos en el tiempo:
    media:  0.004976
   mediana:  0.008300
  desv_std:  0.388390
  varianza:  0.150847
       min: -3.276700
       max:  3.275300
     rango:  6.552000
       RMS:  0.388420
 asimetría: -1.404344
  curtosis:  21.952436

  Estadisticos de frecuencia:
  
Frecuencia media   : 460.018 Hz
Frecuencia mediana : 248.864 Hz
Desv.Std frecuencia: 494.674 Hz

<img width="1280" height="491" alt="image" src="https://github.com/user-attachments/assets/844640a9-4a73-4a08-9914-81350fb93ff2" />

Imagen [17] Histograma de la señal 

# Conclusiones
Al finalizar este laboratorio comprendimos de manera práctica el funcionamiento y la utilidad de dos operaciones fundamentales en el procesamiento digital de señales: la convolución y la correlación cruzada.

En la Parte A, al construir las señales a partir de los dígitos de mi cédula y código estudiantil, pude observar cómo se comportan gráficamente estas secuencias y cómo la convolución permite combinarlas en una nueva señal más larga que refleja la interacción entre ambas. Esto me ayudó a visualizar un concepto que en la teoría suele ser abstracto, pero que en la práctica se convierte en una herramienta clave para modelar sistemas lineales e invariantes en el tiempo.

En la Parte B, con las señales sinusoidales definidas en la guía, confirmé que la correlación cruzada es un método efectivo para medir la similitud entre señales en función de un desplazamiento. Al aplicarla, noté cómo se pueden identificar los desfases donde existe mayor coincidencia, lo cual es de gran importancia en aplicaciones de telecomunicaciones, detección de patrones y análisis espectral.

En general, este laboratorio permitió reforzar conocimientos teóricos, aplicarlos en Python y analizar los resultados obtenidos a través de gráficas y secuencias numéricas. Considero que la experiencia fue muy útil, ya que me ayudó a conectar la teoría con la práctica y a entender mejor la relevancia de estas operaciones en el campo de la ingeniería.
# BIBLIOGRAFIA 
Oppenheim, A. V., & Schafer, R. W. (2010). Discrete-Time Signal Processing (3rd ed.). Pearson.

Proakis, J. G., & Manolakis, D. G. (2007). Digital Signal Processing: Principles, Algorithms, and Applications (4th ed.). Pearson Prentice Hall.

Mitra, S. K. (2011). Digital Signal Processing: A Computer-Based Approach (4th ed.). McGraw-Hill.

Hayes, M. H. (1996). Statistical Digital Signal Processing and Modeling. John Wiley & Sons.

Universidad Militar Nueva Granada. (2022). Guía de Laboratorio 2: Convolución, correlación y transformación. Facultad de Ingeniería Biomédica. 

Smith, S. W. (1997). The Scientist and Engineer’s Guide to Digital Signal Processing. California Technical Publishing. Disponible en: http://www.dspguide.com
