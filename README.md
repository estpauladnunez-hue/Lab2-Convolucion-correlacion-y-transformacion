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

# Análisis de la Gráfica – Parte A: Paula Núñez

En la primera figura se observa la señal construida a partir de los dígitos de la cédula:
[1,0,5,3,3,2,2,1,7,6]. Esta secuencia presenta una variación de valores entre 0 y 7, lo que refleja una distribución dispersa de amplitudes. Se nota que los ceros y valores bajos se alternan con picos más altos (como en las posiciones 2, 8 y 9), generando una señal no periódica y sin un patrón repetitivo claro, lo cual es característico ya que los datos provienen de un número personal.

En la segunda figura se representa la señal del código estudiantil :
[5,6,0,0,7,2,0]. Aquí se observa una secuencia más corta que la de la cédula, con valores que oscilan entre 0 y 7. La presencia de varios ceros genera intervalos de reposo, intercalados con picos notables en las posiciones 0, 1 y 4. Esto provoca que la señal tenga un comportamiento más irregular y con saltos bruscos entre valores consecutivos.

Comparando ambas señales, se puede decir que la de la cédula es más extensa y con mayor variedad de amplitudes, mientras que la del código es más breve y con valores concentrados en pocos puntos destacados. Este contraste será importante al momento de realizar la convolución, ya que el resultado reflejará cómo los picos del código se expanden sobre la señal de la cédula, generando una nueva secuencia que combina ambas características.


<img width="1189" height="494" alt="image" src="https://github.com/user-attachments/assets/b202c676-2714-4218-9d51-89b214bd68eb" />
                [2]Graficas de cedula y codigo-2do integrante.
                
# Análisis de la Gráfica – Parte A: Kevin 

En la primera figura se observa la señal construida a partir de los dígitos de la cédula [1, 0, 7, 5, 6, 8, 7, 9, 3, 4]. Esta secuencia presenta una variación amplia de valores entre 0 y 9, lo que refleja una distribución dispersa de amplitudes. Se aprecia que algunos dígitos corresponden a valores bajos, como el 0 en la posición 1 y el 3 en la posición 8, mientras que otros alcanzan picos altos como el 9 en la posición 7 y el 8 en la posición 5. Esta variabilidad genera una señal irregular, sin un patrón periódico claro, lo cual es coherente con que los datos provienen de un número personal.

En la segunda figura se representa la señal del código estudiantil [5, 6, 0, 0, 7, 1, 8]. Aquí se observa una secuencia más corta en comparación con la de la cédula, con valores que oscilan entre 0 y 8. Se destaca la presencia de dos ceros consecutivos en las posiciones 2 y 3, lo que genera un intervalo de reposo, mientras que en la última posición aparece el valor máximo de 8, que produce un pico al final de la señal. Este comportamiento evidencia un contraste marcado entre los valores bajos y los altos, generando saltos bruscos entre dígitos consecutivos.

Comparando ambas señales, se puede decir que la de la cédula es más extensa y presenta mayor variedad de amplitudes distribuidas de manera irregular, mientras que la del código estudiantil es más breve y concentra los valores más significativos en ciertos puntos específicos. Este contraste entre extensión y variabilidad permite diferenciar el comportamiento de cada secuencia y resulta útil para el análisis de señales discretas.

<img width="1189" height="494" alt="image" src="https://github.com/user-attachments/assets/60655c62-1c79-46ed-a549-fe78086cea67" />
                [3]Graficas de cedula y codigo-3er integrante.

                
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
                 [4]Union (convolucion) datos del 1er integrante.
                 
 En la gráfica se observa el resultado de la convolución, la cual genera una señal más larga que las secuencias originales, extendiéndose desde la muestra 0 hasta la 15. La amplitud alcanza un máximo cercano a 90 en la posición 9, mostrando cómo la superposición de los valores de entrada refuerza ciertos puntos. La señal presenta un crecimiento progresivo hasta el pico máximo, seguido de un descenso gradual, lo que refleja el comportamiento característico de la convolución: acumulación inicial, punto de máxima coincidencia y luego disminución al agotarse las superposiciones.

<img width="850" height="474" alt="image" src="https://github.com/user-attachments/assets/e6a32589-71ae-463f-9549-c8623d90e92d" />
                 [5]Union (convolucion) datos del 2do integrante.

En la gráfica se aprecia el resultado de la convolución, que se extiende desde la muestra 0 hasta la 15. La señal muestra un crecimiento progresivo en amplitud hasta alcanzar su pico máximo cercano a 160 en la posición 8, lo que indica el punto de mayor coincidencia entre las secuencias originales. Después de este valor, la amplitud comienza a descender de manera gradual, reflejando la disminución de las superposiciones. El comportamiento general evidencia una forma triangular asimétrica, típica de la convolución, donde la energía se concentra en el centro de la señal.

<img width="850" height="474" alt="image" src="https://github.com/user-attachments/assets/c5b4d4e5-25cf-4518-a8cd-c1e5c879e3e9" />
                 [6]Union (convolucion) datos del 3er integrante.

En la gráfica se observa el resultado de la convolución, que abarca desde la muestra 0 hasta la 15. La señal inicia con valores bajos y va aumentando progresivamente hasta alcanzar su máximo cercano a 120 en la posición 9, lo que corresponde al punto de mayor solapamiento entre las secuencias. Posteriormente, la amplitud disminuye de manera gradual hasta llegar nuevamente a valores bajos. Este comportamiento refleja el patrón típico de la convolución: crecimiento inicial, un pico central marcado y un descenso simétrico hacia el final de la señal.

# PARTE B: Correlacion Cruzada.
En esta segunda parte del laboratorio, trabajamos con dos señales sinusoidales generadas matemáticamente para estudiar su relación y similitud utilizando el concepto de correlación cruzada.

<img width="800" height="563" alt="image" src="https://github.com/user-attachments/assets/3b6f941f-139a-454f-8676-27f20dc77625" />
 Imagen X1 [n] =cos (π/4n)
muestra un coseno discreto con periodo de 8 muestras, que oscila entre -1 y 1 de forma simétrica, repitiéndose cada 8 puntos.


<img width="800" height="563" alt="image" src="https://github.com/user-attachments/assets/5bf5e23a-046c-4bbb-8c76-89815506e177" />

Imagen X2 [n] =sin (π/4n)


<img width="768" height="563" alt="image" src="https://github.com/user-attachments/assets/307132b5-4c06-4e9d-82d1-90995dabc543" />
Imagen [Correlacion cruzada] 

La correlación cruzada confirma que las señales seno y coseno son ortogonales (desplazadas un cuarto de ciclo), mostrando picos positivos y negativos en función del retardo 
k

<img width="787" height="563" alt="image" src="https://github.com/user-attachments/assets/afd15713-871a-41df-ab99-9fcb737cdb66" />
Imagen correlacion cruzada normalizada 






# conclusiones
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
