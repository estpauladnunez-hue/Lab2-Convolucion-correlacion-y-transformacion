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

# Graficar resultados por integrante
for integ in integrantes:
    cod = np.array(integ["codigo"])
    ced = np.array(integ["cedula"])

    # Union= convolución entre cédula y código
    union = np.convolve(cod, ced)

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

    # Grafica de la Unión de Cédula y Código
    plt.figure(figsize=(10, 5))
    plt.title(f"Parte A - {integ['nombre']} | Unión (Convolución)", fontsize=15, fontweight="bold")
    plt.stem(range(len(union)), union, basefmt="k", linefmt="r-", markerfmt="ro")
    plt.xlabel("n (muestras)")
    plt.ylabel("Amplitud")
    plt.xticks(range(len(union)))  # Mostrar todos los puntos claramente
    plt.grid(True)
    plt.show()


#-------------------PARTE B - CORRELACIÓN CRUZADA-------------------

# Definicion de las señales
Ts = 1.25e-3      # Periodo de muestreo (1.25 ms)
f = 100           # Frecuencia en Hz
n = np.arange(0, 9)
w = 2 * np.pi * f * Ts

x1 = np.cos(w * n)
x2 = np.sin(w * n)

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

# Mostrar en consola el máximo coeficiente de correlación
max_corr = np.max(r12_norm)
k_max = lags[np.argmax(r12_norm)]
print(f"\nMáxima correlación normalizada = {max_corr:.3f} en k = {k_max}")
