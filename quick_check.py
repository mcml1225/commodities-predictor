# quick_check.py
import pandas as pd

xl = pd.ExcelFile("Bloomberg_commodities_prices.xlsx")
df = xl.parse('Monthly', header=None)

print("=== ESTRUCTURA DEL ARCHIVO ===\n")
print(f"Dimensiones: {df.shape}\n")

print("=== PRIMERAS 15 FILAS ===")
for i in range(min(15, df.shape[0])):
    row = df.iloc[i, :10]  # Primeras 10 columnas
    print(f"Fila {i}: {row.tolist()}\n")

print("\n=== ÚLTIMAS 10 FILAS ===")
for i in range(max(0, df.shape[0]-10), df.shape[0]):
    row = df.iloc[i, :5]
    print(f"Fila {i}: {row.tolist()}")