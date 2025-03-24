import pandas as pd

# Citește fișierul Parquet
df = pd.read_parquet("veridion_entity_resolution_challenge.snappy.parquet")

# Afișează informații despre dataset
print(df.info())  # Tipurile de date și numărul de rânduri/coloane
print(df.head(10))  # Primele 10 rânduri pentru a vedea structura datelor
