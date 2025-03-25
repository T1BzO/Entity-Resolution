# Entity Resolution Project

## Descriere

Acest proiect se concentrează pe **identificarea și gruparea entităților duplicate** dintr-un set mare de date de companii. Scopul este de a aplica metode de **deduplicare** pentru a identifica și grupa aceleași entități (companii) dintr-un fișier care conține informații despre diverse companii.

Am utilizat **Python** și tehnici de preprocesare a datelor, inclusiv tehnici de **similaritate** bazate pe `TF-IDF`, `cosine similarity` și **algoritmi de clustering** pentru a grupa companiile duplicate. De asemenea, am optimizat soluția pentru scalabilitate, folosind blocuri și grupuri pe baza unor atribute esențiale, cum ar fi `main_country` și `main_city`.

## Pași pentru rulare

### 1. Instalarea dependențelor

Asigură-te că ai instalate următoarele librării:


pip install pandas networkx scikit-learn tqdm

##

### 2. Cum să rulezi scriptul
După ce ai configurat mediul tău Python cu dependințele necesare, rulează scriptul principal:

bash
Copiază
python entity_resolution.py

##

### 3. Descrierea scriptului
entity_resolution.py: Scriptul principal care realizează următoarele:

Citește datele din fișierul .parquet

Normalizarea câmpurilor esențiale (nume companie, domeniu, oraș)

Calculul scorurilor de similaritate între companii

Gruparea companiilor duplicate folosind algoritmi de clustering (NetworkX)

Salvarea rezultatelor într-un fișier .csv

##

### 4. Intrare / Ieșire
Intrare:
Fișierul de date de intrare este un fișier .parquet care conține informații despre companii.

Ieșire:
Fișierele .csv rezultate:

normalized_company_data.csv: Fișier cu datele normalizate.

similarity_pairs.csv: Perechi de companii similare cu scorurile lor de similaritate.

final_companies_with_groups.csv: Companiile grupate în funcție de similitudine.

Pași implementați
Preprocesare și Normalizare

Am normalizat datele pentru a asigura că nu există diferențe de formatare în câmpurile cheie (company_name, website_domain, etc.).

Calcularea Similarității

Am utilizat TfidfVectorizer și cosine_similarity pentru a calcula similitudinea între companii pe baza câmpurilor relevante.

Clustering

Am utilizat networkx pentru a construi un graf neorientat și a identifica componentele conexe (grupuri de duplicate).

Optimizare pentru scalabilitate

Am implementat un sistem de blocking pe main_country și main_city, pentru a reduce numărul de comparații inutile între companii din locații diferite.

Concluzii și direcții viitoare
Ce am realizat până acum:
Am finalizat un pipeline de deduplicare folosind metode tradiționale de preprocesare și similaritate.

Am reușit să grupăm companiile duplicate eficient, chiar și cu un număr mare de entități.

Direcția viitoare:
Machine Learning:

Vom folosi tehnici de învățare automată pentru a îmbunătăți exactitatea deduplicării, folosind un neural network siamese.

Vom antrena un model bazat pe embeddings din texte (de exemplu, folosind SentenceTransformer).

Optimizare:

##
Vom implementa soluții de paralelizare și salvare progresivă pentru a îmbunătăți performanța pe seturi mari de date.




