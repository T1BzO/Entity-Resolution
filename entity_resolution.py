import pandas as pd
import networkx as nx
import unicodedata
import itertools
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time

start_time = time.time()

# 1. Citesc fișierul Parquet
print("1. Încărcarea fișierului .parquet...")
df = pd.read_parquet("veridion_entity_resolution_challenge.snappy.parquet")

# 2. Selectez coloanele relevante
print("2. Selectarea coloanelor relevante...")
selected_cols = [
    "company_name",
    "company_legal_names",
    "company_commercial_names",
    "website_domain",
    "main_country",
    "main_city",
    "main_postcode",
    "emails",
    "phone_numbers",
    "linkedin_url",
    "facebook_url"
]

df_selected = df[selected_cols].copy()
df_selected["record_id"] = df_selected.index

# 3. Normalizez textul
def normalize_text(val):
    if pd.isnull(val):
        return ""
    val = str(val).strip().lower()
    val = unicodedata.normalize("NFKD", val)
    return "".join([c for c in val if not unicodedata.combining(c)])

print("3. Normalizarea textului în câmpurile relevante...")
for col in selected_cols:
    df_selected[col] = df_selected[col].apply(normalize_text)

# 4. Înlocuirea valorilor NaN cu un șir gol (pentru câmpurile esențiale)
print("4. Înlocuirea valorilor NaN cu un șir gol...")
df_selected.fillna("", inplace=True)

# 5. Calculul similarității între perechi
def text_similarity(a, b):
    a = str(a).strip() if pd.notnull(a) else ""
    b = str(b).strip() if pd.notnull(b) else ""

    if not a or not b or len(a.split()) < 1 or len(b.split()) < 1:
        return 0.0

    try:
        vec = TfidfVectorizer().fit_transform([a, b])
        return cosine_similarity(vec[0:1], vec[1:2])[0][0]
    except ValueError:
        return 0.0

def calculate_similarity(row1, row2, weights=None):
    if weights is None:
        weights = {
            "company_name": 0.3,
            "website_domain": 0.25,
            "main_city": 0.15,
            "emails": 0.15,
            "linkedin_url": 0.15
        }

    score = 0.0
    for field, weight in weights.items():
        score += weight * text_similarity(row1[field], row2[field])

    return round(score, 3)

# 6. Gruparea companiilor duplicate
def compute_similarity_pairs_blocked(df, threshold=0.85, blocking_keys=["main_country", "main_city"], max_block_size=300):
    print(f"6. Procesarea datelor pe blocuri folosind {len(df)} înregistrări...")
    similar_pairs = []

    grouped = df.groupby(blocking_keys)
    print(f"   - Total blocuri: {len(grouped)}")

    for _, group in grouped:
        if len(group) < 2 or len(group) > max_block_size:
            continue

        for i, j in itertools.combinations(group.itertuples(index=False), 2):
            sim_score = calculate_similarity(i._asdict(), j._asdict())
            if sim_score >= threshold:
                similar_pairs.append({
                    "record_id_1": i.record_id,
                    "record_id_2": j.record_id,
                    "similarity_score": sim_score
                })

    print(f"   - Finalizat procesul pentru {len(similar_pairs)} perechi similare.")
    return pd.DataFrame(similar_pairs)

# 7. Rulez și salvez perechile similare
print("7. Calcularea perechilor similare...")
similar_pairs_df = compute_similarity_pairs_blocked(df_selected, threshold=0.85)
similar_pairs_df.to_csv("similarity_pairs.csv", index=False)

# 8. Gruparea finală a companiilor duplicate
def assign_group_ids(similarity_df, df_entities):
    print("8. Gruparea finală a companiilor duplicate folosind NetworkX...")
    G = nx.Graph()
    G.add_edges_from(similarity_df[["record_id_1", "record_id_2"]].values)

    connected_components = list(nx.connected_components(G))

    record_to_group = {}
    for group_id, component in enumerate(connected_components):
        for record_id in component:
            record_to_group[record_id] = group_id

    df_entities["company_group_id"] = df_entities["record_id"].map(record_to_group)

    # Filtrăm grupurile care au doar 1 element
    df_entities = df_entities[df_entities["company_group_id"].map(df_entities["company_group_id"].value_counts()) > 1]

    next_id = len(connected_components)
    df_entities.loc[:, "company_group_id"] = df_entities["company_group_id"].fillna(df_entities["record_id"] + next_id)
    return df_entities

print("9. Aplicarea grupării finale...")
df_selected = assign_group_ids(similar_pairs_df, df_selected)
df_selected.to_csv("final_companies_with_groups.csv", index=False)

print("Procesul a fost finalizat cu succes!")

end_time = time.time()
execution_time = end_time - start_time

print(f"Timpul de execuție al scriptului este: {execution_time} secunde")