import pandas as pd
import networkx as nx
import unicodedata
import itertools
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# 1. Citesc fișierul Parquet
df = pd.read_parquet("veridion_entity_resolution_challenge.snappy.parquet")

# 2. Selectez coloanele relevante
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

for col in selected_cols:
    df_selected[col] = df_selected[col].apply(normalize_text)

# 4. Salvez rezultatul pentru următorii pași (opțional)
df_selected.to_csv("normalized_company_data.csv", index=False)

# ==============================
# 5. Similaritate între perechi
# ==============================

def text_similarity(a, b):
    a = str(a) if pd.notnull(a) else ""
    b = str(b) if pd.notnull(b) else ""
    if not a or not b:
        return 0.0
    vec = TfidfVectorizer().fit_transform([a, b])
    return cosine_similarity(vec[0:1], vec[1:2])[0][0]

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

def compute_similarity_pairs(df, threshold=0.85, group_by="main_country", max_rows=500):
    sample_df = df.head(max_rows).copy()
    blocks = sample_df.groupby(group_by)
    
    similar_pairs = []

    for _, group in blocks:
        for i, j in itertools.combinations(group.itertuples(index=False), 2):
            sim_score = calculate_similarity(i._asdict(), j._asdict())
            if sim_score >= threshold:
                similar_pairs.append({
                    "record_id_1": i.record_id,
                    "record_id_2": j.record_id,
                    "similarity_score": sim_score
                })

    return pd.DataFrame(similar_pairs)

# 6. Rulez pe un subset de 500 companii și salvez perechile similare
similar_pairs_df = compute_similarity_pairs(df_selected, threshold=0.85, max_rows=500)
similar_pairs_df.to_csv("similarity_pairs.csv", index=False)
# ==============================
# 7.  Gruparea companiilor duplicate
# ==============================
def assign_group_ids(similarity_df, df_entities):
    """
    similarity_df: DataFrame cu coloanele record_id_1, record_id_2, similarity_score
    df_entities: DataFrame cu toate companiile normalizate (inclusiv record_id)
    """

    # 1. Construim graful
    G = nx.Graph()
    G.add_edges_from(similarity_df[["record_id_1", "record_id_2"]].values)

    # 2. Extragem componentele conexe (fiecare e un grup de companii duplicate)
    connected_components = list(nx.connected_components(G))

    # 3. Creăm un dicționar {record_id: group_id}
    record_to_group = {}
    for group_id, component in enumerate(connected_components):
        for record_id in component:
            record_to_group[record_id] = group_id

    # 4. Adăugăm coloana `company_group_id` în DataFrame-ul cu companii
    df_entities["company_group_id"] = df_entities["record_id"].map(record_to_group)

    # 5. Companiile necuplate (fără perechi similare) le punem în grupuri unice
    next_id = len(connected_components)
    df_entities["company_group_id"] = df_entities["company_group_id"].fillna(
        df_entities["record_id"] + next_id
    ).astype(int)

    return df_entities
df_selected = assign_group_ids(similar_pairs_df, df_selected)
df_selected.to_csv("final_companies_with_groups.csv", index=False)
