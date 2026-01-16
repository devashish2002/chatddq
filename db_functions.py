from langchain_core.documents import Document#from langchain.schema import Document
import psycopg2
from sklearn.cluster import MiniBatchKMeans
import numpy as np

def load_unique_document_names():
    conn = psycopg2.connect('postgresql://postgres:GWoRQdEdN3I8N6S7@db.hfgczcibyczibduiqxcb.supabase.co:5432/postgres')
    cur = conn.cursor()

    cur.execute("""
        SELECT DISTINCT
            cmetadata->>'source' AS source
        FROM langchain_pg_embedding
        WHERE cmetadata ? 'source'
        ORDER BY source
    """)

    rows = cur.fetchall()
    cur.close()
    conn.close()

    return [row[0] for row in rows if row[0]]


def load_all_documents_from_db():
    conn = psycopg2.connect('postgresql://postgres:GWoRQdEdN3I8N6S7@db.hfgczcibyczibduiqxcb.supabase.co:5432/postgres')
    cur = conn.cursor()

    cur.execute("""
        SELECT document, cmetadata
        FROM langchain_pg_embedding
    """)

    rows = cur.fetchall()
    cur.close()
    conn.close()

    return [
        Document(
            page_content=row[0],
            metadata=row[1] or {}
        )
        for row in rows
    ]

def parse_embedding(embedding):
    if isinstance(embedding, str):
        embedding = embedding.strip("[]")
        embedding = np.fromstring(embedding, sep=",", dtype=np.float32)
    else:
        embedding = np.array(embedding, dtype=np.float32)
    return embedding


def load_embeddings_for_selected_docs(conn, selected_sources):

    conn = psycopg2.connect('postgresql://postgres:GWoRQdEdN3I8N6S7@db.hfgczcibyczibduiqxcb.supabase.co:5432/postgres')

    if not selected_sources:
        return []

    query = """
        SELECT document, embedding
        FROM langchain_pg_embedding
        WHERE cmetadata->>'source' = ANY(%s)
    """

    with conn.cursor() as cur:
        cur.execute(query, (selected_sources,))
        rows = cur.fetchall()

    docs = []
    for document, embedding in rows:
        docs.append({
            "text": document,
            "embedding": parse_embedding(embedding)
        })

    return docs


def cluster_embeddings(docs, n_clusters=10, random_state=42):
    embeddings = np.vstack([d["embedding"] for d in docs])

    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        batch_size=256
    )

    labels = kmeans.fit_predict(embeddings)

    clusters = {}
    for doc, label in zip(docs, labels):
        clusters.setdefault(label, []).append(doc)

    return clusters
