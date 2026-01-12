import psycopg2
import streamlit as st
from langchain_core.documents import Document#from langchain.schema import Document

def load_unique_document_names():
    conn = psycopg2.connect(st.secrets["PG_CONNECTION_STRING"])
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
    conn = psycopg2.connect(st.secrets["PG_CONNECTION_STRING"])
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
