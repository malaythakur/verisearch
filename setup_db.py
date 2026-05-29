"""Create documents table in Railway PostgreSQL."""
import psycopg2

conn = psycopg2.connect("postgresql://postgres:CGKHxaIGUXGCuuahbmXULSUmsKqymImX@zephyr.proxy.rlwy.net:23427/railway")
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS documents (
    document_id TEXT PRIMARY KEY,
    source_url TEXT NOT NULL UNIQUE,
    version INTEGER NOT NULL DEFAULT 1,
    content_hash TEXT NOT NULL,
    cleaned_text TEXT NOT NULL,
    credibility_score FLOAT DEFAULT 0.5,
    ai_generated_likelihood FLOAT DEFAULT 0.5,
    visible BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
""")

cur.execute("CREATE INDEX IF NOT EXISTS idx_documents_url ON documents(source_url);")

conn.commit()
print("Documents table created successfully!")

# Verify
cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public';")
tables = cur.fetchall()
print(f"Tables in database: {[t[0] for t in tables]}")

cur.close()
conn.close()
