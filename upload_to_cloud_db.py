"""
Dual-Environment Cloud Database Persistence Engine:
Kaiser Permanente Member Experience Intelligence Platform (Capstone Project 3)

Uploads processed Star Schema tables to a cloud PostgreSQL database (e.g., Neon Postgres, AWS RDS, Azure Database)
with automated lowercasing, connection pooling, SSL enforcement, primary key constraints, and foreign key indexing
for Power BI DirectQuery and live report refreshes.
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.engine import Engine

# Set project root
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import BASE_DIR, DATA_PROCESSED_DIR, OUTPUT_FILES


def load_env_file(env_path: Path):
    """Simple parser to load key=value environment variables from .env or .env.local without external dependencies."""
    if not env_path.exists():
        return
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip('"').strip("'")
                    if key and key not in os.environ:
                        os.environ[key] = val
    except Exception as e:
        print(f"  [WARN] Could not parse env file {env_path.name}: {e}")


# Load local environment configuration if present
for candidate_env in [BASE_DIR / ".env", BASE_DIR / ".env.local"]:
    load_env_file(candidate_env)


class CloudDatabaseUploader:
    """
    Manages secure cloud PostgreSQL connectivity, table migration, and index optimization.
    """

    TABLES_TO_SYNC = [
        ("dim_member_enriched", "Dim_Member_Enriched.csv", "member_id"),
        ("dim_agent", "Dim_Agent.csv", "agent_id"),
        ("dim_date", "Dim_Date.csv", "date_key"),
        ("fact_interactions_enriched", "Fact_Interactions_Enriched.csv", "interaction_id"),
        ("rag_corpus_staging", "RAG_Corpus_Staging.csv", "interaction_id")
    ]

    INDEX_DEFINITIONS = {
        "fact_interactions_enriched": [
            ("idx_fact_member_id", "member_id"),
            ("idx_fact_agent_id", "agent_id"),
            ("idx_fact_date_key", "date_key"),
            ("idx_fact_channel", "contact_channel"),
            ("idx_fact_escalation", "escalation_flag"),
            ("idx_fact_topic", "dominant_topic")
        ],
        "dim_member_enriched": [
            ("idx_dim_mem_region", "region"),
            ("idx_dim_mem_segment", "member_segment")
        ],
        "dim_agent": [
            ("idx_dim_agent_dept", "department"),
            ("idx_dim_agent_queue", "queue")
        ],
        "dim_date": [
            ("idx_dim_date_year_month", "year, month")
        ],
        "rag_corpus_staging": [
            ("idx_rag_topic", "topic"),
            ("idx_rag_dept", "department")
        ]
    }

    def __init__(self, connection_url: Optional[str] = None):
        """
        Initializes cloud database connection engine.
        Reads credentials from connection_url, environment variables, or default Neon/Postgres settings.
        """
        self.db_url = self._resolve_db_url(connection_url)
        self.engine: Engine = self._create_ssl_engine()

    def _resolve_db_url(self, explicit_url: Optional[str]) -> str:
        """Determines the appropriate database connection string."""
        if explicit_url:
            return explicit_url

        # Check standard cloud environment variables (Neon, Heroku, Supabase, Render)
        for env_var in [
            "DATABASE_URL", "NEON_DATABASE_URL", "POSTGRES_URL",
            "PGDATABASE_URL", "KAISER_DB_URL"
        ]:
            val = os.environ.get(env_var)
            if val:
                # Ensure driver is postgresql+psycopg2
                if val.startswith("postgres://"):
                    val = val.replace("postgres://", "postgresql+psycopg2://", 1)
                elif val.startswith("postgresql://") and not val.startswith("postgresql+psycopg2://"):
                    val = val.replace("postgresql://", "postgresql+psycopg2://", 1)
                return val

        # Assemble from discrete parameters or fallback defaults
        user = os.environ.get("DB_USER", "postgres")
        password = os.environ.get("DB_PASS", "postgres")
        host = os.environ.get("DB_HOST", "localhost")
        port = os.environ.get("DB_PORT", "5432")
        database = os.environ.get("DB_NAME", "kaiser_analytics")

        return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"

    def _create_ssl_engine(self) -> Engine:
        """Configures SQLAlchemy engine with connection pooling and SSL mode."""
        connect_args = {}
        # If connecting to remote cloud host (like Neon, AWS, Azure), enforce SSL
        if "localhost" not in self.db_url and "127.0.0.1" not in self.db_url:
            connect_args["sslmode"] = "require"

        return create_engine(
            self.db_url,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
            connect_args=connect_args
        )

    def mask_url(self) -> str:
        """Masks database password for secure logging."""
        if "@" in self.db_url:
            prefix, rest = self.db_url.split("://", 1)
            creds, host_db = rest.split("@", 1)
            user = creds.split(":", 1)[0] if ":" in creds else creds
            return f"{prefix}://{user}:****@{host_db}"
        return self.db_url

    def test_connection(self) -> Tuple[bool, str]:
        """Validates cloud database connectivity before migration."""
        try:
            with self.engine.connect() as conn:
                res = conn.execute(text("SELECT version();")).scalar()
                return True, f"Connected to PostgreSQL: {res}"
        except Exception as e:
            return False, f"Connection error: {str(e)}"

    def upload_star_schema(self, data_dir: Optional[Path] = None) -> Dict[str, Any]:
        """
        Loads CSV files from data/processed/, lowers table and column names,
        migrates them to cloud PostgreSQL via chunked to_sql, and creates PK/FK indexes.
        """
        source_dir = Path(data_dir) if data_dir else DATA_PROCESSED_DIR
        start_time = time.time()
        print("\n" + "=" * 70)
        print(" KAISER PERMANENTE CLOUD DATABASE PERSISTENCE ENGINE")
        print(f" Target Endpoint: {self.mask_url()}")
        print(f" Source Directory: {source_dir}")
        print("=" * 70)

        # Pre-flight connection test
        connected, msg = self.test_connection()
        if not connected:
            print(f"\n[FATAL] Unable to connect to target cloud database:\n{msg}")
            print("\nPlease ensure your credentials in .env, environment variables, or CLI are correct.")
            return {"success": False, "error": msg}

        print(f"[PRE-FLIGHT CHECK] {msg}\n")
        results = {"success": True, "tables_synced": {}, "errors": []}

        for table_name, csv_filename, pk_col in self.TABLES_TO_SYNC:
            file_path = source_dir / csv_filename
            if not file_path.exists():
                fallback = BASE_DIR / csv_filename
                if fallback.exists():
                    file_path = fallback
                else:
                    err = f"File not found: {csv_filename} in {source_dir} or {BASE_DIR}"
                    print(f"  [ERROR] {err}")
                    results["errors"].append(err)
                    continue

            try:
                print(f"  * Ingesting {csv_filename} ...", end=" ")
                df = pd.read_csv(file_path)

                # Standardize all column names to lowercase for PostgreSQL compatibility
                df.columns = [c.lower() for c in df.columns]

                # Specific field harmonization
                if table_name == "fact_interactions_enriched":
                    if "predicted_escalation_class" not in df.columns and "predicted_escalation_prob" in df.columns:
                        df["predicted_escalation_class"] = (df["predicted_escalation_prob"] >= 0.50).astype(int)
                    if "date_key" not in df.columns and "interaction_date" in df.columns:
                        df["date_key"] = pd.to_datetime(df["interaction_date"]).dt.strftime("%Y-%m-%d")

                elif table_name == "dim_date":
                    if "date_key" not in df.columns and "date" in df.columns:
                        df["date_key"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

                # Ingest to Cloud PostgreSQL using chunked replacement
                df.to_sql(
                    name=table_name.lower(),
                    con=self.engine,
                    if_exists="replace",
                    index=False,
                    chunksize=1000
                )
                print(f"Uploaded {len(df):,} rows -> [{table_name.lower()}]")
                results["tables_synced"][table_name.lower()] = len(df)

                # Apply Primary Key & Indexing
                self._apply_constraints_and_indexes(table_name.lower(), pk_col)

            except Exception as e:
                err = f"Failed to upload table {table_name}: {str(e)}"
                print(f"\n  [ERROR] {err}")
                results["errors"].append(err)
                results["success"] = False

        duration = time.time() - start_time
        results["duration_seconds"] = round(duration, 2)
        print("=" * 70)
        print(f" MIGRATION FINISHED in {duration:.2f}s | Success: {results['success']}")
        print("=" * 70 + "\n")
        return results

    def _apply_constraints_and_indexes(self, table_name: str, pk_col: str):
        """Applies explicit primary key constraints and analytical indexes via SQL."""
        with self.engine.begin() as conn:
            # Add Primary Key constraint
            try:
                pk_name = f"pk_{table_name}_{pk_col}"
                # In PostgreSQL, primary key columns cannot contain NULL
                conn.execute(text(f"ALTER TABLE {table_name} ALTER COLUMN {pk_col} SET NOT NULL;"))
                conn.execute(text(f"ALTER TABLE {table_name} ADD CONSTRAINT {pk_name} PRIMARY KEY ({pk_col});"))
                print(f"      + Added Primary Key constraint on: {pk_col}")
            except Exception as e:
                # If constraint already exists or not supported, continue
                pass

            # Create Indexes
            indexes = self.INDEX_DEFINITIONS.get(table_name, [])
            for idx_name, cols in indexes:
                try:
                    sql = f"CREATE INDEX IF NOT EXISTS {idx_name} ON {table_name} ({cols});"
                    conn.execute(text(sql))
                except Exception:
                    pass
            if indexes:
                print(f"      + Created {len(indexes)} index(es) on {table_name}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Upload Kaiser Permanente Star Schema to Cloud Database")
    parser.add_argument("--db-url", type=str, default=None, help="Explicit PostgreSQL connection URL (e.g. Neon connection string)")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to processed Star Schema CSV folder")
    parser.add_argument("--test-only", action="store_true", help="Test database connection without uploading data")
    args = parser.parse_args()

    uploader = CloudDatabaseUploader(connection_url=args.db_url)
    
    if args.test_only:
        ok, msg = uploader.test_connection()
        print(f"Connection test result: {ok}\n{msg}")
        sys.exit(0 if ok else 1)

    res = uploader.upload_star_schema(data_dir=args.data_dir)
    sys.exit(0 if res.get("success") else 1)


if __name__ == "__main__":
    main()
