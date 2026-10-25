"""
Database Persistence & Star Schema Synchronization Layer:
Kaiser Permanente Member Experience Intelligence Platform

Provides DatabaseSyncManager to synchronize enriched Star Schema and RAG staging tables
into a local or remote relational database (SQLite default, MySQL / PostgreSQL toggle)
via SQLAlchemy for Power BI DirectQuery / live refreshes.
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import pandas as pd
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.engine import Engine

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import BASE_DIR, DATA_PROCESSED_DIR, OUTPUT_FILES

DEFAULT_SQLITE_PATH = BASE_DIR / "data" / "kaiser_analytics.db"


class DatabaseSyncManager:
    """
    Manages database connectivity and synchronization for the Star Schema
    and RAG staging tables using SQLAlchemy.
    """

    TABLE_PK_MAP = {
        "Dim_Member_Enriched": "Member_ID",
        "Dim_Agent": "Agent_ID",
        "Dim_Date": "Date_Key",
        "Fact_Interactions_Enriched": "Interaction_ID",
        "RAG_Corpus_Staging": "Interaction_ID"
    }

    TABLE_INDEXES = {
        "Fact_Interactions_Enriched": [
            "idx_fact_member_id (Member_ID)",
            "idx_fact_agent_id (Agent_ID)",
            "idx_fact_date_key (Date_Key)",
            "idx_fact_channel (Contact_Channel)",
            "idx_fact_escalation (Escalation_Flag)",
            "idx_fact_topic (Dominant_Topic)"
        ],
        "Dim_Member_Enriched": [
            "idx_dim_mem_region (Region)",
            "idx_dim_mem_segment (Member_Segment)"
        ],
        "Dim_Agent": [
            "idx_dim_agent_dept (Department)",
            "idx_dim_agent_queue (Queue)"
        ],
        "Dim_Date": [
            "idx_dim_date_year_month (Year, Month)"
        ],
        "RAG_Corpus_Staging": [
            "idx_rag_topic (Topic)",
            "idx_rag_dept (Department)",
            "idx_rag_channel (Channel)"
        ]
    }

    def __init__(self, db_url: Optional[str] = None):
        """
        Initializes the DatabaseSyncManager.
        
        Args:
            db_url: SQLAlchemy connection string. Defaults to environment variable
                    KAISER_DB_URL, or local SQLite database at data/kaiser_analytics.db.
        """
        if db_url:
            self.db_url = db_url
        else:
            env_url = os.environ.get("KAISER_DB_URL")
            if env_url:
                self.db_url = env_url
            else:
                DEFAULT_SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
                self.db_url = f"sqlite:///{DEFAULT_SQLITE_PATH.as_posix()}"

        self.engine: Engine = self._create_engine()

    def _create_engine(self) -> Engine:
        """Creates and configures the SQLAlchemy engine."""
        if self.db_url.startswith("sqlite"):
            return create_engine(
                self.db_url,
                connect_args={"check_same_thread": False},
                pool_pre_ping=True
            )
        else:
            return create_engine(
                self.db_url,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True
            )

    def test_connection(self) -> Tuple[bool, str]:
        """Tests database connectivity."""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True, f"Successfully connected to database at: {self._safe_db_url()}"
        except Exception as e:
            return False, f"Connection failed: {str(e)}"

    def _safe_db_url(self) -> str:
        """Returns database URL with masked passwords for logging."""
        if "@" in self.db_url:
            prefix, rest = self.db_url.split("://", 1)
            creds, host_db = rest.split("@", 1)
            if ":" in creds:
                user, _ = creds.split(":", 1)
                return f"{prefix}://{user}:****@{host_db}"
        return self.db_url

    def sync_star_schema_to_db(self, data_dir: Optional[Path] = None) -> Dict[str, Any]:
        """
        Loads all Star Schema CSVs and RAG staging tables into the relational database.
        Creates primary key / foreign key indexing for optimal Power BI DirectQuery reads.
        
        Args:
            data_dir: Directory containing processed CSVs. Defaults to data/processed.
            
        Returns:
            Dictionary with sync status, row counts, and duration.
        """
        source_dir = Path(data_dir) if data_dir else DATA_PROCESSED_DIR
        start_time = time.time()
        print(f"\n========================================================")
        print(f" DATABASE PERSISTENCE SYNC: STAR SCHEMA -> RELATIONAL DB")
        print(f" Target: {self._safe_db_url()}")
        print(f" Source: {source_dir}")
        print(f"========================================================")

        results = {"success": True, "tables_synced": {}, "errors": []}

        # Table file mapping
        table_files = {
            "Dim_Member_Enriched": OUTPUT_FILES["dim_member"],
            "Dim_Agent": OUTPUT_FILES["dim_agent"],
            "Dim_Date": OUTPUT_FILES["dim_date"],
            "Fact_Interactions_Enriched": OUTPUT_FILES["fact_interactions"],
            "RAG_Corpus_Staging": OUTPUT_FILES["rag_corpus"],
        }

        for table_name, filename in table_files.items():
            file_path = source_dir / filename
            if not file_path.exists():
                fallback_path = BASE_DIR / filename
                if fallback_path.exists():
                    file_path = fallback_path
                else:
                    err = f"File not found: {filename} in {source_dir} or {BASE_DIR}"
                    print(f"  [ERROR] {err}")
                    results["errors"].append(err)
                    results["success"] = False
                    continue

            try:
                print(f"  * Reading {filename} ...", end=" ")
                df = pd.read_csv(file_path)

                # Ensure critical schema consistency
                if table_name == "Fact_Interactions_Enriched":
                    if "Predicted_Escalation_Class" not in df.columns and "Predicted_Escalation_Prob" in df.columns:
                        df["Predicted_Escalation_Class"] = (df["Predicted_Escalation_Prob"] >= 0.50).astype(int)
                    if "Date_Key" not in df.columns and "Interaction_Date" in df.columns:
                        df["Date_Key"] = pd.to_datetime(df["Interaction_Date"]).dt.strftime("%Y-%m-%d")

                elif table_name == "Dim_Date":
                    if "Date_Key" not in df.columns and "Date" in df.columns:
                        df["Date_Key"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")

                # Ingest to SQL
                df.to_sql(
                    name=table_name,
                    con=self.engine,
                    if_exists="replace",
                    index=False,
                    chunksize=1000
                )
                print(f"Loaded {len(df):,} rows -> Database Table [{table_name}]")
                results["tables_synced"][table_name] = len(df)

                # Create indexes
                self._create_table_indexes(table_name)

            except Exception as e:
                err = f"Failed to sync table {table_name}: {str(e)}"
                print(f"\n  [ERROR] {err}")
                results["errors"].append(err)
                results["success"] = False

        duration = time.time() - start_time
        results["duration_seconds"] = round(duration, 2)
        print(f"========================================================")
        print(f" DATABASE SYNC COMPLETED in {duration:.2f}s | Success: {results['success']}")
        print(f"========================================================\n")
        return results

    def _create_table_indexes(self, table_name: str):
        """Creates primary and secondary performance indexes on synced tables."""
        is_sqlite = self.db_url.startswith("sqlite")
        with self.engine.begin() as conn:
            # 1. Primary key index
            pk_col = self.TABLE_PK_MAP.get(table_name)
            if pk_col:
                pk_idx_name = f"idx_pk_{table_name}_{pk_col}"
                try:
                    conn.execute(text(f"CREATE INDEX IF NOT EXISTS {pk_idx_name} ON {table_name} ({pk_col})"))
                except Exception:
                    pass

            # 2. Secondary foreign key / analytical indexes
            indexes = self.TABLE_INDEXES.get(table_name, [])
            for idx_def in indexes:
                idx_name, cols = idx_def.split(" ", 1)
                sql = f"CREATE INDEX IF NOT EXISTS {idx_name}_{table_name} ON {table_name} {cols}"
                try:
                    conn.execute(text(sql))
                except Exception:
                    pass

    def query_table(self, table_name: str, limit: int = 100, where_clause: Optional[str] = None) -> pd.DataFrame:
        """
        Executes a safe SELECT query against a synced table.
        
        Args:
            table_name: Target table name.
            limit: Maximum rows to retrieve.
            where_clause: Optional SQL WHERE condition (without 'WHERE' keyword).
        """
        inspector = inspect(self.engine)
        if table_name not in inspector.get_table_names():
            return pd.DataFrame()

        sql = f"SELECT * FROM {table_name}"
        if where_clause:
            sql += f" WHERE {where_clause}"
        sql += f" LIMIT {int(limit)}"

        with self.engine.connect() as conn:
            df = pd.read_sql(text(sql), conn)
        return df

    def get_table_counts(self) -> Dict[str, int]:
        """Returns row count for all tables present in the database."""
        counts = {}
        inspector = inspect(self.engine)
        tables = inspector.get_table_names()
        with self.engine.connect() as conn:
            for t in tables:
                try:
                    res = conn.execute(text(f"SELECT COUNT(*) FROM {t}"))
                    counts[t] = res.scalar() or 0
                except Exception:
                    counts[t] = 0
        return counts


def run_database_sync(db_url: Optional[str] = None, data_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Helper entry point for pipeline and CLI invocation."""
    manager = DatabaseSyncManager(db_url=db_url)
    return manager.sync_star_schema_to_db(data_dir=data_dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Kaiser Permanente Database Sync Manager")
    parser.add_argument("--db-url", type=str, default=None, help="SQLAlchemy Database URL")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to processed CSV directory")
    args = parser.parse_args()

    sync_results = run_database_sync(db_url=args.db_url, data_dir=args.data_dir)
    if sync_results["success"]:
        sys.exit(0)
    else:
        sys.exit(1)
