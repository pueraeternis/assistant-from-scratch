# tools/sql_query.py

import logging
import sqlite3
from pathlib import Path
from typing import Any

from core.base_tool import BaseTool

# --- Configuration ---
DB_FILE = Path("data/company.db")

logger = logging.getLogger(__name__)


class SQLQueryTool(BaseTool):
    """
    Tool for executing SQL queries against the internal company database (SQLite).
    """

    name = "SQLQueryTool"
    description = (
        "Use this tool to execute an SQL query against the company's SQLite database. "
        "The database contains information about employees and departments. "
        "Provide a valid SQL query as input using the 'query' parameter."
    )

    def _run(self, **kwargs: Any) -> str:
        """
        Executes an SQL query and returns the result as a Markdown table.
        Expects 'query' as a named argument.
        """
        query = kwargs.get("query")
        if not query or not isinstance(query, str):
            return "Error: The 'query' argument is missing or is not a string for SQLQueryTool."

        if not DB_FILE.exists():
            return f"Error: Database file not found at '{DB_FILE}'. Please run the setup script first."

        logger.info("Executing SQL query: %s", query)

        try:
            # Using context manager for safe connection handling
            with sqlite3.connect(DB_FILE) as conn:
                cursor = conn.cursor()
                cursor.execute(query)
                rows = cursor.fetchall()

                # Handle empty result (e.g., UPDATE or empty SELECT)
                if not rows:
                    if conn.total_changes > 0:
                        return f"Query executed successfully. {conn.total_changes} rows affected."
                    return "Query executed successfully, no results returned."

                # Retrieve column names
                column_names = [desc[0] for desc in cursor.description]

                # --- Format result as Markdown table ---
                header = "| " + " | ".join(column_names) + " |"
                separator = "| " + " | ".join(["---"] * len(column_names)) + " |"

                body_rows = ["| " + " | ".join(map(str, row)) + " |" for row in rows]

                return "\n".join([header, separator] + body_rows)

        except sqlite3.Error as e:
            logger.error("SQLite error while executing query: %s", e)
            return f"SQLite Error: {e}. Please check your query syntax."
        except Exception as e:
            logger.error("Unexpected error occurred in SQLQueryTool: %s", e)
            return f"An unexpected error occurred: {e}"
