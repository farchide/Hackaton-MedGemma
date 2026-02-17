"""Storage package for the Digital Twin Tumor system.

Exports the primary storage backends and a factory function to select the
appropriate backend based on configuration.
"""

from __future__ import annotations

from typing import Any

from digital_twin_tumor.storage.audit import AuditLogger
from digital_twin_tumor.storage.sqlite_backend import SQLiteBackend

__all__ = [
    "AuditLogger",
    "SQLiteBackend",
    "get_storage_backend",
]


def get_storage_backend(config: dict[str, Any]) -> SQLiteBackend | Any:
    """Return the appropriate storage backend based on configuration.

    The ``config`` dictionary is inspected for a ``"backend"`` key:

    - ``"postgres"`` -- returns a :class:`PostgresBackend` using connection
      parameters from the config (``host``, ``port``, ``database``, ``user``,
      ``password``).  Requires ``psycopg`` to be installed.
    - Any other value (or absent) -- returns a :class:`SQLiteBackend` using
      the ``"db_path"`` key (defaults to ``.cache/digital_twin.db``).

    Parameters
    ----------
    config:
        Dictionary with storage configuration.  Expected keys depend on the
        chosen backend.

    Returns
    -------
    SQLiteBackend | PostgresBackend
        The configured storage backend instance.

    Raises
    ------
    ImportError
        If ``"postgres"`` backend is requested but ``psycopg`` is not
        installed.
    """
    backend_type = config.get("backend", "sqlite")

    if backend_type == "postgres":
        from digital_twin_tumor.domain.models import DatabaseConfig
        from digital_twin_tumor.storage.postgres_client import PostgresBackend

        db_config = DatabaseConfig(
            host=config.get("host", "localhost"),
            port=config.get("port", 5432),
            database=config.get("database", "digital_twin"),
            user=config.get("user", ""),
            password=config.get("password", ""),
        )
        return PostgresBackend(db_config)

    db_path = config.get("db_path", ".cache/digital_twin.db")
    return SQLiteBackend(db_path=db_path)
