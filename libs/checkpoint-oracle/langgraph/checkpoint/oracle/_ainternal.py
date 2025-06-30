"""Shared async utility functions for the Oracle checkpoint & storage classes."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Union

import oracledb

# Type for Oracle async connections
Conn = Union[oracledb.AsyncConnection, oracledb.AsyncConnectionPool]


@asynccontextmanager
async def get_connection(
        conn: Conn) -> AsyncIterator[oracledb.AsyncConnection]:
    """Get an async connection from a connection object or pool.

    Args:
        conn: Either an Oracle async connection or an async connection pool.

    Yields:
        An Oracle async connection.
    """
    if isinstance(conn, oracledb.AsyncConnection):
        yield conn
    elif isinstance(conn, oracledb.AsyncConnectionPool):
        connection = await conn.acquire()
        try:
            yield connection
        finally:
            await conn.release(connection)
    else:
        raise TypeError(f"Invalid connection type: {type(conn)}")
