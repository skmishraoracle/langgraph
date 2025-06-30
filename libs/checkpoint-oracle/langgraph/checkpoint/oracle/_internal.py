"""Shared utility functions for the Oracle checkpoint & storage classes."""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Union

import oracledb

# Type for Oracle synchronous connections
Conn = Union[oracledb.Connection, oracledb.SessionPool]


@contextmanager
def get_connection(conn: Conn) -> Iterator[oracledb.Connection]:
    """Get a connection from a connection object or pool.

    Args:
        conn: Either an Oracle connection or a connection pool.

    Yields:
        An Oracle connection.
    """
    if isinstance(conn, oracledb.Connection):
        yield conn
    elif isinstance(conn, oracledb.SessionPool):
        connection = conn.acquire()
        try:
            yield connection
        finally:
            conn.release(connection)
    else:
        raise TypeError(f"Invalid connection type: {type(conn)}")
