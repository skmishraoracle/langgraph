"""Implementation of LangGraph checkpoint saver using Oracle Database."""
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any, Dict, Optional, Tuple

import oracledb
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
)
from langgraph.checkpoint.serde.base import SerializerProtocol
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

Conn = oracledb.Connection  # For type hints


class OracleCheckpointer(BaseCheckpointSaver):
    """Oracle-based checkpoint saver implementation.

    This checkpoint saver stores checkpoints in an Oracle database using the
    oracledb Python driver. It is compatible with Oracle Database 19c and later.

    Args:
        conn: The Oracle connection object, or a connection pool.
        schema: The schema to use for the tables. Defaults to the current user's schema.
        serde: The serializer to use for serialization/deserialization.
            Defaults to JsonPlusSerializer.
        table_prefix: Prefix to use for the tables. Defaults to "LANGGRAPH_CHECKPOINT_".

    Note:
        You should call the setup() method once to create the necessary tables
        before using this checkpointer.
    """

    conn: Conn
    _lock: threading.RLock
    _schema: str
    _table_prefix: str

    def __init__(
        self,
        conn: Conn,
        *,
        schema: Optional[str] = None,
        serde: Optional[SerializerProtocol] = None,
        table_prefix: str = "LANGGRAPH_CHECKPOINT_",
    ) -> None:
        """Initialize the Oracle checkpointer.

        Args:
            conn: Oracle connection object or connection pool
            schema: Database schema to use (defaults to current user's schema)
            serde: Serializer to use (defaults to JsonPlusSerializer)
            table_prefix: Prefix for the tables (defaults to "LANGGRAPH_CHECKPOINT_")
        """
        super().__init__(serde=serde or JsonPlusSerializer())
        self.conn = conn
        self._lock = threading.RLock()
        self._schema = schema or ''
        self._table_prefix = table_prefix

    @property
    def _checkpoints_table(self) -> str:
        """Get the fully qualified checkpoints table name."""
        if self._schema:
            return f"{self._schema}.{self._table_prefix}CHECKPOINTS"
        return f"{self._table_prefix}CHECKPOINTS"

    @property
    def _writes_table(self) -> str:
        """Get the fully qualified writes table name."""
        if self._schema:
            return f"{self._schema}.{self._table_prefix}WRITES"
        return f"{self._table_prefix}WRITES"

    @classmethod
    @contextmanager
    def from_conn_string(
        cls,
        conn_string: str,
        *,
        schema: Optional[str] = None,
        serde: Optional[SerializerProtocol] = None,
        table_prefix: str = "LANGGRAPH_CHECKPOINT_",
    ) -> Iterator["OracleCheckpointer"]:
        """Create an OracleCheckpointer from a connection string.

        The connection will be closed when the context manager exits.

        Args:
            conn_string: Oracle connection string (username/password@host/service)
            schema: Database schema to use (defaults to current user's schema)
            serde: Serializer to use (defaults to JsonPlusSerializer)
            table_prefix: Prefix for the tables (defaults to "LANGGRAPH_CHECKPOINT_")

        Yields:
            OracleCheckpointer: The checkpointer instance

        """
        conn = None
        try:
            conn = oracledb.connect(conn_string)
            yield cls(
                conn,
                schema=schema,
                serde=serde,
                table_prefix=table_prefix,
            )
        finally:
            if conn is not None:
                conn.close()

    @classmethod
    @contextmanager
    def from_parameters(
        cls,
        *,
        user: str,
        password: str,
        dsn: str,
        schema: Optional[str] = None,
        serde: Optional[SerializerProtocol] = None,
        table_prefix: str = "LANGGRAPH_CHECKPOINT_",
    ) -> Iterator["OracleCheckpointer"]:
        """Create an OracleCheckpointer from connection parameters.

        The connection will be closed when the context manager exits.

        Args:
            user: Database username
            password: Database password
            dsn: Database connection string (host/service)
            schema: Database schema to use (defaults to current user's schema)
            serde: Serializer to use (defaults to JsonPlusSerializer)
            table_prefix: Prefix for the tables (defaults to "LANGGRAPH_CHECKPOINT_")

        Yields:
            OracleCheckpointer: The checkpointer instance

        """
        conn = None
        try:
            conn = oracledb.connect(user=user, password=password, dsn=dsn)
            yield cls(
                conn,
                schema=schema,
                serde=serde,
                table_prefix=table_prefix,
            )
        finally:
            if conn is not None:
                conn.close()

    def setup(self) -> None:
        """Create the necessary tables for the checkpointer.

        This method should be called once before using the checkpointer.
        It creates the tables needed to store the checkpoints and writes.

        """
        with self._lock:
            cursor = self.conn.cursor()
            try:
                # Create checkpoints table if it doesn't exist
                cursor.execute(
                    f"""
                    BEGIN
                      EXECUTE IMMEDIATE '
                        CREATE TABLE {self._checkpoints_table} (
                          thread_id VARCHAR2(512) NOT NULL,
                          thread_ts VARCHAR2(64),
                          checkpoint_id VARCHAR2(64) NOT NULL,
                          checkpoint CLOB NOT NULL,
                          metadata CLOB NOT NULL,
                          created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                          PRIMARY KEY (thread_id, checkpoint_id)
                        )';
                    EXCEPTION
                      WHEN OTHERS THEN
                        IF SQLCODE = -955 THEN NULL;
                        ELSE RAISE;
                        END IF;
                    END;
                  """
                )

                # Create writes table if it doesn't exist
                cursor.execute(
                    f"""
                    BEGIN
                      EXECUTE IMMEDIATE '
                        CREATE TABLE {self._writes_table} (
                          thread_id VARCHAR2(512) NOT NULL,
                          checkpoint_id VARCHAR2(64) NOT NULL,
                          node_name VARCHAR2(512) NOT NULL,
                          write_idx NUMBER NOT NULL,
                          write_data CLOB NOT NULL,
                          task_id VARCHAR2(64) NOT NULL,
                          task_path VARCHAR2(512) DEFAULT NULL,
                          created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                          PRIMARY KEY (thread_id, checkpoint_id, node_name, write_idx)
                        )';
                    EXCEPTION
                      WHEN OTHERS THEN
                        IF SQLCODE = -955 THEN NULL;
                        ELSE RAISE;
                        END IF;
                    END;
                  """
                )

                # Create an index on the checkpoints table for faster lookups
                cursor.execute(
                    f"""
                    BEGIN
                      EXECUTE IMMEDIATE '
                        CREATE INDEX {self._table_prefix}CK_THREAD_IDX ON
                          {self._checkpoints_table} (thread_id, created_at)';
                    EXCEPTION
                      WHEN OTHERS THEN
                        IF SQLCODE = -955 THEN NULL;
                        ELSE RAISE;
                        END IF;
                    END;
                  """
                )

                # Create indexes on the writes table
                cursor.execute(
                    f"""
                    BEGIN
                      EXECUTE IMMEDIATE '
                        CREATE INDEX {self._table_prefix}WR_THREAD_CK_IDX ON
                          {self._writes_table} (thread_id, checkpoint_id)';
                    EXCEPTION
                      WHEN OTHERS THEN
                        IF SQLCODE = -955 THEN NULL;
                        ELSE RAISE;
                        END IF;
                    END;
                  """
                )

                self.conn.commit()
            finally:
                cursor.close()

    def drop_tables(self) -> None:
        """Drop the tables created by the checkpointer.

        This method can be used to clean up the database after the checkpointer
        is no longer needed.

        """
        with self._lock:
            cursor = self.conn.cursor()
            try:
                cursor.execute(
                    f"""
                    BEGIN
                      EXECUTE IMMEDIATE 'DROP TABLE {self._writes_table}';
                    EXCEPTION
                      WHEN OTHERS THEN
                        IF SQLCODE = -942 THEN NULL;
                        ELSE RAISE;
                        END IF;
                    END;
                  """
                )
                cursor.execute(
                    f"""
                    BEGIN
                      EXECUTE IMMEDIATE 'DROP TABLE {self._checkpoints_table}';
                    EXCEPTION
                      WHEN OTHERS THEN
                        IF SQLCODE = -942 THEN NULL;
                        ELSE RAISE;
                        END IF;
                      END;
                  """
                )
                self.conn.commit()
            finally:
                cursor.close()

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Get the checkpoint tuple for a thread.

        Args:
            config: A RunnableConfig containing the thread_id and optionally thread_ts.

        Returns:
            Optional[CheckpointTuple]: The checkpoint tuple, or None if not found.
        """
        # Type-safe access to configurable
        configurable = config.get("configurable", {})
        if not configurable:
            return None

        thread_id = configurable["thread_id"]
        thread_ts = configurable.get("thread_ts")

        with self._lock:
            cursor = self.conn.cursor()
            try:
                if thread_ts is not None:
                    # Get specific checkpoint based on timestamp
                    query = f"""
                      SELECT checkpoint, metadata, checkpoint_id
                      FROM {self._checkpoints_table}
                      WHERE thread_id = :thread_id AND thread_ts = :thread_ts
                    """
                    cursor.execute(query, thread_id=thread_id, thread_ts=thread_ts)
                else:
                    # Get the latest checkpoint for this thread
                    query = f"""
                      SELECT checkpoint, metadata, checkpoint_id
                      FROM {self._checkpoints_table}
                      WHERE thread_id = :thread_id
                      ORDER BY created_at DESC
                      FETCH FIRST 1 ROWS ONLY
                    """
                    cursor.execute(query, thread_id=thread_id)

                row = cursor.fetchone()
                if not row:
                    return None

                checkpoint_json, metadata_json, checkpoint_id = row

                # Deserialize the checkpoint and metadata
                checkpoint = self.serde.loads(checkpoint_json)
                metadata = self.serde.loads(metadata_json)

                # Fetch pending writes for this checkpoint
                query = f"""
                            SELECT node_name, write_idx, write_data
                            FROM {self._writes_table}
                            WHERE thread_id = :thread_id AND checkpoint_id = :checkpoint_id
                            ORDER BY node_name, write_idx
                        """
                cursor.execute(query, thread_id=thread_id, checkpoint_id=checkpoint_id)

                # Convert to the expected format for CheckpointTuple
                pending_writes = []
                for node_name, write_idx, write_data in cursor:
                    # Deserialize the write data
                    write = self.serde.loads(write_data)
                    pending_writes.append((node_name, write_idx, write))

                return CheckpointTuple(
                    config={
                        "configurable": {
                            "thread_id": thread_id,
                            "thread_ts": checkpoint["ts"],
                            "checkpoint_id": checkpoint_id,
                        }
                    },
                    checkpoint=checkpoint,
                    metadata=metadata,
                    pending_writes=pending_writes,
                )
            finally:
                cursor.close()

    def list(
        self,
        config: Optional[RunnableConfig] = None,
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints for a thread.

        Args:
            config: A RunnableConfig containing the thread_id.
            filter: Optional filter criteria for metadata.
            before: Optional config to get checkpoints before a specific checkpoint.
            limit: Optional limit on the number of checkpoints to return.

        Returns:
            Iterator[CheckpointTuple]: An iterator of checkpoint tuples.
        """
        filter = filter or {}
        # Type-safe access to configurable
        thread_id = None
        if config is not None:
            configurable = config.get("configurable", {})
            if configurable:
                thread_id = configurable["thread_id"]

        with self._lock:
            cursor = self.conn.cursor()
            try:
                # Build the query based on the parameters
                query_parts = [f"SELECT checkpoint, metadata, checkpoint_id, thread_id FROM {self._checkpoints_table}"]
                params = {}

                where_clauses = []
                if thread_id is not None:
                    where_clauses.append("thread_id = :thread_id")
                    params["thread_id"] = thread_id

                # Add before condition if specified
                if before is not None:
                    before_configurable = before.get("configurable", {})
                    if before_configurable:
                        before_ts = before_configurable.get("thread_ts")
                        if before_ts is not None:
                            where_clauses.append(
                                "created_at < (SELECT created_at FROM {0} WHERE thread_ts = :before_ts)".format(
                                    self._checkpoints_table))
                            params["before_ts"] = before_ts

                if where_clauses:
                    query_parts.append("WHERE " + " AND ".join(where_clauses))

                query_parts.append("ORDER BY created_at DESC")

                if limit is not None:
                    query_parts.append(f"FETCH FIRST {limit} ROWS ONLY")

                query = " ".join(query_parts)
                cursor.execute(query, **params)

                # Fetch all matching checkpoints
                rows = cursor.fetchall()
                for checkpoint_json, metadata_json, checkpoint_id, row_thread_id in rows:
                    checkpoint = self.serde.loads(checkpoint_json)
                    metadata = self.serde.loads(metadata_json)

                    # Filter based on metadata if filter is provided
                    if filter and not all(
                        metadata.get(k) == v for k, v in filter.items()
                    ):
                        continue

                    # Fetch pending writes for this checkpoint
                    writes_query = f"""
                        SELECT node_name, write_idx, write_data
                        FROM {self._writes_table}
                        WHERE thread_id = :thread_id AND checkpoint_id = :checkpoint_id
                        ORDER BY node_name, write_idx
                    """
                    cursor.execute(
                        writes_query,
                        thread_id=row_thread_id,
                        checkpoint_id=checkpoint_id)

                    # Convert to the expected format for CheckpointTuple
                    pending_writes = []
                    for node_name, write_idx, write_data in cursor:
                        write = self.serde.loads(write_data)
                        pending_writes.append((node_name, write_idx, write))

                    yield CheckpointTuple(
                        config={
                            "configurable": {
                                "thread_id": row_thread_id,
                                "thread_ts": checkpoint["ts"],
                                "checkpoint_id": checkpoint_id,
                            }
                        },
                        checkpoint=checkpoint,
                        metadata=metadata,
                        pending_writes=pending_writes,
                    )
            finally:
                cursor.close()

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Store a checkpoint.

        Args:
            config: A RunnableConfig containing the thread_id.
            checkpoint: The checkpoint data to store.
            metadata: Metadata associated with the checkpoint.
            new_versions: New channel versions.

        Returns:
            RunnableConfig: Updated config with thread_ts and checkpoint_id.
        """
        # Type-safe access to configurable
        configurable = config.get("configurable", {})
        if not configurable:
            raise ValueError("Config must contain 'configurable' key")

        configurable = configurable.copy()
        thread_id = configurable.get("thread_id")
        if thread_id is None:
            raise ValueError("thread_id is required")

        # Get or generate checkpoint ID
        checkpoint_id = configurable.get("checkpoint_id") or get_checkpoint_id(config)

        with self._lock:
            cursor = self.conn.cursor()
            try:
                # Serialize checkpoint and metadata
                checkpoint_json = self.serde.dumps(checkpoint)
                metadata_json = self.serde.dumps(metadata)

                # Insert or update the checkpoint
                query = f"""
                        MERGE INTO {self._checkpoints_table} t
                        USING (SELECT :thread_id AS thread_id, :checkpoint_id AS checkpoint_id FROM dual) s
                        ON (t.thread_id = s.thread_id AND t.checkpoint_id = s.checkpoint_id)
                        WHEN MATCHED THEN
                            UPDATE SET
                                thread_ts = :thread_ts,
                                checkpoint = :checkpoint,
                                metadata = :metadata,
                                created_at = CURRENT_TIMESTAMP
                        WHEN NOT MATCHED THEN
                            INSERT (thread_id, thread_ts, checkpoint_id, checkpoint, metadata)
                            VALUES (:thread_id, :thread_ts, :checkpoint_id, :checkpoint, :metadata)
                        """
                cursor.execute(
                    query,
                    thread_id=thread_id,
                    thread_ts=checkpoint["ts"],
                    checkpoint_id=checkpoint_id,
                    checkpoint=checkpoint_json,
                    metadata=metadata_json,
                )
                self.conn.commit()
            finally:
                cursor.close()

        # Return updated config
        configurable["thread_ts"] = checkpoint["ts"]
        configurable["checkpoint_id"] = checkpoint_id
        return {"configurable": configurable}

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Store writes for a checkpoint.

        Args:
            config: A RunnableConfig containing the thread_id, thread_ts, and checkpoint_id.
            writes: A sequence of tuples (node_name, data) to store.
            task_id: Task ID associated with the writes.
            task_path: Optional task path.
        """
        # Type-safe access to configurable
        configurable = config.get("configurable", {})
        if not configurable:
            raise ValueError("Config must contain 'configurable' key")

        thread_id = configurable["thread_id"]
        checkpoint_id = configurable["checkpoint_id"]

        if not writes:
            return

        with self._lock:
            cursor = self.conn.cursor()
            try:
                # Group writes by node
                writes_by_node: Dict[str, list] = {}
                for node_name, data in writes:
                    if node_name not in writes_by_node:
                        writes_by_node[node_name] = []
                    writes_by_node[node_name].append(data)

                # Insert writes to database
                for node_name, data_list in writes_by_node.items():
                    # Get the current write index for this node/checkpoint
                    query = f"""
                        SELECT NVL(MAX(write_idx) + 1, 0)
                        FROM {self._writes_table}
                        WHERE thread_id = :thread_id
                          AND checkpoint_id = :checkpoint_id
                          AND node_name = :node_name
                    """
                    cursor.execute(
                        query,
                        thread_id=thread_id,
                        checkpoint_id=checkpoint_id,
                        node_name=node_name,
                    )
                    write_idx = cursor.fetchone()[0]

                    # Insert each write
                    for i, data in enumerate(data_list):
                        # Serialize the write data
                        write_data = self.serde.dumps(data)

                        query = f"""
                            INSERT INTO {self._writes_table} (
                                thread_id, checkpoint_id, node_name, write_idx,
                                write_data, task_id, task_path
                            ) VALUES (
                                :thread_id, :checkpoint_id, :node_name, :write_idx,
                                :write_data, :task_id, :task_path
                            )
                        """
                        cursor.execute(
                            query,
                            thread_id=thread_id,
                            checkpoint_id=checkpoint_id,
                            node_name=node_name,
                            write_idx=write_idx + i,
                            write_data=write_data,
                            task_id=task_id,
                            task_path=task_path,
                        )

                self.conn.commit()
            finally:
                cursor.close()

    def delete_thread(
        self,
        thread_id: str,
    ) -> None:
        """Delete all checkpoints and writes associated with a specific thread ID.

        Args:
            thread_id: The thread ID whose checkpoints should be deleted.
        """
        with self._lock:
            cursor = self.conn.cursor()
            try:
                # Delete writes first (due to foreign key constraints)
                cursor.execute(
                    f"DELETE FROM {self._writes_table} WHERE thread_id = :thread_id",
                    thread_id=thread_id,
                )
                
                # Delete checkpoints
                cursor.execute(
                    f"DELETE FROM {self._checkpoints_table} WHERE thread_id = :thread_id",
                    thread_id=thread_id,
                )
                
                self.conn.commit()
            finally:
                cursor.close()


# Import async checkpointer
from .aio import AsyncOracleCheckpointer

__all__ = [
    "OracleCheckpointer",
    "AsyncOracleCheckpointer",
    "Conn",
]
