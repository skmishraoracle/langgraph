"""Base implementation for Oracle checkpoint saver in LangGraph."""
import json
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, cast

import oracledb
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,  # Changed from BaseCheckpointer
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    get_checkpoint_id,
)
from langgraph.checkpoint.serde.base import SerializerProtocol
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.serde.types import ChannelProtocol

# Initialize Oracle Client for thick mode if needed
oracledb.init_oracle_client()

# Type hint for connection objects
Conn = Union[oracledb.Connection, oracledb.SessionPool]


def get_connection(conn: Conn) -> oracledb.Connection:
    """Get a connection from a connection or pool.

    Args:
        conn: Either a connection or connection pool

    Returns:
        oracledb.Connection: A usable database connection
    """
    if isinstance(conn, oracledb.SessionPool):
        return conn.acquire()
    return conn


class BaseOracleCheckpointer(
        BaseCheckpointSaver[str]):  # Changed to match PostgreSQL
    """Base class for Oracle checkpoint savers.

    This provides common functionality for both synchronous and asynchronous
    Oracle checkpoint savers. It handles serialization, deserialization,
    and Oracle-specific SQL constructs.

    Attributes:
        serde: Serializer/deserializer for checkpoint data
        _table_prefix: Prefix for database tables
    """

    # Add JsonPlusSerializer like PostgreSQL implementation
    jsonplus_serde = JsonPlusSerializer()

    # Simplified migration approach for initial Oracle port
    # Individual migrations for better error handling
    MIGRATIONS = [
        """CREATE TABLE checkpoint_migrations (v NUMBER(10) PRIMARY KEY)""",

        """CREATE TABLE checkpoints (
        thread_id VARCHAR2(2000) NOT NULL,
        checkpoint_ns VARCHAR2(2000) NOT NULL DEFAULT '',
        checkpoint_id VARCHAR2(2000) NOT NULL,
        type VARCHAR2(2000),
        checkpoint CLOB NOT NULL,
        metadata CLOB NOT NULL DEFAULT '{}',
        parent_checkpoint_id VARCHAR2(2000),
        PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
    )""",

        # Note: Changed to match PostgreSQL - added version field
        """CREATE TABLE checkpoint_blobs (
        thread_id VARCHAR2(2000) NOT NULL,
        checkpoint_ns VARCHAR2(2000) NOT NULL DEFAULT '',
        channel VARCHAR2(2000) NOT NULL,
        version VARCHAR2(2000) NOT NULL,
        type VARCHAR2(2000) NOT NULL,
        blob BLOB,
        PRIMARY KEY (thread_id, checkpoint_ns, channel, version)
    )""",

        """CREATE TABLE checkpoint_writes (
        thread_id VARCHAR2(2000) NOT NULL,
        checkpoint_ns VARCHAR2(2000) NOT NULL DEFAULT '',
        checkpoint_id VARCHAR2(2000) NOT NULL,
        task_id VARCHAR2(2000) NOT NULL,
        idx NUMBER(10) NOT NULL,
        channel VARCHAR2(2000) NOT NULL,
        type VARCHAR2(2000),
        blob BLOB NOT NULL,
        task_path VARCHAR2(2000) DEFAULT '',
        PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
    )""",

        """CREATE INDEX checkpoints_thread_id_idx ON checkpoints(thread_id)""",

        """CREATE INDEX checkpoint_blobs_thread_id_idx ON checkpoint_blobs(thread_id)""",

        """CREATE INDEX checkpoint_writes_thread_id_idx ON checkpoint_writes(thread_id)"""
    ]

    # Oracle uses MERGE INTO for upserts
    # Updated to match PostgreSQL - added version parameter
    UPSERT_CHECKPOINT_BLOBS_SQL = """
        MERGE INTO checkpoint_blobs target
        USING (SELECT :1 AS thread_id, :2 AS checkpoint_ns, :3 AS channel,
                      :4 AS version, :5 AS type, :6 AS blob FROM dual) source
        ON (target.thread_id = source.thread_id AND
            target.checkpoint_ns = source.checkpoint_ns AND
            target.channel = source.channel AND
            target.version = source.version)
        WHEN MATCHED THEN
            UPDATE SET type = source.type, blob = source.blob
        WHEN NOT MATCHED THEN
            INSERT (thread_id, checkpoint_ns, channel, version, type, blob)
            VALUES (source.thread_id, source.checkpoint_ns, source.channel,
                    source.version, source.type, source.blob)
    """

    UPSERT_CHECKPOINTS_SQL = """
        MERGE INTO checkpoints target
        USING (SELECT :1 AS thread_id, :2 AS checkpoint_ns, :3 AS checkpoint_id,
                :4 AS parent_checkpoint_id, :5 AS checkpoint, :6 AS metadata FROM dual) source
        ON (target.thread_id = source.thread_id AND
            target.checkpoint_ns = source.checkpoint_ns AND
            target.checkpoint_id = source.checkpoint_id)
        WHEN MATCHED THEN
            UPDATE SET parent_checkpoint_id = source.parent_checkpoint_id,
                        checkpoint = source.checkpoint,
                        metadata = source.metadata
        WHEN NOT MATCHED THEN
            INSERT (thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, checkpoint, metadata)
            VALUES (source.thread_id, source.checkpoint_ns, source.checkpoint_id,
                    source.parent_checkpoint_id, source.checkpoint, source.metadata)
    """

    UPSERT_CHECKPOINT_WRITES_SQL = """
        MERGE INTO checkpoint_writes target
        USING (SELECT :1 AS thread_id, :2 AS checkpoint_ns, :3 AS checkpoint_id,
                :4 AS task_id, :5 AS task_path, :6 AS idx, :7 AS channel,
                :8 AS type, :9 AS blob FROM dual) source
        ON (target.thread_id = source.thread_id AND
            target.checkpoint_ns = source.checkpoint_ns AND
            target.checkpoint_id = source.checkpoint_id AND
            target.task_id = source.task_id AND
            target.idx = source.idx)
        WHEN MATCHED THEN
            UPDATE SET channel = source.channel, type = source.type, blob = source.blob,
                        task_path = source.task_path
        WHEN NOT MATCHED THEN
            INSERT (thread_id, checkpoint_ns, checkpoint_id, task_id, task_path, idx, channel, type, blob)
            VALUES (source.thread_id, source.checkpoint_ns, source.checkpoint_id,
                    source.task_id, source.task_path, source.idx, source.channel,
                    source.type, source.blob)
    """

    INSERT_CHECKPOINT_WRITES_SQL = """
        MERGE INTO checkpoint_writes target
        USING (SELECT :1 AS thread_id, :2 AS checkpoint_ns, :3 AS checkpoint_id,
                :4 AS task_id, :5 AS task_path, :6 AS idx, :7 AS channel,
                :8 AS type, :9 AS blob FROM dual) source
        ON (target.thread_id = source.thread_id AND
            target.checkpoint_ns = source.checkpoint_ns AND
            target.checkpoint_id = source.checkpoint_id AND
            target.task_id = source.task_id AND
            target.idx = source.idx)
        WHEN NOT MATCHED THEN
            INSERT (thread_id, checkpoint_ns, checkpoint_id, task_id, task_path, idx, channel, type, blob)
            VALUES (source.thread_id, source.checkpoint_ns, source.checkpoint_id,
                    source.task_id, source.task_path, source.idx, source.channel,
                    source.type, source.blob)
    """

    # Oracle query for selecting checkpoints with nested data
    # Updated for version field matching and parent_checkpoint_id for
    # pending_sends
    SELECT_SQL = """
        SELECT
            c.thread_id,
            c.checkpoint_id,
            c.checkpoint_ns,
            c.parent_checkpoint_id,
            c.checkpoint,
            c.metadata,
            (
                SELECT CAST(COLLECT(
                        CAST(MULTISET(
                            SELECT bl.channel, bl.type, bl.blob
                            FROM JSON_TABLE(c.checkpoint, '$.channel_versions.*' COLUMNS
                                (channel_key PATH '$', channel_value PATH '$')) jt
                            INNER JOIN checkpoint_blobs bl
                                ON bl.thread_id = c.thread_id
                                AND bl.checkpoint_ns = c.checkpoint_ns
                                AND bl.channel = jt.channel_key
                                AND bl.version = jt.channel_value
                        ) AS t_nested_table)
                    ) AS t_nested_table)
                    FROM DUAL
                ) AS channel_values,
                (
                    SELECT CAST(COLLECT(
                        CAST(MULTISET(
                            SELECT cw.task_id, cw.channel, cw.type, cw.blob
                            FROM checkpoint_writes cw
                            WHERE cw.thread_id = c.thread_id
                                AND cw.checkpoint_ns = c.checkpoint_ns
                                AND cw.checkpoint_id = c.checkpoint_id
                            ORDER BY cw.task_id, cw.idx
                        ) AS t_nested_table)
                    ) AS t_nested_table)
                    FROM DUAL
                ) AS pending_writes,
                (
                    SELECT CAST(COLLECT(
                        CAST(MULTISET(
                            SELECT cw.type, cw.blob
                            FROM checkpoint_writes cw
                            WHERE cw.thread_id = c.thread_id
                                AND cw.checkpoint_ns = c.checkpoint_ns
                                AND cw.checkpoint_id = c.parent_checkpoint_id
                                AND cw.channel = :tasks_channel
                            ORDER BY cw.task_path, cw.task_id, cw.idx
                        ) AS t_nested_table)
                    ) AS t_nested_table)
                    FROM DUAL
                ) AS pending_sends
            FROM checkpoints c
        """

    def __init__(
        self,
        serde: SerializerProtocol,
        table_prefix: str = "",
    ) -> None:
        """Initialize BaseOracleCheckpointer.

        Args:
            serde: The serializer/deserializer to use
            table_prefix: Optional prefix for database tables
        """
        super().__init__(serde=serde)
        self._table_prefix = table_prefix

    def _dump_checkpoint(self, checkpoint: Checkpoint) -> dict[str, Any]:
        """Dump checkpoint data to dict.

        Args:
            checkpoint: The checkpoint to serialize

        Returns:
            dict: Dictionary representation of checkpoint
        """
        # Match PostgreSQL implementation
        return {**checkpoint, "pending_sends": []}

    def _dump_metadata(self, metadata: CheckpointMetadata) -> str:
        """Dump metadata to JSON string.

        Args:
            metadata: The metadata to serialize

        Returns:
            str: JSON string representation of metadata
        """
        # Match PostgreSQL implementation
        serialized_metadata = self.jsonplus_serde.dumps(metadata)
        # NOTE: we're using JSON serializer (not msgpack), so we need to remove
        # null characters before writing
        return serialized_metadata.decode().replace("\\u0000", "")

    def _load_checkpoint(
        self,
        checkpoint: dict[str, Any],
        channel_values: Optional[List],
        pending_sends: Optional[List],
    ) -> Checkpoint:
        """Load checkpoint from database representation.

        Args:
            checkpoint: Dictionary of checkpoint data
            channel_values: Oracle nested table of channel values
            pending_sends: Oracle nested table of pending sends

        Returns:
            Checkpoint: The reconstructed checkpoint
        """
        # Match PostgreSQL implementation
        result = cast(Checkpoint, {
            **checkpoint,
            "pending_sends": [
                self.serde.loads_typed((c.decode(), b)) for c, b in pending_sends or []
            ],
            "channel_values": self._load_blobs(channel_values),
        })
        return result

    def _load_blobs(
        self, blob_values: Optional[List]
    ) -> dict[str, Any]:
        """Load blobs from database representation.

        Args:
            blob_values: List of blob values from database

        Returns:
            dict: Dictionary of channel -> value
        """
        # Added to match PostgreSQL implementation
        if not blob_values:
            return {}

        # Type-safe conversion of blob_values
        typed_blob_values: list[tuple[bytes, bytes, bytes]] = []
        for item in blob_values:
            if isinstance(item, (list, tuple)) and len(item) >= 3:
                typed_blob_values.append(
                    (item[0] if isinstance(
                        item[0], bytes) else str(
                        item[0]).encode(), item[1] if isinstance(
                        item[1], bytes) else str(
                        item[1]).encode(), item[2] if isinstance(
                        item[2], bytes) else b''))

        return {
            k.decode(): self.serde.loads_typed((t.decode(), v))
            for k, t, v in typed_blob_values
            if t.decode() != "empty"
        }

    def _load_metadata(self, metadata_json: str) -> CheckpointMetadata:
        """Load metadata from JSON string.

        Args:
            metadata_json: JSON string of metadata

        Returns:
            CheckpointMetadata: The reconstructed metadata
        """
        # Match PostgreSQL implementation
        return self.jsonplus_serde.loads(
            self.jsonplus_serde.dumps(
                json.loads(metadata_json)))

    def _load_writes(
            self, pending_writes: Optional[List]) -> List[Tuple[str, str, Any]]:
        """Load pending writes from database representation.

        Args:
            pending_writes: Oracle nested table of pending writes

        Returns:
            List[Tuple[str, str, Any]]: List of (task_id, channel, value) tuples
        """
        # Updated to match PostgreSQL implementation
        return (
            [
                (
                    tid.decode(),
                    channel.decode(),
                    self.serde.loads_typed((t.decode(), v)),
                )
                for tid, channel, t, v in pending_writes
            ]
            if pending_writes
            else []
        )

    def _dump_writes(
        self,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
        task_id: str,
        task_path: str,
        writes: Sequence[Tuple[str, Any]],
    ) -> List[Tuple]:
        """Prepare writes for database storage.

        Args:
            thread_id: Thread identifier
            checkpoint_ns: Checkpoint namespace
            checkpoint_id: Checkpoint identifier
            task_id: Task identifier
            task_path: Task path
            writes: Sequence of (channel, value) pairs

        Returns:
            List[Tuple]: Parameters for database insert/update
        """
        # Updated to match PostgreSQL implementation
        return [
            (
                thread_id,
                checkpoint_ns,
                checkpoint_id,
                task_id,
                task_path,
                # Changed to match PostgreSQL
                WRITES_IDX_MAP.get(channel, idx),
                channel,
                *self.serde.dumps_typed(value),
            )
            for idx, (channel, value) in enumerate(writes)
        ]

    def _dump_blobs(
        self,
        thread_id: str,
        checkpoint_ns: str,
        values: Dict[str, Any],
        versions: ChannelVersions,
    ) -> List[Tuple]:
        """Prepare blob data for database storage.

        Args:
            thread_id: Thread identifier
            checkpoint_ns: Checkpoint namespace
            values: Dictionary of channel values
            versions: Channel versions

        Returns:
            List[Tuple]: Parameters for database insert/update
        """
        # Updated to match PostgreSQL implementation with version field
        if not versions:
            return []

        return [
            (
                thread_id,
                checkpoint_ns,
                k,
                cast(str, ver),  # Added version parameter
                *(self.serde.dumps_typed(values[k])
                  if k in values else ("empty", None)),
            )
            # Changed to iterate over items() instead of keys
            for k, ver in versions.items()
        ]

    def _search_where(
        self,
        config: Optional[RunnableConfig],
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
    ) -> Tuple[str, Tuple]:
        """Generate WHERE clause for checkpoint queries.

        Args:
            config: Configuration with thread_id
            filter: Additional filtering criteria
            before: Only include checkpoints before this one

        Returns:
            Tuple[str, Tuple]: WHERE clause and parameters
        """
        conditions = []
        args = []

        if config is not None:
            # Type-safe access to configurable
            configurable = config.get("configurable", {})
            if not configurable:
                raise ValueError("Config must contain 'configurable' key")

            thread_id = configurable["thread_id"]
            conditions.append("thread_id = :1")
            args.append(thread_id)

            checkpoint_ns = configurable.get("checkpoint_ns", "")
            conditions.append("checkpoint_ns = :2")
            args.append(checkpoint_ns)

            # Add checkpoint_id condition if present in config
            if checkpoint_id := get_checkpoint_id(config):
                conditions.append("checkpoint_id = :3")
                args.append(checkpoint_id)

        if before is not None:
            # Type-safe access to before configurable
            before_configurable = before.get("configurable", {})
            if not before_configurable:
                raise ValueError(
                    "Before config must contain 'configurable' key")

            # Adjust parameter index if checkpoint_id was added
            idx = len(args) + 1
            before_id = before_configurable["checkpoint_id"]
            conditions.append(f"checkpoint_id < :{idx}")
            args.append(before_id)

        if filter:
            # Oracle JSON path expressions are different from PostgreSQL
            # Need to use proper Oracle JSON_EXISTS and JSON_VALUE functions
            idx = len(args) + 1
            for key, value in filter.items():
                # Use Oracle's JSON path syntax with $ prefix
                json_path = f"$.{key}"
                conditions.append(
                    f"JSON_VALUE(metadata, '{json_path}') = :{idx}")
                args.append(str(value))
                idx += 1

        if conditions:
            return "WHERE " + " AND ".join(conditions), tuple(args)

        return "", tuple()

    def get_next_version(
            self,
            current: Optional[str],
            channel: ChannelProtocol) -> str:
        """Get the next version for a channel.

        Args:
            current: Current version
            channel: Channel protocol

        Returns:
            str: Next version
        """
        # Added to match PostgreSQL implementation
        if current is None:
            current_v = 0
        elif isinstance(current, int):
            current_v = current
        else:
            current_v = int(current.split(".")[0])
        next_v = current_v + 1
        next_h = random.random()
        return f"{next_v:032}.{next_h:016}"
