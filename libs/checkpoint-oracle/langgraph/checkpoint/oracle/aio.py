"""Implementation of asynchronous LangGraph checkpoint saver using Oracle Database."""
import asyncio
from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import asynccontextmanager
from typing import Any, Optional, cast

import oracledb
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
    get_checkpoint_metadata,
)
from langgraph.checkpoint.serde.base import SerializerProtocol
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from . import _ainternal
from .base import BaseOracleCheckpointer

Conn = _ainternal.Conn


class AsyncOracleCheckpointer(BaseOracleCheckpointer):
    """Asynchronous checkpointer that stores checkpoints in an Oracle database.

    This checkpoint saver stores checkpoints in an Oracle database using the
    oracledb Python driver. It is compatible with Oracle Database 19c and later.

    Args:
        conn: The Oracle connection object, connection pool, or async connection/pool.
        serde: The serializer to use for serialization/deserialization.
            Defaults to JsonPlusSerializer.
        table_prefix: Prefix to use for the tables. Defaults to empty string.

    Note:
        You should call the setup() method once to create the necessary tables
        before using this checkpointer.

    """

    lock: asyncio.Lock

    def __init__(
        self,
        conn: _ainternal.Conn,
        *,
        serde: Optional[SerializerProtocol] = None,
        table_prefix: str = "",
    ) -> None:
        """Initialize the AsyncOracleCheckpointer.

        Args:
            conn: Oracle connection object, connection pool, or async connection/pool
            serde: Serializer to use (defaults to JsonPlusSerializer)
            table_prefix: Prefix for the tables (defaults to "")
        """
        super().__init__(serde=serde or JsonPlusSerializer(), table_prefix=table_prefix)
        self.conn = conn
        self.lock = asyncio.Lock()
        self.loop = asyncio.get_running_loop()

    @classmethod
    @asynccontextmanager
    async def from_conn_string(
        cls,
        conn_string: str,
        *,
        serde: Optional[SerializerProtocol] = None,
        table_prefix: str = "",
    ) -> AsyncIterator["AsyncOracleCheckpointer"]:
        """Create an AsyncOracleCheckpointer from a connection string.

        The connection will be closed when the context manager exits.

        Args:
            conn_string: Oracle connection string (username/password@host/service)
            serde: Serializer to use (defaults to JsonPlusSerializer)
            table_prefix: Prefix for the tables (defaults to "")

        Yields:
            AsyncOracleCheckpointer: The checkpointer instance
        """
        conn = None
        try:
            conn = await oracledb.connect_async(conn_string)
            yield cls(
                conn,
                serde=serde,
                table_prefix=table_prefix,
            )
        finally:
            if conn is not None:
                await conn.close()

    @classmethod
    @asynccontextmanager
    async def from_parameters(
        cls,
        *,
        user: str,
        password: str,
        dsn: str,
        serde: Optional[SerializerProtocol] = None,
        table_prefix: str = "",
    ) -> AsyncIterator["AsyncOracleCheckpointer"]:
        """Create an AsyncOracleCheckpointer from connection parameters.

        The connection will be closed when the context manager exits.

        Args:
            user: Database username
            password: Database password
            dsn: Database connection string (host/service)
            serde: Serializer to use (defaults to JsonPlusSerializer)
            table_prefix: Prefix for the tables (defaults to "")

        Yields:
            AsyncOracleCheckpointer: The checkpointer instance
        """
        conn = None
        try:
            conn = await oracledb.connect_async(user=user, password=password, dsn=dsn)
            yield cls(
                conn,
                serde=serde,
                table_prefix=table_prefix,
            )
        finally:
            if conn is not None:
                await conn.close()

    async def setup(self) -> None:
        """Create the necessary tables for the checkpointer.

        This method should be called once before using the checkpointer.
        It creates the tables needed to store the checkpoints and writes.
        """
        async with self.lock:
            async with _ainternal.get_connection(self.conn) as conn:
                async with conn.cursor() as cursor:
                    await cursor.execute(self.MIGRATIONS[0])
                    await cursor.execute(
                        "SELECT v FROM checkpoint_migrations ORDER BY v DESC FETCH FIRST 1 ROWS ONLY"
                    )
                    row = await cursor.fetchone()
                    if row is None:
                        version = -1
                    else:
                        version = row[0]

                    for v, migration in zip(
                        range(version + 1, len(self.MIGRATIONS)),
                        self.MIGRATIONS[version + 1:],
                    ):
                        await cursor.execute(migration)
                        await cursor.execute(f"INSERT INTO checkpoint_migrations (v) VALUES ({v})")

                    await conn.commit()

    async def alist(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """List checkpoints from the database asynchronously.

        This method retrieves a list of checkpoint tuples from the Oracle database based
        on the provided config. The checkpoints are ordered by checkpoint ID in descending
        order (newest first).

        Args:
            config: Base configuration for filtering checkpoints.
            filter: Additional filtering criteria for metadata.
            before: If provided, only checkpoints before the specified checkpoint ID are returned.
            limit: Maximum number of checkpoints to return.

        Yields:
            AsyncIterator[CheckpointTuple]: An asynchronous iterator of matching checkpoint tuples.
        """
        where, args = self._search_where(config, filter, before)
        query = self.SELECT_SQL + where + " ORDER BY checkpoint_id DESC"
        if limit:
            query += f" FETCH FIRST {limit} ROWS ONLY"

        async with self._cursor() as cur:
            await cur.execute(query, args)
            rows = await cur.fetchall()
            for value in rows:
                # In Oracle, column names are uppercase by default
                value_dict = {
                    k.lower(): v for k, v in zip(
                        cur.description, value)}
                yield CheckpointTuple(
                    {
                        "configurable": {
                            "thread_id": value_dict["thread_id"],
                            "checkpoint_ns": value_dict["checkpoint_ns"],
                            "checkpoint_id": value_dict["checkpoint_id"],
                        }
                    },
                    await asyncio.to_thread(
                        self._load_checkpoint,
                        value_dict["checkpoint"],
                        value_dict["channel_values"],
                        value_dict["pending_sends"],
                    ),
                    self._load_metadata(value_dict["metadata"]),
                    (
                        {
                            "configurable": {
                                "thread_id": value_dict["thread_id"],
                                "checkpoint_ns": value_dict["checkpoint_ns"],
                                "checkpoint_id": value_dict["parent_checkpoint_id"],
                            }
                        }
                        if value_dict["parent_checkpoint_id"]
                        else None
                    ),
                    await asyncio.to_thread(self._load_writes, value_dict["pending_writes"]),
                )

    async def aget_tuple(
            self,
            config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Get a checkpoint tuple from the database asynchronously.

        This method retrieves a checkpoint tuple from the Oracle database based on the
        provided config. If the config contains a "checkpoint_id" key, the checkpoint with
        the matching thread ID and "checkpoint_id" is retrieved. Otherwise, the latest checkpoint
        for the given thread ID is retrieved.

        Args:
            config: The config to use for retrieving the checkpoint.

        Returns:
            Optional[CheckpointTuple]: The retrieved checkpoint tuple, or None if no matching
            checkpoint was found.
        """
        # Type-safe access to configurable
        configurable = config.get("configurable", {})
        if not configurable:
            return None

        thread_id = configurable["thread_id"]
        checkpoint_id = get_checkpoint_id(config)
        checkpoint_ns = configurable.get("checkpoint_ns", "")
        if checkpoint_id:
            args: tuple[Any, ...] = (thread_id, checkpoint_ns, checkpoint_id)
            where = "WHERE thread_id = :1 AND checkpoint_ns = :2 AND checkpoint_id = :3"
        else:
            args = (thread_id, checkpoint_ns)
            where = "WHERE thread_id = :1 AND checkpoint_ns = :2 ORDER BY checkpoint_id DESC FETCH FIRST 1 ROWS ONLY"

        async with self._cursor() as cur:
            await cur.execute(self.SELECT_SQL + where, args)
            row = await cur.fetchone()
            if row:
                value_dict = {
                    k.lower(): v for k, v in zip(
                        cur.description, row)}
                return CheckpointTuple(
                    {
                        "configurable": {
                            "thread_id": thread_id,
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": value_dict["checkpoint_id"],
                        }
                    },
                    await asyncio.to_thread(
                        self._load_checkpoint,
                        value_dict["checkpoint"],
                        value_dict["channel_values"],
                        value_dict["pending_sends"],
                    ),
                    self._load_metadata(value_dict["metadata"]),
                    (
                        {
                            "configurable": {
                                "thread_id": thread_id,
                                "checkpoint_ns": checkpoint_ns,
                                "checkpoint_id": value_dict["parent_checkpoint_id"],
                            }
                        }
                        if value_dict["parent_checkpoint_id"]
                        else None
                    ),
                    await asyncio.to_thread(self._load_writes, value_dict["pending_writes"]),
                )
            return None

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Save a checkpoint to the database asynchronously.

        This method saves a checkpoint to the Oracle database. The checkpoint is associated
        with the provided config and its parent config (if any).

        Args:
            config: The config to associate with the checkpoint.
            checkpoint: The checkpoint to save.
            metadata: Additional metadata to save with the checkpoint.
            new_versions: New channel versions as of this write.

        Returns:
            RunnableConfig: Updated configuration after storing the checkpoint.
        """
        # Type-safe access to configurable
        configurable = config.get("configurable", {})
        if not configurable:
            raise ValueError("Config must contain 'configurable' key")

        configurable = configurable.copy()
        thread_id = configurable.pop("thread_id")
        checkpoint_ns = configurable.pop("checkpoint_ns")
        checkpoint_id = configurable.pop(
            "checkpoint_id", configurable.pop("thread_ts", None)
        )

        copy = checkpoint.copy()
        next_config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint["id"],
            }
        }

        async with self._cursor(transaction=True) as cur:
            blob_params = await asyncio.to_thread(
                self._dump_blobs,
                thread_id,
                checkpoint_ns,
                copy.pop("channel_values"),  # type: ignore[misc]
                new_versions,
            )
            for params in blob_params:
                await cur.execute(self.UPSERT_CHECKPOINT_BLOBS_SQL, params)

            checkpoint_params = (
                thread_id,
                checkpoint_ns,
                checkpoint["id"],
                checkpoint_id,
                self._dump_checkpoint(copy),
                self._dump_metadata(get_checkpoint_metadata(config, metadata)),
            )
            await cur.execute(self.UPSERT_CHECKPOINTS_SQL, checkpoint_params)

        return next_config

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Store intermediate writes linked to a checkpoint asynchronously.

        This method saves intermediate writes associated with a checkpoint to the database.

        Args:
            config: Configuration of the related checkpoint.
            writes: List of writes to store, each as (channel, value) pair.
            task_id: Identifier for the task creating the writes.
            task_path: Path of the task creating the writes.
        """
        # Type-safe access to configurable
        configurable = config.get("configurable", {})
        if not configurable:
            raise ValueError("Config must contain 'configurable' key")

        query = (
            self.UPSERT_CHECKPOINT_WRITES_SQL
            if all(w[0] in WRITES_IDX_MAP for w in writes)
            else self.INSERT_CHECKPOINT_WRITES_SQL
        )
        params = await asyncio.to_thread(
            self._dump_writes,
            configurable["thread_id"],
            configurable["checkpoint_ns"],
            configurable["checkpoint_id"],
            task_id,
            task_path,
            writes,
        )
        async with self._cursor(transaction=True) as cur:
            for param in params:
                await cur.execute(query, param)

    async def adelete_thread(self, thread_id: str) -> None:
        """Delete all checkpoints and writes associated with a thread ID.

        Args:
            thread_id: The thread ID to delete.
        """
        async with self._cursor(transaction=True) as cur:
            await cur.execute(
                "DELETE FROM checkpoints WHERE thread_id = :1",
                (str(thread_id),),
            )
            await cur.execute(
                "DELETE FROM checkpoint_blobs WHERE thread_id = :1",
                (str(thread_id),),
            )
            await cur.execute(
                "DELETE FROM checkpoint_writes WHERE thread_id = :1",
                (str(thread_id),),
            )

    @asynccontextmanager
    async def _cursor(
            self, *, transaction: bool = False) -> AsyncIterator[oracledb.AsyncCursor]:
        """Create a database cursor as a context manager.

        Args:
            transaction: Whether to use a transaction for the operations inside the context manager.
        """
        async with self.lock:
            async with _ainternal.get_connection(self.conn) as conn:
                async with conn.cursor() as cur:
                    try:
                        yield cur
                        if transaction:
                            await conn.commit()
                    except Exception:
                        if transaction:
                            await conn.rollback()
                        raise

    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints from the database.

        This method retrieves a list of checkpoint tuples from the Oracle database based
        on the provided config. The checkpoints are ordered by checkpoint ID in descending
        order (newest first).

        Args:
            config: Base configuration for filtering checkpoints.
            filter: Additional filtering criteria for metadata.
            before: If provided, only checkpoints before the specified checkpoint ID are returned.
            limit: Maximum number of checkpoints to return.

        Yields:
            Iterator[CheckpointTuple]: An iterator of matching checkpoint tuples.
        """
        try:
            # check if we are in the main thread, only bg threads can block
            if asyncio.get_running_loop() is self.loop:
                raise asyncio.InvalidStateError(
                    "Synchronous calls to AsyncOracleCheckpointer are only allowed from a "
                    "different thread. From the main thread, use the async interface. "
                    "For example, use `checkpointer.alist(...)` or `await "
                    "graph.ainvoke(...)`.")
        except RuntimeError:
            pass

        # Use a different approach to handle async iteration
        async def collect_all():
            results = []
            async for item in self.alist(config, filter=filter, before=before, limit=limit):
                results.append(item)
            return results

        results = asyncio.run_coroutine_threadsafe(
            collect_all(), self.loop).result()
        for item in results:
            yield item

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Get a checkpoint tuple from the database.

        This method retrieves a checkpoint tuple from the Oracle database based on the
        provided config. If the config contains a "checkpoint_id" key, the checkpoint with
        the matching thread ID and "checkpoint_id" is retrieved. Otherwise, the latest checkpoint
        for the given thread ID is retrieved.

        Args:
            config: The config to use for retrieving the checkpoint.

        Returns:
            Optional[CheckpointTuple]: The retrieved checkpoint tuple, or None if no matching
            checkpoint was found.
        """
        try:
            # check if we are in the main thread, only bg threads can block
            if asyncio.get_running_loop() is self.loop:
                raise asyncio.InvalidStateError(
                    "Synchronous calls to AsyncOracleCheckpointer are only allowed from a "
                    "different thread. From the main thread, use the async interface. "
                    "For example, use `await checkpointer.aget_tuple(...)` or `await "
                    "graph.ainvoke(...)`.")
        except RuntimeError:
            pass

        return asyncio.run_coroutine_threadsafe(
            self.aget_tuple(config), self.loop
        ).result()

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Save a checkpoint to the database.

        This method saves a checkpoint to the Oracle database. The checkpoint is associated
        with the provided config and its parent config (if any).

        Args:
            config: The config to associate with the checkpoint.
            checkpoint: The checkpoint to save.
            metadata: Additional metadata to save with the checkpoint.
            new_versions: New channel versions as of this write.

        Returns:
            RunnableConfig: Updated configuration after storing the checkpoint.
        """
        return asyncio.run_coroutine_threadsafe(
            self.aput(config, checkpoint, metadata, new_versions), self.loop
        ).result()

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Store intermediate writes linked to a checkpoint.

        This method saves intermediate writes associated with a checkpoint to the database.

        Args:
            config: Configuration of the related checkpoint.
            writes: List of writes to store, each as (channel, value) pair.
            task_id: Identifier for the task creating the writes.
            task_path: Path of the task creating the writes.
        """
        return asyncio.run_coroutine_threadsafe(
            self.aput_writes(config, writes, task_id, task_path), self.loop
        ).result()

    def delete_thread(self, thread_id: str) -> None:
        """Delete all checkpoints and writes associated with a thread ID.

        Args:
            thread_id: The thread ID to delete.
        """
        try:
            # check if we are in the main thread, only bg threads can block
            if asyncio.get_running_loop() is self.loop:
                raise asyncio.InvalidStateError(
                    "Synchronous calls to AsyncOracleCheckpointer are only allowed from a "
                    "different thread. From the main thread, use the async interface. "
                    "For example, use `await checkpointer.adelete_thread(...)` or `await "
                    "graph.ainvoke(...)`.")
        except RuntimeError:
            pass

        return asyncio.run_coroutine_threadsafe(
            self.adelete_thread(thread_id), self.loop
        ).result()


__all__ = ["AsyncOracleCheckpointer", "Conn"]
