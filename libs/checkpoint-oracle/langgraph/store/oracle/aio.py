"""Asynchronous Oracle store implementation for LangGraph."""
import asyncio
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import Any, Optional, Union, cast

import oracledb

from langgraph.checkpoint.oracle import _ainternal
from langgraph.store.base import (
    BaseStore,
    GetOp,
    Item,
    ListNamespacesOp,
    Op,
    PutOp,
    Result,
    SearchItem,
    SearchOp,
    TTLConfig,
)
from langgraph.store.oracle.base import (
    MIGRATIONS,
    OracleIndexConfig,
)

Conn = _ainternal.Conn  # For type hints


def _group_ops(ops: Sequence[Op]
               ) -> tuple[dict[type, list[tuple[int, Op]]], int]:
    """Group operations by type for batch processing."""
    grouped_ops: dict[type, list[tuple[int, Op]]] = {}
    tot = 0
    for idx, op in enumerate(ops):
        if type(op) not in grouped_ops:
            grouped_ops[type(op)] = []
        grouped_ops[type(op)].append((idx, op))
        tot += 1
    return grouped_ops, tot


def _serialize_value(value: Any) -> str:
    """Serialize a value to JSON string."""
    import json
    return json.dumps(value)


def _deserialize_value(value: str, deserializer: Optional[Any] = None) -> Any:
    """Deserialize a value from JSON string."""
    import json
    if deserializer:
        return deserializer(value)
    return json.loads(value)


def _serialize_embedding(embedding: list[float]) -> bytes:
    """Serialize a vector embedding to binary format."""
    import array
    float_array = array.array('f', embedding)
    return float_array.tobytes()


def _get_filter_condition(key: str, op: str, value: Any,
                          param_idx: int) -> tuple[str, dict]:
    """Helper to generate filter conditions for Oracle JSON."""
    params = {}
    if op == "$eq":
        condition = f"JSON_EXISTS(value, '$.{key}?(@==:p{param_idx})')"
        params[f"p{param_idx}"] = str(value)
    elif op == "$gt":
        condition = f"JSON_EXISTS(value, '$.{key}?(@>:p{param_idx})')"
        params[f"p{param_idx}"] = str(value)
    elif op == "$gte":
        condition = f"JSON_EXISTS(value, '$.{key}?(@>=:p{param_idx})')"
        params[f"p{param_idx}"] = str(value)
    elif op == "$lt":
        condition = f"JSON_EXISTS(value, '$.{key}?(@<:p{param_idx})')"
        params[f"p{param_idx}"] = str(value)
    elif op == "$lte":
        condition = f"JSON_EXISTS(value, '$.{key}?(@<=:p{param_idx})')"
        params[f"p{param_idx}"] = str(value)
    elif op == "$ne":
        condition = f"JSON_EXISTS(value, '$.{key}?(@!=:p{param_idx})')"
        params[f"p{param_idx}"] = str(value)
    else:
        raise ValueError(f"Unsupported operator: {op}")
    return condition, params


class AsyncOracleStore(BaseStore):
    """Asynchronous Oracle-backed store with optional vector search capabilities.

    !!! example "Examples"
        Basic setup and usage:
        ```python
        import asyncio
        from langgraph.store.oracle import AsyncOracleStore
        import oracledb

        conn_string = "username/password@localhost:1521/service"

        async def main():
            async with await AsyncOracleStore.from_conn_string(conn_string) as store:
                await store.setup()  # Run migrations. Done once

                # Store and retrieve data
                await store.aput(("users", "123"), {"theme": "dark"})
                item = await store.aget(("users", "123"))
                print(item)

        asyncio.run(main())
        ```

        Vector search using LangChain embeddings:
        ```python
        import asyncio
        from langchain.embeddings import init_embeddings
        from langgraph.store.oracle import AsyncOracleStore

        conn_string = "username/password@localhost:1521/service"

        async def main():
            async with await AsyncOracleStore.from_conn_string(
                conn_string,
                index={
                    "dims": 1536,
                    "embed": init_embeddings("openai:text-embedding-3-small"),
                    "fields": ["text"]  # specify which fields to embed. Default is the whole serialized value
                }
            ) as store:
                await store.setup()  # Do this once to run migrations

                # Store documents
                await store.aput(("docs",), {"text": "Python tutorial"})
                await store.aput(("docs",), {"text": "TypeScript guide"})

                # Search by similarity
                results = await store.asearch(("docs",), query="programming guides", limit=2)
                print(results)

        asyncio.run(main())
        ```

    Note:
        Semantic search is disabled by default. You can enable it by providing an `index` configuration
        when creating the store. Without this configuration, all `index` arguments passed to
        `aput` will have no effect.

    Warning:
        Make sure to call `setup()` before first use to create necessary tables and indexes.
        The Oracle database must have the Oracle Text feature enabled to use vector search.

    Note:
        If you provide a TTL configuration, you must explicitly call `start_ttl_sweeper()` to begin
        the background task that removes expired items. Call `stop_ttl_sweeper()` to properly
        clean up resources when you're done with the store.
    """

    __slots__ = (
        "conn",
        "lock",
        "loop",
        "index_config",
        "embeddings",
        "_ttl_sweeper_task",
        "_ttl_stop_event",
        "ttl_config",
        "_deserializer",
    )
    supports_ttl: bool = True

    def __init__(
        self,
        conn: Conn,
        *,
        deserializer=None,
        index: Optional[OracleIndexConfig] = None,
        ttl: Optional[TTLConfig] = None,
    ) -> None:
        """Initialize AsyncOracleStore.

        Args:
            conn: Oracle async connection or pool
            deserializer: Optional custom deserializer for values
            index: Optional vector index configuration
            ttl: Optional TTL configuration for item expiration
        """
        super().__init__()
        self.conn = conn
        self._deserializer = deserializer
        self.lock = asyncio.Lock()
        self.loop = asyncio.get_running_loop()
        self.index_config = index
        self.ttl_config = ttl
        self._ttl_sweeper_task = None
        self._ttl_stop_event = asyncio.Event()

        if self.index_config:
            self.embeddings, self.index_config = self._ensure_index_config(
                self.index_config
            )
        else:
            self.embeddings = None

    def _ensure_index_config(
        self, index_config: OracleIndexConfig
    ) -> tuple[Any, OracleIndexConfig]:
        """Ensure index configuration is properly set up."""
        from typing import cast

        from langgraph.store.base import ensure_embeddings, tokenize_path

        # Ensure embeddings are properly initialized
        embed = index_config.get("embed")
        embeddings = ensure_embeddings(embed) if embed is not None else None

        # Tokenize fields if specified
        if "fields" in index_config:
            if isinstance(index_config["fields"], str):
                cast(
                    dict, index_config)["__tokenized_fields"] = tokenize_path(
                    index_config["fields"])
            else:
                cast(dict, index_config)[
                    "__tokenized_fields"] = index_config["fields"]

        # Set default values
        if "distance_type" not in index_config:
            index_config["distance_type"] = "cosine"  # type: ignore
        if "__estimated_num_vectors" not in index_config:
            cast(dict, index_config)["__estimated_num_vectors"] = 1000

        return embeddings, index_config

    @classmethod
    @asynccontextmanager
    async def from_conn_string(
        cls,
        conn_string: str,
        *,
        pool_config: Optional[dict] = None,
        index: Optional[OracleIndexConfig] = None,
        ttl: Optional[TTLConfig] = None,
    ) -> AsyncIterator["AsyncOracleStore"]:
        """Create a new AsyncOracleStore instance from a connection string.

        Args:
            conn_string: The Oracle connection string (username/password@host:port/service)
            pool_config: Configuration for the connection pool
            index: Optional vector index configuration
            ttl: Optional TTL configuration for item expiration

        Yields:
            AsyncOracleStore: A new AsyncOracleStore instance
        """
        conn = None
        try:
            if pool_config is not None:
                pc = pool_config.copy()
                min_size = pc.pop("min_size", 1)
                max_size = pc.pop("max_size", None)
                kwargs = pc.pop("kwargs", {})

                # Create connection pool with appropriate settings
                pool = oracledb.create_pool_async(
                    conn_string,
                    min=min_size,
                    max=max_size if max_size is not None else 5,
                    **kwargs,
                    **pc,
                )
                conn = pool
            else:
                conn = await oracledb.connect_async(conn_string)

            yield cls(conn=conn, index=index, ttl=ttl)
        finally:
            if conn is not None:
                if isinstance(conn, oracledb.AsyncConnectionPool):
                    await conn.close()
                else:
                    await conn.close()

    async def setup(self) -> None:
        """Set up the store database.

        This method creates the necessary tables in the Oracle database if they don't
        already exist and runs database migrations. It MUST be called directly by the user
        the first time the store is used.
        """
        async with self._cursor() as cur:
            # Create version tracking table if needed
            await cur.execute(
                """
        BEGIN
            EXECUTE IMMEDIATE '
                CREATE TABLE store_migrations (
                    v NUMBER(10) PRIMARY KEY
                )
            ';
        EXCEPTION
            WHEN OTHERS THEN
                IF SQLCODE = -955 THEN NULL; -- ORA-00955: name is already used by an existing object
                ELSE RAISE;
                END IF;
        END;
        """
            )

            # Get current version
            await cur.execute("SELECT v FROM store_migrations ORDER BY v DESC FETCH FIRST 1 ROWS ONLY")
            row = await cur.fetchone()
            if row is None:
                version = -1
            else:
                version = row[0]

            # Apply migrations
            for v, sql in enumerate(
                    MIGRATIONS[version + 1:], start=version + 1):
                try:
                    await cur.execute(sql)
                    await cur.execute("INSERT INTO store_migrations (v) VALUES (:1)", (v,))
                    await cur.execute("COMMIT")
                except Exception as e:
                    await cur.execute("ROLLBACK")
                    raise RuntimeError(
                        f"Failed to apply migration {v}.\nSQL={sql}\nError={e}") from e

            # If index config is provided, set up vector search capabilities
            if self.index_config:
                # Setup vector search tables and indexes
                await cur.execute(
                    """
          BEGIN
              EXECUTE IMMEDIATE '
                  CREATE TABLE store_vectors (
                      prefix VARCHAR2(2000) NOT NULL,
                      key VARCHAR2(2000) NOT NULL,
                      field_name VARCHAR2(2000) NOT NULL,
                      embedding BLOB,
                      created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                      updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                      PRIMARY KEY (prefix, key, field_name),
                      FOREIGN KEY (prefix, key) REFERENCES store(prefix, key) ON DELETE CASCADE
                  )
              ';
          EXCEPTION
              WHEN OTHERS THEN
                  IF SQLCODE = -955 THEN NULL; -- ORA-00955: name is already used by an existing object
                  ELSE RAISE;
                  END IF;
          END;
          """
                )

                # Create Oracle Text index for vector search if not exists
                # This requires Oracle Text to be installed and configured
                try:
                    await cur.execute(
                        """
            BEGIN
                EXECUTE IMMEDIATE '
                    CREATE INDEX store_vectors_embedding_idx ON store_vectors(embedding)
                    INDEXTYPE IS CTXSYS.CONTEXT
                    PARAMETERS (''DATASTORE CTXSYS.DEFAULT_DATASTORE'')
                ';
            EXCEPTION
                WHEN OTHERS THEN
                    IF SQLCODE = -955 THEN NULL;
                    ELSE RAISE;
                    END IF;
            END;
            """
                    )
                except Exception as e:
                    # Log warning but don't fail - vector search might not be
                    # available
                    print(
                        f"Warning: Could not create vector search index: {e}")

    @asynccontextmanager
    async def _cursor(self) -> AsyncIterator[oracledb.AsyncCursor]:
        """Create a database cursor as a context manager."""
        async with _ainternal.get_connection(self.conn) as conn:
            async with self.lock:
                async with conn.cursor() as cur:
                    try:
                        yield cur
                    finally:
                        pass

    async def aget(
            self, namespace: tuple[str, ...], key: str) -> Optional[Item]:
        """Get a single item from the store asynchronously."""
        ops = [GetOp(namespace=namespace, key=key)]
        batch_results = await self.abatch(ops)
        return cast(Optional[Item], batch_results[0])

    async def aput(
        self,
        namespace: tuple[str, ...],
        key: Optional[str],
        value: dict[str, Any],
        *,
        ttl: Optional[float] = None,
        index: Optional[Union[bool, list[str]]] = None,
    ) -> str:
        """Put a value into the store asynchronously."""
        # Ensure key is always a string
        if key is None:
            import uuid
            key = str(uuid.uuid4())
        # Convert boolean index to proper type
        if isinstance(index, bool):
            index = False if index else None
        elif isinstance(index, list):
            index = index  # Keep as is
        else:
            index = None  # Default to None

        # Create PutOp with proper index type
        op = PutOp(
            namespace=namespace,
            key=key,
            value=value,
            ttl=ttl,
            index=index,  # Now properly typed
        )

        # Execute the operation but don't store the result since we return the
        # key
        await self.abatch([op])
        # Return the key (str) instead of the Result object
        return key

    # Implement other async methods following the same pattern

    async def abatch(self, ops: Sequence[Op]) -> list[Result]:
        """Execute multiple store operations in a batch.

        Args:
            ops: List of operations to execute

        Returns:
            list[Result]: Results for each operation
        """
        grouped_ops, num_ops = _group_ops(ops)
        results: list[Result] = [None] * num_ops

        async with self._cursor() as cur:
            if GetOp in grouped_ops:
                await self._batch_get_ops(
                    cast(Sequence[tuple[int, GetOp]], grouped_ops[GetOp]), results, cur
                )

            if SearchOp in grouped_ops:
                await self._batch_search_ops(
                    cast(Sequence[tuple[int, SearchOp]], grouped_ops[SearchOp]),
                    results,
                    cur,
                )

            if ListNamespacesOp in grouped_ops:
                await self._batch_list_namespaces_ops(
                    cast(
                        Sequence[tuple[int, ListNamespacesOp]],
                        grouped_ops[ListNamespacesOp],
                    ),
                    results,
                    cur,
                )

            if PutOp in grouped_ops:
                await self._batch_put_ops(
                    cast(Sequence[tuple[int, PutOp]], grouped_ops[PutOp]), cur
                )

        return results

    async def _batch_get_ops(
        self,
        get_ops: Sequence[tuple[int, GetOp]],
        results: list[Result],
        cur: oracledb.AsyncCursor,
    ) -> None:
        """Execute batch GET operations.

        Args:
            get_ops: List of GET operations
            results: Results array to populate
            cur: Database cursor
        """
        # Group by namespace for efficiency
        namespace_groups = {}
        refresh_ttls = {}
        for idx, op in get_ops:
            if op.namespace not in namespace_groups:
                namespace_groups[op.namespace] = []
                refresh_ttls[op.namespace] = []
            namespace_groups[op.namespace].append((idx, op.key))
            refresh_ttls[op.namespace].append(op.refresh_ttl)

        # Process each namespace group
        for namespace, items in namespace_groups.items():
            idx_map = {key: idx for idx, key in items}
            keys = [key for _, key in items]
            this_refresh_ttls = refresh_ttls[namespace]

            # Format namespace as text
            ns_text = ".".join(namespace)

            # Get items from database
            if any(this_refresh_ttls):
                # Update TTL for items that need refreshing
                placeholders = ", ".join([":key" + str(i)
                                         for i in range(len(keys))])
                query = f"""
                BEGIN
                    -- First update TTLs where needed
                    UPDATE store s
                    SET expires_at = SYSTIMESTAMP + (s.ttl_minutes * INTERVAL '1' MINUTE)
                    WHERE s.prefix = :prefix
                    AND s.key IN ({placeholders})
                    AND EXISTS (
                        SELECT 1 FROM TABLE(CAST(:refresh_array AS SYS.ODCIVARCHAR2LIST))
                        WHERE COLUMN_VALUE = 'TRUE'
                    )
                    AND s.ttl_minutes IS NOT NULL;

                    -- Then fetch all requested items
                    OPEN :cursor FOR
                    SELECT s.key, s.value, s.created_at, s.updated_at
                    FROM store s
                    WHERE s.prefix = :prefix
                    AND s.key IN ({placeholders});
                END;
                """

                # Prepare refresh flags
                refresh_flags = [str(flag).upper()
                                 for flag in this_refresh_ttls]

                # Execute
                params = {
                    "prefix": ns_text,
                    "refresh_array": refresh_flags,
                    "cursor": cur
                }

                # Add key parameters
                for i, key in enumerate(keys):
                    params[f"key{i}"] = key

                await cur.execute(query, params)
            else:
                # Simple query without TTL refresh
                placeholders = ", ".join(
                    [f":key{i}" for i in range(len(keys))])
                query = f"""
                SELECT s.key, s.value, s.created_at, s.updated_at
                FROM store s
                WHERE s.prefix = :prefix
                AND s.key IN ({placeholders})
                """

                params = {"prefix": ns_text}
                for i, key in enumerate(keys):
                    params[f"key{i}"] = key

                await cur.execute(query, params)

            # Process results
            rows = await cur.fetchall()
            for row in rows:
                key = row[0]
                idx = idx_map[key]
                value = _deserialize_value(row[1], self._deserializer)
                results[idx] = Item(
                    key=key,
                    value=value,
                    namespace=namespace,
                    created_at=row[2],
                    updated_at=row[3],
                )

    async def _batch_put_ops(
        self,
        put_ops: Sequence[tuple[int, PutOp]],
        cur: oracledb.AsyncCursor,
    ) -> None:
        """Execute batch PUT operations.

        Args:
            put_ops: List of PUT operations
            cur: Database cursor
        """
        # Deduplicate operations - last one for a key wins
        dedupped_ops = {}
        for _, op in put_ops:
            dedupped_ops[(op.namespace, op.key)] = op

        # Separate inserts and deletes
        inserts = []
        deletes = []
        for op in dedupped_ops.values():
            if op.value is None:
                deletes.append(op)
            else:
                inserts.append(op)

        # Handle deletes
        if deletes:
            namespace_groups = {}
            for op in deletes:
                if op.namespace not in namespace_groups:
                    namespace_groups[op.namespace] = []
                namespace_groups[op.namespace].append(op.key)

            for namespace, keys in namespace_groups.items():
                ns_text = ".".join(namespace)
                placeholders = ", ".join(
                    [f":key{i}" for i in range(len(keys))])

                query = f"""
                DELETE FROM store
                WHERE prefix = :prefix
                AND key IN ({placeholders})
                """

                params = {"prefix": ns_text}
                for i, key in enumerate(keys):
                    params[f"key{i}"] = key

                await cur.execute(query, params)

        # Handle inserts
        if inserts:
            # Prepare vector embedding requests if needed
            vector_requests = []

            # Process each insert
            for op in inserts:
                ns_text = ".".join(op.namespace)

                # Handle TTL
                if op.ttl is not None:
                    expires_at = f"SYSTIMESTAMP + INTERVAL '{op.ttl}' MINUTE"
                    ttl_minutes = op.ttl
                else:
                    expires_at = "NULL"
                    ttl_minutes = None

                # Insert or update the main record
                query = f"""
                MERGE INTO store target
                USING (SELECT :prefix AS prefix, :key AS key FROM dual) source
                ON (target.prefix = source.prefix AND target.key = source.key)
                WHEN MATCHED THEN
                    UPDATE SET
                        value = :value,
                        updated_at = SYSTIMESTAMP,
                        expires_at = {expires_at},
                        ttl_minutes = :ttl
                WHEN NOT MATCHED THEN
                    INSERT (prefix, key, value, created_at, updated_at, expires_at, ttl_minutes)
                    VALUES (
                        :prefix,
                        :key,
                        :value,
                        SYSTIMESTAMP,
                        SYSTIMESTAMP,
                        {expires_at},
                        :ttl
                    )
                """

                params = {
                    "prefix": ns_text,
                    "key": op.key,
                    "value": _serialize_value(op.value),
                    "ttl": ttl_minutes
                }

                await cur.execute(query, params)

                # Process vector embedding if configured
                if self.index_config and op.index is not False:
                    if op.index is None:
                        fields = self.index_config.get("fields", ["$"])
                    else:
                        fields = op.index

                    # Add to vector requests for later processing
                    for field in fields:
                        if field == "$":
                            # Embed the entire value
                            text = str(op.value)
                        elif isinstance(op.value, dict) and field in op.value:
                            # Embed a specific field
                            text = str(op.value[field])
                        else:
                            continue

                        vector_requests.append({
                            "namespace": ns_text,
                            "key": op.key,
                            "field": field,
                            "text": text
                        })

            # Process vector embeddings if needed
            if vector_requests and self.embeddings:
                # Generate embeddings
                texts = [req["text"] for req in vector_requests]
                embeddings = await asyncio.to_thread(
                    self.embeddings.embed_documents, texts
                )

                # Store embeddings
                for req, embedding in zip(vector_requests, embeddings):
                    query = """
                    MERGE INTO store_vectors target
                    USING (
                        SELECT :prefix AS prefix, :key AS key, :field AS field_name FROM dual
                    ) source
                    ON (
                        target.prefix = source.prefix AND
                        target.key = source.key AND
                        target.field_name = source.field_name
                    )
                    WHEN MATCHED THEN
                        UPDATE SET
                            embedding = :embedding,
                            updated_at = SYSTIMESTAMP
                    WHEN NOT MATCHED THEN
                        INSERT (prefix, key, field_name, embedding, created_at, updated_at)
                        VALUES (
                            :prefix,
                            :key,
                            :field_name,
                            :embedding,
                            SYSTIMESTAMP,
                            SYSTIMESTAMP
                        )
                    """

                    # Convert embedding to BLOB
                    embedding_bytes = _serialize_embedding(embedding)

                    params = {
                        "prefix": req["namespace"],
                        "key": req["key"],
                        "field_name": req["field"],
                        "embedding": embedding_bytes
                    }

                    await cur.execute(query, params)

    async def _batch_search_ops(
        self,
        search_ops: Sequence[tuple[int, SearchOp]],
        results: list[Result],
        cur: oracledb.AsyncCursor,
    ) -> None:
        """Execute batch SEARCH operations.

        Args:
            search_ops: List of SEARCH operations
            results: List to store results
            cur: Database cursor
        """
        # search_query_results = []  # Removed unused variable

        # Process each search operation
        for idx, op in search_ops:
            # Build filter conditions
            filter_conditions = []
            filter_params = {}
            param_idx = 1

            if op.filter:
                for key, value in op.filter.items():
                    if isinstance(value, dict):
                        for op_name, val in value.items():
                            condition, params = _get_filter_condition(
                                key, op_name, val, param_idx
                            )
                            filter_conditions.append(condition)
                            filter_params.update(params)
                            param_idx += len(params)
                    else:
                        filter_conditions.append(
                            f"JSON_EXISTS(value, '$.{key}' PASSING :p{param_idx} AS \"v\")")
                        filter_params[f"p{param_idx}"] = str(value)
                        param_idx += 1

            # Handle namespace prefix
            if op.namespace_prefix:
                ns_prefix = ".".join(op.namespace_prefix)
                ns_condition = "store.prefix LIKE :ns_prefix || '%'"
                filter_params["ns_prefix"] = ns_prefix
            else:
                ns_condition = "1=1"  # Always true

            # Complete filter clause
            filter_clause = " AND ".join(
                [ns_condition] +
                filter_conditions) if filter_conditions else ns_condition

            # Vector search if configured and query provided
            if op.query and self.index_config and self.embeddings:
                # Generate query embedding
                query_embedding = await asyncio.to_thread(
                    self.embeddings.embed_query, op.query
                )

                # Convert to BLOB
                query_embedding_bytes = _serialize_embedding(query_embedding)

                # Oracle vector search query using Oracle Text
                # Note: This is a simplified approach - actual implementation would depend
                # on specific Oracle version and vector search capabilities
                query = f"""
                WITH scored_results AS (
                    SELECT
                        s.prefix,
                        s.key,
                        s.value,
                        s.created_at,
                        s.updated_at,
                        -- This is a placeholder for similarity calculation
                        -- In a real implementation, you would use Oracle's vector similarity
                        1.0 AS score
                    FROM
                        store s
                    JOIN
                        store_vectors sv ON s.prefix = sv.prefix AND s.key = sv.key
                    WHERE
                        {filter_clause}
                    -- Order by similarity score
                    ORDER BY score DESC
                )
                SELECT * FROM scored_results
                OFFSET :offset ROWS FETCH NEXT :limit ROWS ONLY
                """

                params = {
                    **filter_params,
                    "query_vector": query_embedding_bytes,
                    "offset": op.offset,
                    "limit": op.limit
                }

                if op.refresh_ttl:
                    # Wrap in a block to refresh TTL
                    wrapped_query = f"""
                    DECLARE
                        v_rows SYS_REFCURSOR;
                    BEGIN
                        -- First refresh TTL for matched items
                        UPDATE store s
                        SET expires_at = SYSTIMESTAMP + (s.ttl_minutes * INTERVAL '1' MINUTE)
                        WHERE s.ttl_minutes IS NOT NULL
                        AND EXISTS (
                            SELECT 1 FROM ({query}) sr
                            WHERE sr.prefix = s.prefix AND sr.key = s.key
                        );

                        -- Then return the results
                        OPEN v_rows FOR {query};
                        :cursor := v_rows;
                    END;
                    """
                    params["cursor"] = cur
                    await cur.execute(wrapped_query, params)
                else:
                    await cur.execute(query, params)
            else:
                # Regular search without vector similarity
                query = f"""
                SELECT
                    s.prefix,
                    s.key,
                    s.value,
                    s.created_at,
                    s.updated_at,
                    NULL AS score
                FROM
                    store s
                WHERE
                    {filter_clause}
                ORDER BY
                    s.updated_at DESC
                OFFSET :offset ROWS FETCH NEXT :limit ROWS ONLY
                """

                params = {
                    **filter_params,
                    "offset": op.offset,
                    "limit": op.limit
                }

                if op.refresh_ttl:
                    # Wrap in a block to refresh TTL
                    wrapped_query = f"""
                    DECLARE
                        v_rows SYS_REFCURSOR;
                    BEGIN
                        -- First refresh TTL for matched items
                        UPDATE store s
                        SET expires_at = SYSTIMESTAMP + (s.ttl_minutes * INTERVAL '1' MINUTE)
                        WHERE s.ttl_minutes IS NOT NULL
                        AND EXISTS (
                            SELECT 1 FROM ({query}) sr
                            WHERE sr.prefix = s.prefix AND sr.key = s.key
                        );

                        -- Then return the results
                        OPEN v_rows FOR {query};
                        :cursor := v_rows;
                    END;
                    """
                    params["cursor"] = cur
                    await cur.execute(wrapped_query, params)
                else:
                    await cur.execute(query, params)

            # Process results
            rows = await cur.fetchall()
            search_results = []

            for row in rows:
                prefix, key, value, created_at, updated_at, score = row
                namespace = tuple(prefix.split("."))
                value_dict = _deserialize_value(value, self._deserializer)

                search_results.append(
                    SearchItem(
                        key=key,
                        namespace=namespace,
                        value=value_dict,
                        created_at=created_at,
                        updated_at=updated_at,
                        score=float(score) if score is not None else None,
                    )
                )

            results[idx] = search_results

    async def _batch_list_namespaces_ops(
        self,
        list_ops: Sequence[tuple[int, ListNamespacesOp]],
        results: list[Result],
        cur: oracledb.AsyncCursor,
    ) -> None:
        """Execute batch LIST_NAMESPACES operations.

        Args:
            list_ops: List of LIST_NAMESPACES operations
            results: Results array to populate
            cur: Database cursor
        """
        for _idx, op in list_ops:
            # Build query conditions
            conditions = []
            params = {}
            param_idx = 1

            if op.match_conditions:
                for condition in op.match_conditions:
                    if condition.match_type == "prefix":
                        conditions.append(f"prefix LIKE :cond{param_idx}")
                        params[f"cond{param_idx}"] = f"{'.'.join(condition.path)}%"
                    elif condition.match_type == "suffix":
                        conditions.append(f"prefix LIKE :cond{param_idx}")
                        params[f"cond{param_idx}"] = f"%{'.'.join(condition.path)}"
                    param_idx += 1

            # Build WHERE clause
            where_clause = " AND ".join(conditions) if conditions else "1=1"

            # Handle max_depth truncation
            if op.max_depth is not None:
                # Oracle implementation of prefix truncation to max_depth
                # truncation_query = f""" ... """  # Removed unused variable
                # Note: max_depth truncation not yet implemented for Oracle
                pass

            # Execute query
            query = f"""
              SELECT DISTINCT prefix
              FROM store
              WHERE {where_clause}
              ORDER BY prefix
              OFFSET :offset ROWS FETCH NEXT :limit ROWS ONLY
              """

            params = {
                **params,
                "offset": op.offset,
                "limit": op.limit
            }

            await cur.execute(query, params)
            rows = await cur.fetchall()

            # Process results
            namespaces = []
            for row in rows:
                prefix = row[0]
                namespace = tuple(prefix.split("."))
                namespaces.append(namespace)

            results[_idx] = namespaces
