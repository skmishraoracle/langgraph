"""Base implementation for Oracle store in LangGraph."""
import asyncio
import concurrent.futures
import json
import logging
import threading
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from typing import (
    Any,
    Callable,
    List,
    Literal,
    Optional,
    Union,
    cast,
)

import oracledb
from langgraph.checkpoint.oracle import _internal
from langgraph.store.base import (
    BaseStore,
    GetOp,
    IndexConfig,
    Item,
    ListNamespacesOp,
    MatchCondition,
    Op,
    PutOp,
    Result,
    SearchItem,
    SearchOp,
    TTLConfig,
    ensure_embeddings,
    tokenize_path,
)
from typing_extensions import TypedDict

logger = logging.getLogger(__name__)

# Schema migrations for Oracle
MIGRATIONS = [
    """
  CREATE TABLE store (
      prefix VARCHAR2(2000) NOT NULL,
      key VARCHAR2(2000) NOT NULL,
      value CLOB NOT NULL,
      created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
      updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
      expires_at TIMESTAMP WITH TIME ZONE,
      ttl_minutes NUMBER,
      CONSTRAINT pk_store PRIMARY KEY (prefix, key)
  )
  """,

    """
  CREATE INDEX store_prefix_idx ON store (prefix)
  """,

    """
  CREATE INDEX idx_store_expires_at ON store (expires_at)
  WHERE expires_at IS NOT NULL
  """,
]


class OracleIndexConfig(IndexConfig, total=False):
    """Configuration for vector embeddings in Oracle store.

    Extends IndexConfig with Oracle-specific options for vector search.
    """
    dims: int
    """Dimensionality of the embeddings"""

    embed: Any
    """Embeddings provider (LangChain compatible)"""

    fields: Union[List[str], str]
    """Fields to embed for vector search (default is entire document)"""

    distance_type: Literal["l2", "cosine", "inner_product"]
    """Distance metric to use:
  - 'l2': Euclidean distance
  - 'cosine': Cosine similarity
  - 'inner_product': Dot product
  """

    # Internal fields for processing
    __tokenized_fields: list
    __estimated_num_vectors: int


class PoolConfig(TypedDict, total=False):
    """Connection pool settings for Oracle connections.

    Controls connection lifecycle and resource utilization.
    """
    min_size: int
    """Minimum number of connections maintained in the pool. Defaults to 1."""

    max_size: Optional[int]
    """Maximum number of connections allowed in the pool. None means unlimited."""

    kwargs: dict
    """Additional connection arguments passed to each connection in the pool."""


def _serialize_value(value: Any) -> str:
    """Serialize a value to JSON string.

    Args:
        value: Value to serialize

    Returns:
        str: JSON string
    """
    return json.dumps(value)


def _deserialize_value(
    value: str,
    deserializer: Optional[Callable[[Union[bytes, str]], Any]] = None
) -> Any:
    """Deserialize a value from JSON string.

    Args:
        value: JSON string to deserialize
        deserializer: Optional custom deserializer

    Returns:
        Any: Deserialized value
    """
    if deserializer:
        return deserializer(value)
    return json.loads(value)


def _serialize_embedding(embedding: list[float]) -> bytes:
    """Serialize a vector embedding to binary format.

    Args:
        embedding: Vector embedding

    Returns:
        bytes: Binary representation
    """
    import array

    # Convert to binary format
    float_array = array.array('f', embedding)
    return float_array.tobytes()


def get_filter_condition(key: str, op: str, value: Any, param_idx: int
                         ) -> tuple[str, dict]:
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


class OracleStore(BaseStore):
    """Oracle-backed store with optional vector search capabilities.
    """

    __slots__ = (
        "conn",
        "_deserializer",
        "lock",
        "index_config",
        "embeddings",
        "_ttl_sweeper_thread",
        "_ttl_stop_event",
        "ttl_config",
        "vector_store",
    )
    supports_ttl: bool = True

    def __init__(
        self,
        conn: _internal.Conn,
        *,
        deserializer=None,
        index: Optional[OracleIndexConfig] = None,
        ttl: Optional[TTLConfig] = None,
    ) -> None:
        """Initialize OracleStore.

        Args:
            conn: Oracle connection or pool
            deserializer: Optional custom deserializer for values
            index: Optional vector index configuration
            ttl: Optional TTL configuration for item expiration
        """
        super().__init__()
        self.conn = conn
        self._deserializer = deserializer
        self.lock = threading.Lock()
        self.index_config = index
        self.ttl_config = ttl
        self._ttl_sweeper_thread = None
        self._ttl_stop_event = threading.Event()

        if self.index_config:
            self.embeddings, self.index_config = self._ensure_index_config(
                self.index_config
            )
        else:
            self.embeddings = None

    @classmethod
    @contextmanager
    def from_conn_string(
        cls,
        conn_string: str,
        *,
        pool_config: Optional[PoolConfig] = None,
        index: Optional[OracleIndexConfig] = None,
        ttl: Optional[TTLConfig] = None,
    ) -> Iterator["OracleStore"]:
        """Create a new OracleStore instance from a connection string.

        Args:
            conn_string: The Oracle connection string (username/password@host:port/service)
            pool_config: Configuration for the connection pool
            index: Optional vector index configuration
            ttl: Optional TTL configuration for item expiration

        Returns:
            OracleStore: A new OracleStore instance.
        """
        conn = None
        try:
            if pool_config is not None:
                pc = pool_config.copy()
                min_size = pc.pop("min_size", 1)
                max_size = pc.pop("max_size", None)
                kwargs = pc.pop("kwargs", {})

                # Create connection pool with appropriate settings
                pool = oracledb.create_pool(
                    conn_string,
                    min=min_size,
                    max=max_size if max_size is not None else 5,
                    **kwargs,
                )
                conn = pool
            else:
                conn = oracledb.connect(conn_string)

            yield cls(conn=conn, index=index, ttl=ttl)
        finally:
            if conn is not None:
                if isinstance(conn, oracledb.SessionPool):
                    conn.close()
                else:
                    conn.close()

    def setup(self) -> None:
        """Set up the store database.

        This method creates the necessary tables in the Oracle database if they don't
        already exist and runs database migrations. It MUST be called directly by the user
        the first time the store is used.
        """
        with self._cursor() as cur:
            # Create version tracking table if needed
            cur.execute(
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
            cur.execute(
                "SELECT v FROM store_migrations ORDER BY v DESC FETCH FIRST 1 ROWS ONLY")
            row = cur.fetchone()
            if row is None:
                version = -1
            else:
                version = row[0]

            # Apply migrations
            for v, sql in enumerate(
                    MIGRATIONS[version + 1:], start=version + 1):
                try:
                    cur.execute(sql)
                    cur.execute(
                        "INSERT INTO store_migrations (v) VALUES (:1)", (v,))
                    cur.execute("COMMIT")
                except Exception as e:
                    cur.execute("ROLLBACK")
                    raise RuntimeError(
                        f"Failed to apply migration {v}.\nSQL={sql}\nError={e}") from e

            # If index config is provided, set up vector search capabilities
            if self.index_config:
                # Setup vector search tables and indexes
                cur.execute(
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
                    cur.execute(
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
                    logger.warning(
                        f"Could not create vector search index: {e}")

    @contextmanager
    def _cursor(self) -> Iterator[oracledb.Cursor]:
        """Create a database cursor as a context manager."""
        with _internal.get_connection(self.conn) as conn:
            with self.lock:
                cursor = conn.cursor()
                try:
                    yield cursor
                finally:
                    cursor.close()

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        """Execute multiple store operations in a batch.

        Args:
            ops: List of operations to execute

        Returns:
            list[Result]: Results for each operation
        """
        grouped_ops, num_ops = self._group_ops(ops)
        results: list[Result] = [None] * num_ops

        with self._cursor() as cur:
            if GetOp in grouped_ops:
                self._batch_get_ops(
                    cast(Sequence[tuple[int, GetOp]], grouped_ops[GetOp]), results, cur
                )

            if SearchOp in grouped_ops:
                self._batch_search_ops(
                    cast(Sequence[tuple[int, SearchOp]], grouped_ops[SearchOp]),
                    results,
                    cur,
                )

            if ListNamespacesOp in grouped_ops:
                self._batch_list_namespaces_ops(
                    cast(
                        Sequence[tuple[int, ListNamespacesOp]],
                        grouped_ops[ListNamespacesOp],
                    ),
                    results,
                    cur,
                )

            if PutOp in grouped_ops:
                self._batch_put_ops(
                    cast(Sequence[tuple[int, PutOp]], grouped_ops[PutOp]), cur
                )

        return results

    def _batch_get_ops(
        self,
        get_ops: Sequence[tuple[int, GetOp]],
        results: list[Result],
        cur: oracledb.Cursor,
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
                refresh_placeholders = ", ".join(
                    [":refresh" + str(i) for i in range(len(keys))])

                # First update TTLs where needed
                if any(this_refresh_ttls):
                    refresh_update = f"""
                        UPDATE store s
                        SET expires_at = SYSTIMESTAMP + (s.ttl_minutes * INTERVAL '1' MINUTE)
                        WHERE s.prefix = :prefix
                        AND s.key IN ({placeholders})
                        AND s.ttl_minutes IS NOT NULL
                    """
                    params = {"prefix": ns_text}

                    # Add key parameters
                    for i, key in enumerate(keys):
                        params[f"key{i}"] = key

                    # Apply TTL refresh
                    refresh_items = []
                    for i, refresh in enumerate(this_refresh_ttls):
                        if refresh:
                            refresh_items.append(keys[i])

                    if refresh_items:
                        # Only execute if there are items to refresh
                        refresh_placeholders = ", ".join(
                            [f":rkey{i}" for i in range(len(refresh_items))])
                        refresh_update += f" AND s.key IN ({refresh_placeholders})"

                        for i, key in enumerate(refresh_items):
                            params[f"rkey{i}"] = key

                        cur.execute(refresh_update, params)

                # Then fetch all requested items
                query = f"""
                SELECT s.key, s.value, s.created_at, s.updated_at
                FROM store s
                WHERE s.prefix = :prefix
                AND s.key IN ({placeholders})
                """

                cur.execute(query, params)
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

                cur.execute(query, params)

            # Process results
            rows = cur.fetchall()
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

    def _batch_put_ops(
        self,
        put_ops: Sequence[tuple[int, PutOp]],
        cur: oracledb.Cursor,
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

                cur.execute(query, params)

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
                query = """
                MERGE INTO store target
                USING (SELECT :prefix AS prefix, :key AS key FROM dual) source
                ON (target.prefix = source.prefix AND target.key = source.key)
                WHEN MATCHED THEN
                    UPDATE SET
                        value = :value,
                        updated_at = SYSTIMESTAMP,
                        expires_at = {expires},
                        ttl_minutes = :ttl
                WHEN NOT MATCHED THEN
                    INSERT (prefix, key, value, created_at, updated_at, expires_at, ttl_minutes)
                    VALUES (
                        :prefix,
                        :key,
                        :value,
                        SYSTIMESTAMP,
                        SYSTIMESTAMP,
                        {expires},
                        :ttl
                    )
                """.format(expires=expires_at)

                params = {
                    "prefix": ns_text,
                    "key": op.key,
                    "value": _serialize_value(op.value),
                    "ttl": ttl_minutes
                }

                cur.execute(query, params)

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
                embeddings = self.embeddings.embed_documents(texts)

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

                    cur.execute(query, params)

    def _batch_search_ops(
        self,
        search_ops: Sequence[tuple[int, SearchOp]],
        results: list[Result],
        cur: oracledb.Cursor,
    ) -> None:
        """Execute batch SEARCH operations.

        Args:
            search_ops: List of SEARCH operations
            results: Results array to populate
            cur: Database cursor
        """
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
                            condition, params = get_filter_condition(
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
                query_embedding = self.embeddings.embed_query(op.query)

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
                    # Refresh TTL for matched items
                    refresh_query = f"""
                    BEGIN
                        -- First refresh TTL for matched items
                        UPDATE store s
                        SET expires_at = SYSTIMESTAMP + (s.ttl_minutes * INTERVAL '1' MINUTE)
                        WHERE s.ttl_minutes IS NOT NULL
                        AND EXISTS (
                            SELECT 1 FROM ({query}) sr
                            WHERE sr.prefix = s.prefix AND sr.key = s.key
                        );
                    END;
                    """
                    cur.execute(refresh_query, params)

                # Execute search query
                cur.execute(query, params)
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
                    # Refresh TTL for matched items
                    refresh_query = f"""
                    BEGIN
                        -- First refresh TTL for matched items
                        UPDATE store s
                        SET expires_at = SYSTIMESTAMP + (s.ttl_minutes * INTERVAL '1' MINUTE)
                        WHERE s.ttl_minutes IS NOT NULL
                        AND EXISTS (
                            SELECT 1 FROM ({query}) sr
                            WHERE sr.prefix = s.prefix AND sr.key = s.key
                        );
                    END;
                    """
                    cur.execute(refresh_query, params)

                # Execute search query
                cur.execute(query, params)

            # Process results
            rows = cur.fetchall()
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

    def _batch_list_namespaces_ops(
        self,
        list_ops: Sequence[tuple[int, ListNamespacesOp]],
        results: list[Result],
        cur: oracledb.Cursor,
    ) -> None:
        """Execute batch LIST_NAMESPACES operations.

        Args:
            list_ops: List of LIST_NAMESPACES operations
            results: Results array to populate
            cur: Database cursor
        """
        for idx, op in list_ops:
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
                truncation_query = f"""
                SELECT
                    CASE
                        WHEN :max_depth IS NOT NULL THEN
                            SUBSTR(prefix, 1,
                                DECODE(
                                    INSTR(
                                        SUBSTR(prefix, 1,
                                            INSTR(prefix || '.', '.', 1, :max_depth) - 1
                                        ),
                                        '.'
                                    ),
                                    0, LENGTH(prefix),
                                    INSTR(prefix || '.', '.', 1, :max_depth) - 1
                                )
                            )
                        ELSE prefix
                    END AS truncated_prefix,
                    prefix
                FROM (
                    SELECT DISTINCT prefix FROM store WHERE {where_clause}
                )
                """
                params["max_depth"] = op.max_depth

                # Complete query with pagination
                query = f"""
                SELECT DISTINCT truncated_prefix
                FROM ({truncation_query})
                ORDER BY truncated_prefix
                OFFSET :offset ROWS FETCH NEXT :limit ROWS ONLY
                """
            else:
                # Without max_depth, just return distinct prefixes
                query = f"""
                SELECT DISTINCT prefix AS truncated_prefix
                FROM store
                WHERE {where_clause}
                ORDER BY truncated_prefix
                OFFSET :offset ROWS FETCH NEXT :limit ROWS ONLY
                """

            # Add pagination parameters
            params["offset"] = op.offset
            params["limit"] = op.limit

            # Execute query
            cur.execute(query, params)

            # Process results
            rows = cur.fetchall()
            namespace_results = []

            for row in rows:
                namespace = tuple(row[0].split("."))
                namespace_results.append(namespace)

            results[idx] = namespace_results

    def get(self, namespace: tuple[str, ...], key: str) -> Optional[Item]:
        """Get a single item from the store.

        Args:
            namespace: Item namespace
            key: Item key

        Returns:
            Item or None if not found
        """
        ops = [GetOp(namespace=namespace, key=key)]
        batch_results = self.batch(ops)
        return cast(Optional[Item], batch_results[0])

    def put(
        self,
        namespace: tuple[str, ...],
        value: dict[str, Any],
        key: Optional[str] = None,
        *,
        ttl: Optional[float] = None,
        index: Union[list[str], Literal[False], None] = None,
    ) -> str:
        """Store an item in the store.

        Args:
            namespace: Item namespace
            value: Item value
            key: Optional item key (will be generated if not provided)
            ttl: Optional time-to-live in minutes
            index: Whether to index the item for vector search, or list of fields to index
                If None, uses default configuration
                If False, skips indexing
                If a list, indexes only the specified fields

        Returns:
            str: The key of the stored item
        """
        if key is None:
            # Generate UUID
            import uuid
            key = str(uuid.uuid4())

        self.batch([PutOp(namespace=namespace, key=key,
                   value=value, ttl=ttl, index=index)])
        return key

    def search(
        self,
        namespace_prefix: tuple[str, ...],
        query: Optional[str] = None,
        *,
        filter: Optional[dict] = None,
        limit: int = 10,
        offset: int = 0,
        refresh_ttl: bool = False,
    ) -> list[SearchItem]:
        """Search for items in the store."""
        ops = [
            SearchOp(
                namespace_prefix=namespace_prefix,
                query=query,
                filter=filter,
                limit=limit,
                offset=offset,
                refresh_ttl=refresh_ttl,
            )
        ]
        batch_results = self.batch(ops)
        return cast(list[SearchItem], batch_results[0])

    def list_namespaces(
        self,
        *,
        match_prefix: Optional[tuple[str, ...]] = None,
        match_suffix: Optional[tuple[str, ...]] = None,
        max_depth: Optional[int] = None,
        limit: int = 10,
        offset: int = 0,
    ) -> list[tuple[str, ...]]:
        """List namespaces in the store.

        Args:
            match_prefix: Optional namespace prefix to match
            match_suffix: Optional namespace suffix to match
            max_depth: Optional maximum depth of namespaces to return
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            list[tuple[str, ...]]: Matching namespaces
        """
        conditions = []
        if match_prefix:
            conditions.append(
                MatchCondition(
                    match_type="prefix", path=match_prefix))
        if match_suffix:
            conditions.append(
                MatchCondition(
                    match_type="suffix", path=match_suffix))

        ops = [
            ListNamespacesOp(
                match_conditions=tuple(conditions) if conditions else None,
                max_depth=max_depth,
                limit=limit,
                offset=offset,
            )
        ]
        batch_results = self.batch(ops)
        return cast(list[tuple[str, ...]], batch_results[0])

    def delete(self, namespace: tuple[str, ...], key: str) -> None:
        """Delete an item from the store.

        Args:
            namespace: Item namespace
            key: Item key
        """
        self.batch([PutOp(namespace=namespace, key=key, value=None)])

    def delete_namespace(self, namespace: tuple[str, ...]) -> None:
        """Delete all items in a namespace.

        Args:
            namespace: Namespace to delete
        """
        with self._cursor() as cur:
            ns_text = ".".join(namespace)
            # Delete all items in namespace and its subnamespaces
            cur.execute(
                "DELETE FROM store WHERE prefix = :prefix OR prefix LIKE :prefix_like", {
                    "prefix": ns_text, "prefix_like": f"{ns_text}.%"}, )

    def sweep_ttl(self) -> int:
        """Remove expired items from the store.

        Returns:
            int: Number of items removed.
        """
        if not self.ttl_config:
            return 0

        with self._cursor() as cur:
            cur.execute(
                f"""
                DELETE FROM store 
                WHERE expires_at IS NOT NULL 
                AND expires_at < CURRENT_TIMESTAMP
                """
            )
            deleted_count = cur.rowcount
            return deleted_count

    def start_ttl_sweeper(
        self, sweep_interval_minutes: Optional[int] = None
    ) -> concurrent.futures.Future:
        """Periodically delete expired store items based on TTL.

        Returns:
            Future that can be waited on or cancelled.
        """
        if not self.ttl_config:
            future: concurrent.futures.Future = concurrent.futures.Future()
            future.set_result(None)
            return future

        if self._ttl_sweeper_thread and self._ttl_sweeper_thread.is_alive():
            logger.info("TTL sweeper thread is already running")
            # Return a future that can be used to cancel the existing thread
            future = concurrent.futures.Future()
            future.add_done_callback(
                lambda f: self._ttl_stop_event.set() if f.cancelled() else None
            )
            return future

        self._ttl_stop_event.clear()

        interval = float(sweep_interval_minutes or self.ttl_config.get(
            "sweep_interval_minutes") or 5)
        logger.info(
            f"Starting store TTL sweeper with interval {interval} minutes")

        future = concurrent.futures.Future()

        def _sweep_loop() -> None:
            try:
                while not self._ttl_stop_event.is_set():
                    if self._ttl_stop_event.wait(interval * 60):
                        break

                    try:
                        expired_items = self.sweep_ttl()
                        if expired_items > 0:
                            logger.info(
                                f"Store swept {expired_items} expired items")
                    except Exception as exc:
                        logger.exception(
                            "Store TTL sweep iteration failed", exc_info=exc
                        )
                future.set_result(None)
            except Exception as exc:
                future.set_exception(exc)

        thread = threading.Thread(
            target=_sweep_loop,
            daemon=True,
            name="ttl-sweeper")
        self._ttl_sweeper_thread = thread
        thread.start()

        future.add_done_callback(
            lambda f: self._ttl_stop_event.set() if f.cancelled() else None
        )
        return future

    def stop_ttl_sweeper(self, timeout: Optional[float] = None) -> bool:
        """Stop the TTL sweeper thread if it's running.

        Args:
            timeout: Maximum time to wait for the thread to stop, in seconds.
                If None, wait indefinitely.

        Returns:
            bool: True if the thread was successfully stopped or wasn't running,
                False if the timeout was reached before the thread stopped.
        """
        if not self._ttl_sweeper_thread or not self._ttl_sweeper_thread.is_alive():
            return True

        logger.info("Stopping TTL sweeper thread")
        self._ttl_stop_event.set()

        self._ttl_sweeper_thread.join(timeout)
        success = not self._ttl_sweeper_thread.is_alive()

        if success:
            self._ttl_sweeper_thread = None
            logger.info("TTL sweeper thread stopped")
        else:
            logger.warning("Timed out waiting for TTL sweeper thread to stop")

        return success

    def __del__(self) -> None:
        """Ensure the TTL sweeper thread is stopped when the object is garbage collected."""
        if hasattr(
                self,
                "_ttl_stop_event") and hasattr(
                self,
                "_ttl_sweeper_thread"):
            self.stop_ttl_sweeper(timeout=0.1)

    def _ensure_index_config(
        self, index_config: OracleIndexConfig,
    ) -> tuple[Any, OracleIndexConfig]:
        """Ensure that the index configuration is valid and prepare it for use."""
        index_config = index_config.copy()

        # Process fields for embedding
        tokenized = []
        tot = 0
        text_fields = index_config.get("fields") or ["$"]

        if isinstance(text_fields, str):
            text_fields = [text_fields]

        if not isinstance(text_fields, list):
            raise ValueError(
                f"Text fields must be a list or a string. Got {text_fields}")

        for p in text_fields:
            if p == "$":
                tokenized.append((p, "$"))
                tot += 1
            else:
                toks = tokenize_path(p)
                tokenized.append((p, toks))
                tot += len(toks)

        index_config["__tokenized_fields"] = tokenized
        index_config["__estimated_num_vectors"] = tot

        # Ensure embeddings
        embeddings = ensure_embeddings(index_config.get("embed"))
        return embeddings, index_config

    @staticmethod
    def _group_ops(
            ops: Iterable[Op]) -> tuple[dict[type, list[tuple[int, Op]]], int]:
        """Group operations by type for batch processing.

        Args:
            ops: Operations to group

        Returns:
            Tuple of (grouped operations, total count)
        """
        grouped_ops: dict[type, list[tuple[int, Op]]] = defaultdict(list)
        tot = 0
        for idx, op in enumerate(ops):
            grouped_ops[type(op)].append((idx, op))
            tot += 1
        return grouped_ops, tot

    def _setup_vector_store(self) -> None:
        """Set up vector store capabilities using LangChain's OracleVS."""
        if not self.index_config:
            return

        try:
            from oraclevs import OracleVS  # type: ignore[import-untyped]
            from langchain.vectorstores import VectorStore  # type: ignore[import-untyped]

            # Connect our embeddings object with OracleVS
            self.vector_store = OracleVS(
                connection=self.conn,
                embedding=self.embeddings,
                table_name="store_vectors",
                dimension=self.index_config.get("dims", 1536),  # Default dimension
                distance_strategy=self.index_config.get(
                    "distance_type",
                    "cosine"))
        except ImportError:
            logger.warning(
                "OracleVS not found. Vector search capabilities will be limited. "
                "Install with: pip install oracle-vector-search")
            self.vector_store = None

    def _perform_vector_search(
        self,
        query: str,
        namespace_prefix: Optional[tuple[str, ...]] = None,
        filter: Optional[dict] = None,
        limit: int = 10,
        offset: int = 0
    ) -> list[SearchItem]:
        """Perform vector search using LangChain's OracleVS."""
        if not self.vector_store:
            return []

        # Prepare filter condition for OracleVS
        filter_condition = {}
        if namespace_prefix:
            prefix_text = ".".join(namespace_prefix)
            filter_condition["prefix"] = {"$match": f"{prefix_text}%"}

        if filter:
            filter_condition.update(filter)

        # Use OracleVS to search
        results = self.vector_store.similarity_search_with_score(
            query=query,
            k=limit + offset,  # Fetch extra for offset
            filter=filter_condition if filter_condition else None,
        )

        # Convert results to SearchItem format
        search_items = []
        for doc, score in results[offset:offset + limit]:
            search_items.append(
                SearchItem(
                    key=doc.metadata.get("key", ""),
                    namespace=tuple(doc.metadata.get("prefix", "").split(".")),
                    value=doc.metadata,
                    created_at=doc.metadata.get("created_at"),
                    updated_at=doc.metadata.get("updated_at"),
                    score=score,
                )
            )

        return search_items

    async def _perform_vector_search_async(
        self,
        query: str,
        namespace_prefix: Optional[tuple[str, ...]] = None,
        filter: Optional[dict] = None,
        limit: int = 10,
        offset: int = 0
    ) -> list[SearchItem]:
        """Perform vector search asynchronously using LangChain's OracleVS."""
        # Use asyncio.to_thread to run the synchronous OracleVS search in a
        # thread
        if not self.vector_store:
            return []

        # Prepare filter condition for OracleVS
        filter_condition = {}
        if namespace_prefix:
            prefix_text = ".".join(namespace_prefix)
            filter_condition["prefix"] = {"$match": f"{prefix_text}%"}

        if filter:
            filter_condition.update(filter)

        # Use OracleVS to search in a separate thread to avoid blocking the
        # event loop
        results = await asyncio.to_thread(
            self.vector_store.similarity_search_with_score,
            query=query,
            k=limit + offset,
            filter=filter_condition if filter_condition else None,
        )

        # Convert results to SearchItem format
        search_items = []
        for doc, score in results[offset:offset + limit]:
            search_items.append(
                SearchItem(
                    key=doc.metadata.get("key", ""),
                    namespace=tuple(doc.metadata.get("prefix", "").split(".")),
                    value=doc.metadata,
                    created_at=doc.metadata.get("created_at"),
                    updated_at=doc.metadata.get("updated_at"),
                    score=score,
                )
            )

        return search_items

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        """Execute multiple operations asynchronously in a single batch.

        This is a synchronous implementation that delegates to the sync batch method.
        For true async performance, use AsyncOracleStore.

        Args:
            ops: An iterable of operations to execute.

        Returns:
            A list of results, where each result corresponds to an operation in the input.
            The order of results matches the order of input operations.
        """
        # Delegate to sync implementation for now
        # In a real async implementation, this would use async database operations
        return self.batch(ops)
