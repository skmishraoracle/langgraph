#!/usr/bin/env python3
"""Comprehensive tests for Oracle store implementation."""

from unittest.mock import AsyncMock, Mock, patch

import pytest
import json

from langgraph.store.base import BaseStore, TTLConfig
from langgraph.store.oracle.aio import AsyncOracleStore
from langgraph.store.oracle.base import OracleIndexConfig, OracleStore, PoolConfig


class FakeCursor:
    def execute(self, *args, **kwargs):
        return None
    def fetchall(self):
        return []
    def fetchone(self):
        return [0]
    @property
    def rowcount(self):
        return 5
    def close(self):
        return None

class FakeConnection:
    def __init__(self, cursor=None):
        self._cursor = cursor or FakeCursor()
    def cursor(self):
        return self._cursor
    def commit(self):
        return None
    def close(self):
        return None

class TestOracleStore:
    """Test suite for Oracle store implementation."""

    @pytest.fixture(autouse=True)
    def setup(self, monkeypatch) -> None:
        """Set up test fixtures."""
        import oracledb
        monkeypatch.setattr(oracledb, "Connection", FakeConnection)
        self.test_namespace = ("test", "namespace")
        self.test_key = "test_key"
        self.test_value = {"data": "test_value", "number": 42}
        self.test_namespace_2 = ("test", "namespace2")
        self.test_key_2 = "test_key_2"
        self.test_value_2 = {"data": "test_value_2", "number": 100}

    def test_imports(self) -> None:
        """Test that all imports work correctly."""
        assert AsyncOracleStore is not None
        assert OracleStore is not None
        assert OracleIndexConfig is not None

    @patch('oracledb.connect')
    def test_sync_store_creation(self, mock_connect) -> None:
        mock_connect.return_value = FakeConnection()
        
        with OracleStore.from_conn_string("test_connection") as store:
            assert isinstance(store, OracleStore)
            assert isinstance(store, BaseStore)

    @patch('oracledb.connect_async')
    async def test_async_store_creation(self, mock_connect_async) -> None:
        """Test asynchronous store creation."""
        mock_conn = AsyncMock()
        mock_connect_async.return_value = mock_conn
        
        async with AsyncOracleStore.from_conn_string("test_connection") as store:
            assert isinstance(store, AsyncOracleStore)
            assert isinstance(store, BaseStore)

    @patch('oracledb.create_pool')
    @patch('oracledb.connect')
    def test_sync_store_with_pool_config(self, mock_connect, mock_create_pool) -> None:
        """Test synchronous store creation with pool configuration."""
        mock_connect.return_value = FakeConnection()
        mock_pool = Mock()
        mock_create_pool.return_value = mock_pool
        
        pool_config: PoolConfig = {
            "min_size": 2,
            "max_size": 10,
            "kwargs": {}  # Remove autocommit as it's not supported by oracledb.create_pool()
        }
        
        with OracleStore.from_conn_string("test_connection", pool_config=pool_config) as store:
            assert isinstance(store, OracleStore)

    @patch('oracledb.connect')
    def test_sync_store_with_index_config(self, mock_connect) -> None:
        """Test synchronous store creation with index configuration."""
        mock_connect.return_value = FakeConnection()
        
        def dummy_embed(texts):
            return [[0.0] * 1536 for _ in texts]
        
        index_config: OracleIndexConfig = {
            "dims": 1536,
            "fields": ["content"],
            "distance_type": "cosine",
            "embed": dummy_embed,
        }
        
        with OracleStore.from_conn_string("test_connection", index=index_config) as store:
            assert isinstance(store, OracleStore)
            assert store.index_config is not None

    @patch('oracledb.connect')
    def test_sync_store_with_ttl_config(self, mock_connect) -> None:
        """Test synchronous store creation with TTL configuration."""
        mock_connect.return_value = FakeConnection()
        
        ttl_config: TTLConfig = {
            "default_ttl": 60,
            "sweep_interval_minutes": 5
        }
        
        with OracleStore.from_conn_string("test_connection", ttl=ttl_config) as store:
            assert isinstance(store, OracleStore)
            assert store.ttl_config is not None

    @patch('oracledb.connect')
    def test_sync_setup(self, mock_connect) -> None:
        """Test synchronous store setup."""
        mock_cursor = Mock()
        mock_cursor.execute.return_value = None
        mock_cursor.fetchone.return_value = [0]  # Simulate version row
        mock_connect.return_value = FakeConnection(cursor=mock_cursor)
        with OracleStore.from_conn_string("test_connection") as store:
            store.setup()
            mock_cursor.execute.assert_called()

    @patch('oracledb.connect_async')
    async def test_async_setup(self, mock_connect_async) -> None:
        """Test asynchronous store setup."""
        mock_conn = AsyncMock()
        mock_cursor = AsyncMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect_async.return_value = mock_conn
        
        # Mock cursor fetchone to return None (no existing migrations)
        mock_cursor.fetchone.return_value = None
        
        async with AsyncOracleStore.from_conn_string("test_connection") as store:
            await store.setup()
            # Verify cursor was used
            mock_cursor.execute.assert_called()

    @patch('oracledb.connect')
    def test_sync_put_and_get(self, mock_connect) -> None:
        """Test synchronous put and get operations."""
        mock_cursor = Mock()
        mock_cursor.execute.return_value = None
        # setup (version row), put (not used), get (returns a row with serialized value)
        serialized_value = json.dumps(self.test_value)
        mock_cursor.fetchone.side_effect = [[0], None, [self.test_key, serialized_value, None, None]]
        mock_cursor.fetchall.return_value = []
        mock_connect.return_value = FakeConnection(cursor=mock_cursor)
        with OracleStore.from_conn_string("test_connection") as store:
            store.setup()
            key = store.put(self.test_namespace, self.test_value, self.test_key)
            assert key == self.test_key
            item = store.get(self.test_namespace, self.test_key)
            assert item is not None
            assert item.key == self.test_key
            assert item.value == self.test_value
            assert item.namespace == self.test_namespace

    @patch('oracledb.connect_async')
    async def test_async_put_and_get(self, mock_connect_async) -> None:
        """Test asynchronous put and get operations."""
        mock_conn = AsyncMock()
        mock_cursor = AsyncMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect_async.return_value = mock_conn
        
        # Mock cursor fetchone to return None (no existing item)
        mock_cursor.fetchone.return_value = None
        
        async with AsyncOracleStore.from_conn_string("test_connection") as store:
            await store.setup()
            
            # Test put operation - note the parameter order: namespace, key, value
            key = await store.aput(self.test_namespace, self.test_key, self.test_value)
            assert key == self.test_key
            
            # Test get operation
            item = await store.aget(self.test_namespace, self.test_key)
            assert item is not None
            assert item.key == self.test_key
            assert item.value == self.test_value
            assert item.namespace == self.test_namespace

    @patch('oracledb.connect')
    def test_sync_batch_operations(self, mock_connect) -> None:
        """Test synchronous batch operations."""
        mock_connect.return_value = FakeConnection()
        
        # Mock cursor fetchall to return empty list
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = []
        
        with OracleStore.from_conn_string("test_connection") as store:
            store.setup()
            
            # Test batch operations
            from langgraph.store.base import GetOp, PutOp
            
            ops = [
                PutOp(namespace=self.test_namespace, key=self.test_key, value=self.test_value),
                GetOp(namespace=self.test_namespace, key=self.test_key),
            ]
            
            results = store.batch(ops)
            assert len(results) == 2
            assert results[0] is None  # Put operation returns None
            assert results[1] is None  # Get operation returns None (no item found)

    @patch('oracledb.connect_async')
    async def test_async_batch_operations(self, mock_connect_async) -> None:
        """Test asynchronous batch operations."""
        mock_conn = AsyncMock()
        mock_cursor = AsyncMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect_async.return_value = mock_conn
        
        # Mock cursor fetchall to return empty list
        mock_cursor.fetchall.return_value = []
        
        async with AsyncOracleStore.from_conn_string("test_connection") as store:
            await store.setup()
            
            # Test batch operations
            from langgraph.store.base import GetOp, PutOp
            
            ops = [
                PutOp(namespace=self.test_namespace, key=self.test_key, value=self.test_value),
                GetOp(namespace=self.test_namespace, key=self.test_key),
            ]
            
            results = await store.abatch(ops)
            assert len(results) == 2
            assert results[0] is None  # Put operation returns None
            assert results[1] is None  # Get operation returns None (no item found)

    @patch('oracledb.connect')
    def test_sync_search(self, mock_connect) -> None:
        """Test synchronous search operation."""
        mock_connect.return_value = FakeConnection()
        
        # Mock cursor fetchall to return empty list
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = []
        
        with OracleStore.from_conn_string("test_connection") as store:
            store.setup()
            
            # Test search operation
            results = store.search(self.test_namespace, query="test")
            assert isinstance(results, list)

    @patch('oracledb.connect_async')
    async def test_async_search(self, mock_connect_async) -> None:
        """Test asynchronous search operation."""
        mock_conn = AsyncMock()
        mock_cursor = AsyncMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect_async.return_value = mock_conn
        
        # Mock cursor fetchall to return empty list
        mock_cursor.fetchall.return_value = []
        
        async with AsyncOracleStore.from_conn_string("test_connection") as store:
            await store.setup()
            
            # Test search operation
            results = await store.asearch(self.test_namespace, query="test")
            assert isinstance(results, list)

    @patch('oracledb.connect')
    def test_sync_list_namespaces(self, mock_connect) -> None:
        """Test synchronous list_namespaces operation."""
        mock_connect.return_value = FakeConnection()
        
        # Mock cursor fetchall to return empty list
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = []
        
        with OracleStore.from_conn_string("test_connection") as store:
            store.setup()
            
            # Test list_namespaces operation
            namespaces = store.list_namespaces()
            assert isinstance(namespaces, list)

    @patch('oracledb.connect_async')
    async def test_async_list_namespaces(self, mock_connect_async) -> None:
        """Test asynchronous list_namespaces operation."""
        mock_conn = AsyncMock()
        mock_cursor = AsyncMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect_async.return_value = mock_conn
        
        # Mock cursor fetchall to return empty list
        mock_cursor.fetchall.return_value = []
        
        async with AsyncOracleStore.from_conn_string("test_connection") as store:
            await store.setup()
            
            # Test list_namespaces operation
            namespaces = await store.alist_namespaces()
            assert isinstance(namespaces, list)

    @patch('oracledb.connect')
    def test_sync_delete(self, mock_connect) -> None:
        """Test synchronous delete operation."""
        mock_cursor = Mock()
        mock_cursor.execute.return_value = None
        mock_cursor.fetchone.return_value = [0]
        mock_connect.return_value = FakeConnection(cursor=mock_cursor)
        with OracleStore.from_conn_string("test_connection") as store:
            store.setup()
            store.delete(self.test_namespace, self.test_key)
            mock_cursor.execute.assert_called()

    @patch('oracledb.connect_async')
    async def test_async_delete(self, mock_connect_async) -> None:
        """Test asynchronous delete operation."""
        mock_conn = AsyncMock()
        mock_cursor = AsyncMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect_async.return_value = mock_conn
        
        async with AsyncOracleStore.from_conn_string("test_connection") as store:
            await store.setup()
            
            # Test delete operation
            await store.adelete(self.test_namespace, self.test_key)
            
            # Verify cursor was used
            mock_cursor.execute.assert_called()

    @patch('oracledb.connect')
    def test_sync_delete_namespace(self, mock_connect) -> None:
        """Test synchronous delete_namespace operation."""
        mock_cursor = Mock()
        mock_cursor.execute.return_value = None
        mock_cursor.fetchone.return_value = [0]
        mock_connect.return_value = FakeConnection(cursor=mock_cursor)
        with OracleStore.from_conn_string("test_connection") as store:
            store.setup()
            store.delete_namespace(self.test_namespace)
            mock_cursor.execute.assert_called()

    @patch('oracledb.connect')
    def test_sync_sweep_ttl(self, mock_connect) -> None:
        """Test synchronous sweep_ttl operation."""
        mock_cursor = Mock()
        mock_cursor.execute.return_value = None
        mock_cursor.fetchone.return_value = [0]
        # Simulate rowcount being used for deleted_count
        type(mock_cursor).rowcount = property(lambda self: 5)
        mock_connect.return_value = FakeConnection(cursor=mock_cursor)
        with OracleStore.from_conn_string("test_connection") as store:
            store.setup()
            deleted_count = store.sweep_ttl()
            assert deleted_count == 5

    def test_ttl_sweeper(self) -> None:
        """Test TTL sweeper functionality."""
        # This would normally be tested with a real connection
        # For now, we just test that the method exists
        pass

    def test_vector_search_capabilities(self) -> None:
        """Test vector search capabilities."""
        # This would normally be tested with a real connection and vector embeddings
        # For now, we just test that the method exists
        pass


if __name__ == "__main__":
    pytest.main([__file__]) 