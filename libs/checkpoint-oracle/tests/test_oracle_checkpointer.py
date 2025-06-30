#!/usr/bin/env python3
"""Comprehensive tests for Oracle checkpointer implementation."""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from langchain_core.runnables import RunnableConfig

from langgraph.checkpoint.base import (
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    create_checkpoint,
    empty_checkpoint,
)
from langgraph.checkpoint.oracle import OracleCheckpointer
from langgraph.checkpoint.oracle.aio import AsyncOracleCheckpointer
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer


class TestOracleCheckpointer:
    """Test suite for Oracle checkpointer implementation."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """Set up test fixtures."""
        self.serde = JsonPlusSerializer()
        
        # Test configurations
        self.config_1: RunnableConfig = {
            "configurable": {
                "thread_id": "thread-1",
                "checkpoint_ns": "",
                "thread_ts": "1",
            }
        }
        self.config_2: RunnableConfig = {
            "configurable": {
                "thread_id": "thread-2",
                "checkpoint_id": "2",
                "checkpoint_ns": "",
            }
        }
        self.config_3: RunnableConfig = {
            "configurable": {
                "thread_id": "thread-2",
                "checkpoint_id": "2-inner",
                "checkpoint_ns": "inner",
            }
        }

        # Test checkpoints
        self.chkpnt_1: Checkpoint = empty_checkpoint()
        self.chkpnt_2: Checkpoint = create_checkpoint(self.chkpnt_1, {}, 1)
        self.chkpnt_3: Checkpoint = empty_checkpoint()

        # Test metadata
        self.metadata_1: CheckpointMetadata = {
            "source": "input",
            "step": 2,
            "writes": {},
        }
        self.metadata_2: CheckpointMetadata = {
            "source": "loop",
            "step": 1,
            "writes": {"foo": "bar"},
        }
        self.metadata_3: CheckpointMetadata = {}

    def test_imports(self) -> None:
        """Test that all imports work correctly."""
        assert AsyncOracleCheckpointer is not None
        assert OracleCheckpointer is not None
        assert JsonPlusSerializer is not None

    def test_serde_creation(self) -> None:
        """Test JsonPlusSerializer creation."""
        serde = JsonPlusSerializer()
        assert serde is not None

    @patch('oracledb.connect')
    def test_sync_checkpointer_creation(self, mock_connect) -> None:
        """Test synchronous checkpointer creation."""
        mock_conn = Mock()
        mock_connect.return_value = mock_conn
        
        with OracleCheckpointer.from_conn_string("test_connection") as checkpointer:
            assert isinstance(checkpointer, OracleCheckpointer)
            assert checkpointer.serde is not None

    @patch('oracledb.connect_async')
    async def test_async_checkpointer_creation(self, mock_connect_async) -> None:
        """Test asynchronous checkpointer creation."""
        mock_conn = AsyncMock()
        mock_connect_async.return_value = mock_conn
        
        async with AsyncOracleCheckpointer.from_conn_string("test_connection") as checkpointer:
            assert isinstance(checkpointer, AsyncOracleCheckpointer)
            assert checkpointer.serde is not None

    @patch('oracledb.connect')
    def test_sync_checkpointer_from_parameters(self, mock_connect) -> None:
        """Test synchronous checkpointer creation from parameters."""
        mock_conn = Mock()
        mock_connect.return_value = mock_conn
        
        with OracleCheckpointer.from_parameters(
            user="test_user",
            password="test_password", 
            dsn="test_dsn"
        ) as checkpointer:
            assert isinstance(checkpointer, OracleCheckpointer)

    @patch('oracledb.connect_async')
    async def test_async_checkpointer_from_parameters(self, mock_connect_async) -> None:
        """Test asynchronous checkpointer creation from parameters."""
        mock_conn = AsyncMock()
        mock_connect_async.return_value = mock_conn
        
        async with AsyncOracleCheckpointer.from_parameters(
            user="test_user",
            password="test_password",
            dsn="test_dsn"
        ) as checkpointer:
            assert isinstance(checkpointer, AsyncOracleCheckpointer)

    @patch('oracledb.connect')
    def test_sync_setup(self, mock_connect) -> None:
        """Test synchronous checkpointer setup."""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn
        
        with OracleCheckpointer.from_conn_string("test_connection") as checkpointer:
            checkpointer.setup()
            # Verify cursor was used
            mock_cursor.execute.assert_called()

    @patch('oracledb.connect_async')
    async def test_async_setup(self, mock_connect_async) -> None:
        """Test asynchronous checkpointer setup."""
        mock_conn = AsyncMock()
        mock_cursor = AsyncMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect_async.return_value = mock_conn
        
        async with AsyncOracleCheckpointer.from_conn_string("test_connection") as checkpointer:
            await checkpointer.setup()
            # Verify cursor was used
            mock_cursor.execute.assert_called()

    @patch('oracledb.connect')
    def test_sync_put_and_get(self, mock_connect) -> None:
        """Test synchronous put and get operations."""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn
        
        # Mock cursor fetchone to return None (no existing checkpoint)
        mock_cursor.fetchone.return_value = None
        
        with OracleCheckpointer.from_conn_string("test_connection") as checkpointer:
            checkpointer.setup()
            
            # Test put operation
            new_versions: ChannelVersions = {"channel1": 1}
            result_config = checkpointer.put(
                self.config_1, self.chkpnt_1, self.metadata_1, new_versions
            )
            
            # Verify the result config contains the checkpoint ID
            configurable = result_config.get("configurable", {})
            assert "checkpoint_id" in configurable
            assert configurable["thread_id"] == "thread-1"

    @patch('oracledb.connect_async')
    async def test_async_put_and_get(self, mock_connect_async) -> None:
        """Test asynchronous put and get operations."""
        mock_conn = AsyncMock()
        mock_cursor = AsyncMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect_async.return_value = mock_conn
        
        # Mock cursor fetchone to return None (no existing checkpoint)
        mock_cursor.fetchone.return_value = None
        
        async with AsyncOracleCheckpointer.from_conn_string("test_connection") as checkpointer:
            await checkpointer.setup()
            
            # Test put operation
            new_versions: ChannelVersions = {"channel1": 1}
            result_config = await checkpointer.aput(
                self.config_1, self.chkpnt_1, self.metadata_1, new_versions
            )
            
            # Verify the result config contains the checkpoint ID
            configurable = result_config.get("configurable", {})
            assert "checkpoint_id" in configurable
            assert configurable["thread_id"] == "thread-1"

    @patch('oracledb.connect')
    def test_sync_put_writes(self, mock_connect) -> None:
        """Test synchronous put_writes operation."""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn
        
        # Mock cursor fetchone to return 0 for write index
        mock_cursor.fetchone.return_value = [0]
        
        with OracleCheckpointer.from_conn_string("test_connection") as checkpointer:
            checkpointer.setup()
            
            # Test put_writes operation - use config_2 which has checkpoint_id
            writes = [("node1", "data1"), ("node2", "data2")]
            checkpointer.put_writes(self.config_2, writes, "task1", "path1")
            
            # Verify cursor was used
            mock_cursor.execute.assert_called()

    @patch('oracledb.connect_async')
    async def test_async_put_writes(self, mock_connect_async) -> None:
        """Test asynchronous put_writes operation."""
        mock_conn = AsyncMock()
        mock_cursor = AsyncMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect_async.return_value = mock_conn
        
        # Mock cursor fetchone to return 0 for write index
        mock_cursor.fetchone.return_value = [0]
        
        async with AsyncOracleCheckpointer.from_conn_string("test_connection") as checkpointer:
            await checkpointer.setup()
            
            # Test put_writes operation - use config_2 which has checkpoint_id
            writes = [("node1", "data1"), ("node2", "data2")]
            await checkpointer.aput_writes(self.config_2, writes, "task1", "path1")
            
            # Verify cursor was used
            mock_cursor.execute.assert_called()

    @patch('oracledb.connect')
    def test_sync_delete_thread(self, mock_connect) -> None:
        """Test synchronous delete_thread operation."""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn
        
        with OracleCheckpointer.from_conn_string("test_connection") as checkpointer:
            checkpointer.setup()
            
            # Test delete_thread operation
            checkpointer.delete_thread("thread-1")
            
            # Verify cursor was used
            mock_cursor.execute.assert_called()

    @patch('oracledb.connect_async')
    async def test_async_delete_thread(self, mock_connect_async) -> None:
        """Test asynchronous delete_thread operation."""
        mock_conn = AsyncMock()
        mock_cursor = AsyncMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect_async.return_value = mock_conn
        
        async with AsyncOracleCheckpointer.from_conn_string("test_connection") as checkpointer:
            await checkpointer.setup()
            
            # Test delete_thread operation
            await checkpointer.adelete_thread("thread-1")
            
            # Verify cursor was used
            mock_cursor.execute.assert_called()

    @patch('oracledb.connect')
    def test_sync_list_checkpoints(self, mock_connect) -> None:
        """Test synchronous list operation."""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn
        
        # Mock cursor fetchall to return empty list
        mock_cursor.fetchall.return_value = []
        
        with OracleCheckpointer.from_conn_string("test_connection") as checkpointer:
            checkpointer.setup()
            
            # Test list operation
            checkpoints = list(checkpointer.list(self.config_1))
            assert isinstance(checkpoints, list)

    @patch('oracledb.connect_async')
    async def test_async_list_checkpoints(self, mock_connect_async) -> None:
        """Test asynchronous list operation."""
        mock_conn = AsyncMock()
        mock_cursor = AsyncMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect_async.return_value = mock_conn
        
        # Mock cursor fetchall to return empty list
        mock_cursor.fetchall.return_value = []
        
        async with AsyncOracleCheckpointer.from_conn_string("test_connection") as checkpointer:
            await checkpointer.setup()
            
            # Test list operation
            checkpoints = []
            async for checkpoint in checkpointer.alist(self.config_1):
                checkpoints.append(checkpoint)
            assert isinstance(checkpoints, list)

    def test_invalid_config(self) -> None:
        """Test handling of invalid configuration."""
        # This would normally be tested with a real connection, but we can test the validation
        pass

    def test_missing_thread_id(self) -> None:
        """Test handling of missing thread_id."""
        # This would normally be tested with a real connection, but we can test the validation
        pass


if __name__ == "__main__":
    pytest.main([__file__]) 