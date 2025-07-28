# LangGraph Checkpoint Oracle

Implementation of LangGraph CheckpointSaver that uses Oracle.

## Dependencies

By default `langgraph-checkpoint-oracle` installs `oracledb` without any extras. However, you can choose a specific installation that best suits your needs in the [oracledb installation guide](https://python-oracledb.readthedocs.io/en/latest/user_guide/installation.html).

## Usage

> [!IMPORTANT]
> When using Oracle checkpointers for the first time, make sure to call `.setup()` method on them to create required tables. See example below.

> [!IMPORTANT]
> When manually creating Oracle connections and passing them to `OracleCheckpointer` or `AsyncOracleCheckpointer`, make sure to use the `oracledb` driver. The implementation uses the `oracledb` Python driver which is compatible with Oracle Database 19c and later.
>
> **Example of correct usage:**
> 
> ```python
> import oracledb
> from langgraph.checkpoint.oracle import OracleCheckpointer
> 
> # Using connection string
> conn_string = "username/password@localhost:1521/service"
> with OracleCheckpointer.from_conn_string(conn_string) as checkpointer:
>     checkpointer.setup()
>     # Use the checkpointer...
> 
> # Using connection parameters
> with OracleCheckpointer.from_parameters(
>     user="username", 
>     password="password", 
>     dsn="localhost:1521/service"
> ) as checkpointer:
>     checkpointer.setup()
>     # Use the checkpointer...
> 
> ```

```python
from langgraph.checkpoint.oracle import OracleCheckpointer

write_config = {"configurable": {"thread_id": "1", "checkpoint_ns": ""}}
read_config = {"configurable": {"thread_id": "1"}}

DB_URI = "username/password@localhost:1521/service"
with OracleCheckpointer.from_conn_string(DB_URI) as checkpointer:
    # call .setup() the first time you're using the checkpointer
    checkpointer.setup()
    checkpoint = {
        "v": 4,
        "ts": "2024-07-31T20:14:19.804150+00:00",
        "id": "1ef4f797-8335-6428-8001-8a1503f9b875",
        "channel_values": {
            "my_key": "meow",
            "node": "node"
        },
        "channel_versions": {
            "__start__": 2,
            "my_key": 3,
            "start:node": 3,
            "node": 3
        },
        "versions_seen": {
            "__input__": {},
            "__start__": {
                "__start__": 1
            },
            "node": {
                "start:node": 2
            }
        },
    }

    # store checkpoint
    checkpointer.put(write_config, checkpoint, {}, {})

    # load checkpoint
    checkpointer.get_tuple(read_config)

    # list checkpoints
    list(checkpointer.list(read_config))
```

### Async

```python
from langgraph.checkpoint.oracle.aio import AsyncOracleCheckpointer

async with AsyncOracleCheckpointer.from_conn_string(DB_URI) as checkpointer:
    checkpoint = {
        "v": 4,
        "ts": "2024-07-31T20:14:19.804150+00:00",
        "id": "1ef4f797-8335-6428-8001-8a1503f9b875",
        "channel_values": {
            "my_key": "meow",
            "node": "node"
        },
        "channel_versions": {
            "__start__": 2,
            "my_key": 3,
            "start:node": 3,
            "node": 3
        },
        "versions_seen": {
            "__input__": {},
            "__start__": {
                "__start__": 1
            },
            "node": {
                "start:node": 2
            }
        },
    }

    # store checkpoint
    await checkpointer.aput(write_config, checkpoint, {}, {})

    # load checkpoint
    await checkpointer.aget_tuple(read_config)

    # list checkpoints
    [c async for c in checkpointer.alist(read_config)]
```

## Store API

This package also provides Oracle-backed store implementations:

```python
from langgraph.store.oracle import OracleStore, AsyncOracleStore

# Synchronous store
with OracleStore.from_conn_string(DB_URI) as store:
    store.setup()  # Create tables
    store.put(("users", "123"), {"name": "John", "age": 30})
    user = store.get(("users", "123"))

# Asynchronous store
async with AsyncOracleStore.from_conn_string(DB_URI) as store:
    await store.setup()  # Create tables
    await store.aput(("users", "123"), {"name": "John", "age": 30})
    user = await store.aget(("users", "123"))
```
