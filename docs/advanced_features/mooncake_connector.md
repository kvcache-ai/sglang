# Mooncake Connector

SGLang provides a `MooncakeConnector` that allows you to use [Mooncake](https://github.com/kvcache-ai/Mooncake) as a high-performance distributed storage backend for loading model weights and other data.

## Prerequisites

To use the Mooncake connector, you must have the `mooncake` Python package installed and a Mooncake cluster (or standalone instance) running.

### Installation

Follow the [Mooncake installation guide](https://kvcache-ai.github.io/Mooncake/getting_started/build.html) to install the `mooncake` package.

```bash
pip install mooncake-transfer-engine
```

## Configuration

The connector relies on environment variables to configure the connection to the Mooncake cluster. The configuration is compatible with the standard Mooncake setup.

Required environment variables:

-   `MOONCAKE_MASTER`: The address of the Mooncake master server (e.g., `127.0.0.1:50051`).
-   `MOONCAKE_CLIENT`: (Optional) The address of the Mooncake client if running in standalone/client mode.

Other optional variables (see Mooncake docs for details):
-   `MOONCAKE_PROTOCOL`: Protocol to use (e.g., `rdma`, `tcp`).
-   `MOONCAKE_DEVICE`: Network device to use.
-   `MOONCAKE_TE_META_DATA_SERVER`: Metadata server address.

## Usage

You can use the Mooncake connector by specifying a URI starting with `mooncake://`.

### Python API

```python
from sglang.srt.connector import create_remote_connector
import torch

# Initialize the connector
# The URL path can be used to specify a key or prefix, but the base connection 
# is configured via environment variables.
connector = create_remote_connector("mooncake://", device="cpu")

# Store a tensor
key = "my_model_weight"
tensor = torch.randn(1024, 1024)
connector.set(key, tensor)

# Retrieve a tensor
retrieved_tensor = connector.get(key)
assert torch.allclose(tensor, retrieved_tensor)

# Store and retrieve strings
connector.setstr("config_json", '{"hidden_size": 1024}')
config = connector.getstr("config_json")
```

### Model Loading (Future Work)

Currently, the `MooncakeConnector` is primarily designed for programmatic access to the Mooncake store. Integration with `sglang`'s model loader to directly load model weights from Mooncake using `mooncake://` URIs in `--model-path` is possible if the model loader supports the connector interface.

## Limitations

-   **Listing Keys**: The `list()` operation is currently not implemented due to Mooncake's architecture.
-   **File Pulling**: `pull_files()` is not implemented.
