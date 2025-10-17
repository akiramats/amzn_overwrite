#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
python - <<'PY'
from src.utils.ib_client import connect_ib
ib = connect_ib()
print("Managed accounts from IB:", ib.managedAccounts())
ib.disconnect()
PY
