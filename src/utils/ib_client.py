from ib_insync import IB
from dotenv import load_dotenv
import os, sys

def connect_ib():
    load_dotenv(override=False)
    host = os.getenv("IB_HOST", "127.0.0.1")
    port = int(os.getenv("IB_PORT", "4002"))  # Paper default
    client_id = int(os.getenv("IB_CLIENT_ID", "1"))

    ib = IB()
    try:
        ib.RequestTimeout = 45
        ib.connect(host, port, clientId=client_id, timeout=15)
    except Exception as e:
        print(f"[IB CONNECT ERROR] host={host} port={port} clientId={client_id}\n{e}")
        sys.exit(1)

    # 1=live, 2=frozen, 3=delayed, 4=delayed-frozen
    ib.reqMarketDataType(4)
    ib.sleep(0.5)
    return ib
