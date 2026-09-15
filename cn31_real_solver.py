import os
import sys

# Point token saves at your necap server
os.environ.setdefault(
    "https://cn31-server-production-020c.up.railway.app/domain",
    os.environ.get("https://cn31-server-production-020c.up.railway.app/domain", "http://localhost:6000")
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from yidun_proxyless import main, initialize_global_model, get_compiled_js
except ImportError as e:
    print(f"[FATAL] Cannot import solver: {e}")
    sys.exit(1)

if __name__ == "__main__":
    print("[CN31] Starting direct solver...")
    main()