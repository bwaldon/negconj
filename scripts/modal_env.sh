# Source this to export MODAL_TOKEN_ID / MODAL_TOKEN_SECRET from ./.env (values never echoed).
eval "$(python3 - <<'PY'
import shlex, pathlib
for line in pathlib.Path(".env").read_text().splitlines():
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    k, v = (x.strip() for x in line.split("=", 1))
    v = v.strip("'\"")
    if k.startswith("MODAL_"):
        print(f"export {k}={shlex.quote(v)}")
PY
)"
