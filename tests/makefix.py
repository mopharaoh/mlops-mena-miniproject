# شغّله مرة واحدة: python scripts/make_fixture.py  (أو أي مكان مؤقت)
import json
from pathlib import Path

import numpy as np
import pandas as pd

from prodml.config import settings

df = pd.read_csv(settings.data_path)

row = (
    df.drop(columns=["Id", settings.target_column])
    .iloc[0]
    .replace({np.nan: None})  # NaN بقت null زي ما الموديل شايفها
)

payload = {k: (v.item() if isinstance(v, np.generic) else v) for k, v in row.items()}

out = Path("tests/fixtures/house.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

print(f"{len(payload)} features written to {out}")
