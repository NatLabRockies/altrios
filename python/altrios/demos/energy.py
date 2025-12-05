import subprocess
import math
import pandas as pd
import re

from altrios.demos.single_train_sim import corridor

results = []

for L in range(20, 181, 20):
    n_locos = math.ceil(L * 49.3 * 3 / 4400)

    print(f"Running simulation for train length {L} cars, {n_locos} locomotives...")

    result = subprocess.run(
        ["python", "single_train_sim.py", str(L), str(n_locos)],
        capture_output=True,
        text=True
    )

    stdout = result.stdout

    m_energy = re.search(r"Total SOC-corrected fuel used.*?:\s*([\d.]+)", stdout)
    energy_GJ = float(m_energy.group(1)) if m_energy else None

    m_idx = re.search(r"MAX_IDX:(\d+)", stdout)
    m_time = re.search(r"MAX_TIME:([\d\.]+)", stdout)
    m_res = re.search(r"MAX_RES:([\d\.]+)", stdout)

    max_idx = int(m_idx.group(1)) if m_idx else None
    max_time = round(float(m_time.group(1)), 2) if m_time else None
    max_res = round(float(m_res.group(1)), 2) if m_res else None

    results.append({
        "train_length": L,
        "# of locos": n_locos,
        "total_energy_GJ": energy_GJ,
        "max_idx": max_idx,
        "max_time_hr": max_time,
        "max_res_newtons": max_res,
    })

df = pd.DataFrame(results)
df.to_csv(f"{corridor}_batch_energy_results.csv", index=False)
print("\n Saved results to energy_results.csv")
print(df)