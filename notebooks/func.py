import os
import re

import pandas as pd
import pandas.api.types

############# SCORE - ADAPTED FROM KAGGLE https://www.kaggle.com/code/metric/ribonanza-tm-score #############


def parse_tmscore_output(output):
    # grab all “TM-score= <number>” matches
    scores = re.findall(r"TM-score=\s+([\d.]+)", output)
    # if USalign failed or printed only one score, just return 0.0
    return float(scores[1]) if len(scores) > 1 else 0.0


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    # 1) point to your local USalign binary (same folder as this file)
    USALIGN_LOCAL = os.path.join(os.path.dirname(__file__), "USalign")

    # 2) make sure it's executable
    if not os.access(USALIGN_LOCAL, os.X_OK):
        st = os.stat(USALIGN_LOCAL)
        os.chmod(USALIGN_LOCAL, st.st_mode | stat.S_IXUSR)

    # 3) extract the grouping key
    solution["target_id"]   = solution["ID"].str.split("_", 1).str[0]
    submission["target_id"] = submission["ID"].str.split("_", 1).str[0]

    results = []
    for target_id, group_native in solution.groupby("target_id"):
        group_predicted = submission[submission["target_id"] == target_id]
        best_this_target = 0.0

        for pred_idx in range(1, 6):
            best_for_pred = 0.0
            for nat_idx in range(1, 41):
                n_written = write2pdb(group_native,   nat_idx, "native.pdb")
                p_written = write2pdb(group_predicted, pred_idx, "predicted.pdb")

                if n_written > 0 and p_written > 0:
                    cmd = f"{USALIGN_LOCAL} predicted.pdb native.pdb -atom \" C1'\""
                    out = os.popen(cmd).read()
                    score = parse_tmscore_output(out)
                    best_for_pred = max(best_for_pred, score)

            best_this_target = max(best_this_target, best_for_pred)

        results.append(best_this_target)

    return float(sum(results) / len(results))
