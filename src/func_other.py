import os
import re

import pandas as pd
import pandas.api.types

############# SCORE - ADAPTED FROM KAGGLE https://www.kaggle.com/code/metric/ribonanza-tm-score #############


def parse_tmscore_output(output):
    # Extract TM-score based on length of reference structure (second)
    tm_score_match = re.findall(r'TM-score=\s+([\d.]+)', output)[1]
    if not tm_score_match:
        raise ValueError('No TM score found')
    return float(tm_score_match)


def write_target_line(
    atom_name, atom_serial, residue_name, chain_id, residue_num, x_coord, y_coord, z_coord, occupancy=1.0, b_factor=0.0, atom_type='P'
) -> str:
    """
    Writes a single line of PDB format based on provided atom information.

    Args:
        atom_name (str): Name of the atom (e.g., "N", "CA").
        atom_serial (int): Atom serial number.
        residue_name (str): Residue name (e.g., "ALA").
        chain_id (str): Chain identifier.
        residue_num (int): Residue number.
        x_coord (float): X coordinate.
        y_coord (float): Y coordinate.
        z_coord (float): Z coordinate.
        occupancy (float, optional): Occupancy value (default: 1.0).
        b_factor (float, optional): B-factor value (default: 0.0).

    Returns:
        str: A single line of PDB string.
    """
    return f'ATOM  {atom_serial:>5d}  {atom_name:<5s} {residue_name:<3s} {residue_num:>3d}    {x_coord:>8.3f}{y_coord:>8.3f}{z_coord:>8.3f}{occupancy:>6.2f}{b_factor:>6.2f}           {atom_type}\n'


def write2pdb(df: pd.DataFrame, xyz_id: str, target_path: str) -> int:
    resolved_cnt = 0
    with open(target_path, 'w') as target_file:
        for _, row in df.iterrows():
            x_coord = row[f'x_{xyz_id}']
            y_coord = row[f'y_{xyz_id}']
            z_coord = row[f'z_{xyz_id}']

            if x_coord > -1e17 and y_coord > -1e17 and z_coord > -1e17:
                # if True:
                resolved_cnt += 1
                target_line = write_target_line(
                    atom_name="C1'",
                    atom_serial=int(row['resid']),
                    residue_name=row['resname'],
                    chain_id='0',
                    residue_num=int(row['resid']),
                    x_coord=x_coord,
                    y_coord=y_coord,
                    z_coord=z_coord,
                    atom_type='C',
                )
                target_file.write(target_line)
    return resolved_cnt


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    """
    Computes the TM-score between predicted and native RNA structures using USalign.

    This function evaluates the structural similarity of RNA predictions to native structures
    by computing the TM-score. It uses USalign, a structural alignment tool, to compare
    the predicted structures with the native structures.

    Workflow:
    1. Copies the USalign binary to the working directory and grants execution permissions.
    2. Extracts the `target_id` from the `ID` column of both the solution and submission DataFrames.
    3. Iterates over each unique `target_id`, grouping the native and predicted structures.
    4. Writes PDB files for native and predicted structures.
    5. Runs USalign on each predicted-native pair and extracts the TM-score.
    6. Computes the highest TM-score per target and returns aggregated results.

    Args:
        solution (pd.DataFrame): A DataFrame containing the native RNA structures.
        submission (pd.DataFrame): A DataFrame containing the predicted RNA structures.
        row_id_column_name (str): The name of the column containing unique row identifiers.

    Returns:
        float: the average highest TM-scores.
    """

    os.system('cp //kaggle/input/usalign/USalign /kaggle/working/')
    os.system('sudo chmod u+x /kaggle/working//USalign')

    # Extract target_id from ID (target_resid)
    solution['target_id'] = solution['ID'].apply(lambda x: x.split('_')[0])
    submission['target_id'] = submission['ID'].apply(lambda x: x.split('_')[0])

    results = []
    # Iterate through each target_id and generate PDB files for both clean and corrupted data
    for target_id, group_native in solution.groupby('target_id'):
        group_predicted = submission[submission['target_id'] == target_id]
        native_pdb = 'native.pdb'
        predicted_pdb = 'predicted.pdb'

        target_id_scores = []
        for pred_cnt in range(1, 6):
            prediction_scores = []
            for native_cnt in range(1, 41):
                # Write solution PDB
                resolved_cnt = write2pdb(group_native, native_cnt, native_pdb)

                # Write predicted PDB
                _ = write2pdb(group_predicted, pred_cnt, predicted_pdb)

                if resolved_cnt > 0:
                    command = f'/kaggle/working/USalign {predicted_pdb} {native_pdb} -atom " C1\'"'
                    usalign_output = os.popen(command).read()
                    prediction_scores.append(parse_tmscore_output(usalign_output))

            target_id_scores.append(max(prediction_scores))
        results.append(max(target_id_scores))

    return float(sum(results) / len(results))