import pandas as pd
import os



# --- CONFIGURATION ---
input_csv = '../../../data/15_prct/exams.csv'
column_to_split = 'trace_file'
output_folder = '../../../data/15_prct/'

try:
    print("1. Chargement du fichier...")
    df = pd.read_csv(input_csv)
    
    df.columns = df.columns.str.strip()

    print(f"   -> {len(df)} lignes chargées.")
    print(f"   -> Colonnes : {df.columns.tolist()}")

    print(f"2. Découpage basé sur : {column_to_split}...")

    for trace_name, group_df in df.groupby(column_to_split):
        # 1. Nettoyage du nom (transtypage en string + strip)
        raw_name = str(trace_name).strip()

        safe_filename = raw_name.replace('.hdf5', '.csv')

        output_path = os.path.join(output_folder, safe_filename)

        # Sauvegarde
        group_df.to_csv(output_path, index=False)
        print(f"   -> Fichier créé : {safe_filename} ({len(group_df)} lignes)")

except Exception as e:
    print(f"\nERREUR : {e}")
