import h5py
import numpy as np
import pandas as pd



filename = '../../data/15_prct/exams_part0.hdf5'

print(f"Ouverture du fichier : {filename}")
print("="*60)

with h5py.File(filename, 'r') as f:
    # 1. Regarde les clefs des données
    print("Clés trouvées à la racine :", list(f.keys()))

    tracings = f['tracings']
    ids = f['exam_id']

    # 2. Annalyse des shapes
    print("\n--- ANALYSE DES DIMENSIONS ---")
    print(f"Dataset 'tracings' shape : {tracings.shape}")
    print(f"Dataset 'exam_id' shape  : {ids.shape}")

    N_samples = tracings.shape[0]
    dim_1 = tracings.shape[1]
    dim_2 = tracings.shape[2] if len(tracings.shape) > 2 else 1

    print(f"-> Nombre d'échantillons : {N_samples}")

    # 3. ANALYSE DU CONTENU (statistique)
    data = tracings
    patient_id = ids

    print(f"\n--- ZOOM SUR LE PATIENT ID {patient_id} ---")
    print(f"Type de données (dtype) : {data.dtype}")
    print(f"Shape de l'échantillon  : {data.shape}")

    # Vérification des valeurs (Pour savoir si on doit normaliser ou autre)
    print(f"Valeur Min : {np.min(data):.4f}")
    print(f"Valeur Max : {np.max(data):.4f}")
    print(f"Moyenne    : {np.mean(data):.4f}")
    print(f"Écart-type : {np.std(data):.4f}")

    # Vérification de NaN
    has_nan = np.isnan(data).any()
    print(f"Contient des NaN ? : {'OUI' if has_nan else 'Non'}")



# Annlayse du CSV examen.csv
filename = '../../data/15_prct/exams.csv'

print(f"\n\nOuverture du fichier : {filename}")
print("="*60)

# 1. Chargement des données
print("Chargement du fichier CSV...")
df = pd.read_csv(filename)

# Conversion des colonnes booléennes qui pourraient être en texte
bool_cols = ['is_male', '1dAVb', 'RBBB', 'LBBB', 'SB', 'AF', 'ST', 'normal_ecg', 'death']
df[bool_cols] = df[bool_cols].astype(bool)

print("="*50)
print(f"RAPPORT STATISTIQUE DU DATASET ({len(df)} examens)")
print("="*50)

# --- 2. Démographie (Age & Sexe) ---
print("\n--- 1. DÉMOGRAPHIE ---")
mean_age = df['age'].mean()
std_age = df['age'].std()
print(f"Age moyen          : {mean_age:.1f} ans (± {std_age:.1f})")
print(f"Age Min / Max      : {df['age'].min()} / {df['age'].max()}")

perc_male = df['is_male'].mean() * 100
print(f"Répartition Hommes : {perc_male:.1f}%")
print(f"Répartition Femmes : {100 - perc_male:.1f}%")

# --- 3. État Normal ---
print("\n--- 2. GLOBAL ---")
perc_normal = df['normal_ecg'].mean() * 100
print(f"ECG Normaux        : {perc_normal:.1f}%")
print(f"ECG Anormaux       : {100 - perc_normal:.1f}%")

# --- 4. Prévalence des Maladies ---
print("\n--- 3. DÉTAIL DES PATHOLOGIES ---")
diseases = ['1dAVb', 'RBBB', 'LBBB', 'SB', 'AF', 'ST']

for d in diseases:
    count = df[d].sum()
    perc = df[d].mean() * 100
    print(f"- {d.ljust(6)} : {perc:.1f}%  ({count} cas)")

# --- 5. Mortalité ---
print("\n--- 4. MORTALITÉ ---")
# La colonne death est vide pour certains examens, on filtre les NaN
# Note: Le dataset dit que c'est dispo sur le "first exam du patient"
death_data = df.dropna(subset=['death'])
perc_death = death_data['death'].mean() * 100
print(f"Taux de mortalité  : {perc_death:.1f}% (sur {len(death_data)} patients suivis)")

print("="*50)
