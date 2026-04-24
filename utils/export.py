import argparse
import wandb
import pandas as pd
import os



ENTITY = "lecene18-sorbonne-universit-" 
PROJECT = "ECG_Classification_Experiments"

# Liste des métriques pour lesquelles une valeur minimale indique une meilleure performance.
LOWER_IS_BETTER = [
    "performance/temps_inference_sec",
    "stabilité/hamming_loss_mean",
    "test_metrics/mean_brier_score",
    "binaire_malade/loss"
]

# Textes descriptifs pour les légendes des différents tableaux générés.
CAPTIONS = {
    "bin": "Results for the binary classification task.",
    "stat": "Results for the statistical evaluation task (Format: H/m/L. Model identified by Run ID).",
    "perf": "Results for the computational performance task."
}

# Définition des abréviations insérées dans la légende du tableau statistique.
ABBREVIATIONS_TEXT = r" \textbf{Abbreviations:} H/m/L = High / mean / Low, Chal = Challenge Score, HL = Hamming Loss, AUPRC = Area Under PR Curve, Inf\_Time = Inference Time, Exact\_Acc = Exact Match Accuracy."

# Correspondance pour assurer la rétrocompatibilité lors de l'extraction des anciens runs.
LEGACY_MAPPING = {
    "AUPRC": "mean_auprc",
    "AUROC": "mean_auroc",
    "Accuracy": "acc_exact_match",
    "Challenge_Score": "challenge_score",
    "Macro_F1": "macro_f1"
}

# Dictionnaire de traduction pour optimiser la largeur des en-têtes dans le rendu LaTeX.
RENAMING_DICT = {
    "challenge": "Chal",
    "hamming_loss": "HL",
    "macro_f1": "F1_Macro",
    "acc_exact_match": "Exact_Acc",
    "challenge_score": "Chal_Score",
    "mean_brier_score": "Brier",
    "temps_inference_sec": "Inf_Time",
    "mean_auprc": "AUPRC",
    "mean_auroc": "AUROC"
}


def format_hml(m, l, h, is_best):
    """
    Formate un triplet de valeurs en une chaîne représentant les bornes et la moyenne.

    Args:
        m (float): La valeur moyenne ou centrale.
        l (float): La limite inférieure de l'intervalle de confiance.
        h (float): La limite supérieure de l'intervalle de confiance.
        is_best (bool): Indique si la moyenne représente le meilleur résultat de la colonne.

    Returns:
        str: Une chaîne formatée (High / mean / Low), mise en gras si is_best est True.
    """
    if pd.isna(m): 
        return "-"
    res = f"{h:.3f} / {m:.3f} / {l:.3f}"
    return f"\\textbf{{{res}}}" if is_best else res

def format_single(val, is_best):
    """
    Formate une valeur numérique simple pour l'affichage dans le tableau.

    Args:
        val (float|int): La valeur à formater.
        is_best (bool): Indique si la valeur représente le meilleur résultat de la colonne.

    Returns:
        str: La valeur arrondie à trois décimales, mise en gras si is_best est True.
    """
    if pd.isna(val) or not isinstance(val, (int, float)): 
        return "-"
    formatted = f"{val:.3f}"
    return f"\\textbf{{{formatted}}}" if is_best else formatted

def parse_existing_latex(filepath):
    """
    Analyse un fichier LaTeX existant.

    Note: Dans l'implémentation actuelle gérant les fusions complexes (format H/m/L),
    cette fonction retourne intentionnellement un DataFrame vide. Cela force le
    système à re-télécharger les données depuis l'API W&B, garantissant l'intégrité
    et évitant les conflits de parsing.

    Args:
        filepath (str): Chemin vers le fichier LaTeX à analyser.

    Returns:
        pd.DataFrame: Un objet DataFrame vide.
    """
    if not os.path.exists(filepath): 
        return pd.DataFrame()
    return pd.DataFrame() 

def generate_scientific_latex(df, title, filepath, caption_text, label_name, force_fit=False):
    """
    Génère un fichier texte contenant un environnement LaTeX de type tableau (booktabs).

    Args:
        df (pd.DataFrame): Les données formatées à inclure dans le tableau.
        title (str): Identifiant interne pour nommer la table en commentaire.
        filepath (str): Chemin complet de sauvegarde du fichier de sortie.
        caption_text (str): Texte destiné à la commande LaTeX \caption.
        label_name (str): Identifiant destiné à la commande LaTeX \label.
        force_fit (bool, optionnel): Si défini sur True, encapsule le tableau dans un
            \resizebox pour s'assurer qu'il respecte la largeur de la page.
    """
    if df.empty: 
        return
    
    clean_headers = [RENAMING_DICT.get(c, c).replace('_', r'\_') for c in df.columns]
    col_format = "l" + "c" * (len(df.columns) - 1)
    
    latex_lines = [
        f"% --- Table: {title} ---",
        f"% Requires in preamble: \\usepackage{{booktabs}}" + (" and \\usepackage{graphicx}" if force_fit else ""),
        r"\begin{table}[htbp]",
        r"\centering"
    ]
    
    if force_fit:
        latex_lines.append(r"\resizebox{\textwidth}{!}{%")
        
    latex_lines.extend([
        f"\\begin{{tabular}}{{{col_format}}}",
        r"\toprule",
        " & ".join(clean_headers) + r" \\",
        r"\midrule"
    ])

    for _, row in df.iterrows():
        row_strs = [str(val).replace('_', r'\_') if i == 0 else str(val) for i, val in enumerate(row)]
        latex_lines.append(" & ".join(row_strs) + r" \\")

    latex_lines.append(r"\bottomrule")
    latex_lines.append(r"\end{tabular}")

    if force_fit:
        latex_lines.append(r"}")

    latex_lines.extend([
        r"\vspace{0.2cm}",
        f"\\caption{{{caption_text}}}",
        f"\\label{{{label_name}}}",
        r"\end{table}"
    ])

    with open(filepath, 'w') as f:
        f.write("\n".join(latex_lines) + "\n")
    print(f"Generated: {filepath}")


def main():
    """
    Point d'entrée principal du script.

    Gère les arguments de la ligne de commande, télécharge les résumés des exécutions
    W&B requises, agrège les données en fonction des domaines d'évaluation (bin, stat, perf),
    et orchestre l'application des formats et la génération des fichiers cibles.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', "--run_ids", nargs='*', default=[])
    parser.add_argument("--output_dir", type=str, default="../../output/output_latex")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    api = wandb.Api()

    # Phase 1: Accumulation des données brutes
    runs_data = []
    if args.run_ids:
        for rid in args.run_ids:
            try:
                run = api.run(f"{ENTITY}/{PROJECT}/{rid}")
                runs_data.append({"id": rid, "name": run.name, "summary": run.summary})
            except Exception as e: 
                print(f"Error {rid}: {e}")

    # Phase 2: Traitement par catégorie de tableau
    for key in ["bin", "stat", "perf"]:
        rows = []
        for r in runs_data:
            s = r["summary"]
            row = {"Model": r["id"] if key == "stat" else r["name"]}

            metrics_subset = {}
            prefix = {"bin": "binaire_malade/", "stat": "stabilité/", "perf": "performance/"}[key]

            # Extraction et filtrage des métriques
            for k, v in s.items():
                if k.startswith(prefix) and isinstance(v, (int, float)):
                    metrics_subset[k.replace(prefix, "")] = v

                elif key == "stat":
                    if k.startswith("test_metrics/") and isinstance(v, (int, float)):
                        metrics_subset[k.replace("test_metrics/", "")] = v
                    elif k in LEGACY_MAPPING and isinstance(v, (int, float)):
                        standard_key = LEGACY_MAPPING[k]
                        metrics_subset[standard_key] = v

            # Agrégation spécifique pour les intervalles de confiance (Tableau statistique)
            if key == "stat":
                base_metrics = set(k.replace("_mean", "").replace("_ci_lower", "").replace("_ci_upper", "") 
                                  for k in metrics_subset.keys() if any(suffix in k for suffix in ["_mean", "_ci_l", "_ci_u"]))
                
                final_metrics = {}
                for bm in base_metrics:
                    m, l, h = metrics_subset.get(bm+"_mean"), metrics_subset.get(bm+"_ci_lower"), metrics_subset.get(bm+"_ci_upper")
                    if m is not None: 
                        final_metrics[bm] = (m, l, h)

                for k, v in metrics_subset.items():
                    if not any(suffix in k for suffix in ["_mean", "_ci_l", "_ci_u"]):
                        final_metrics[k] = v
                row.update(final_metrics)
            else:
                row.update(metrics_subset)

            rows.append(row)

        df = pd.DataFrame(rows)
        if df.empty or len(df.columns) <= 1: 
            continue

        # Phase 3: Formatage final et détection des valeurs optimales
        df_formatted = pd.DataFrame()
        df_formatted["Model"] = df["Model"]

        for col in df.columns:
            if col == "Model": 
                continue

            if isinstance(df[col].iloc[0], tuple):
                means = pd.Series([x[0] if isinstance(x, tuple) else None for x in df[col]])
                best_val = means.min() if (key+"/"+col+"_mean") in LOWER_IS_BETTER else means.max()
                df_formatted[col] = df[col].apply(lambda x: format_hml(x[0], x[1], x[2], x[0] == best_val) if isinstance(x, tuple) else "-")

            else:
                numeric_col = pd.to_numeric(df[col], errors='coerce')
                best_val = numeric_col.min() if (key+"/"+col) in LOWER_IS_BETTER else numeric_col.max()
                df_formatted[col] = numeric_col.apply(lambda x: format_single(x, x == best_val))

        caption = CAPTIONS[key]
        force_fit = False
        if key == "stat":
            caption += ABBREVIATIONS_TEXT
            force_fit = True

        generate_scientific_latex(df_formatted, key.upper(), os.path.join(args.output_dir, f"tab_{key}.tex"), 
                                 caption, f"tab:{key}", force_fit=force_fit)

if __name__ == "__main__":
    main()
