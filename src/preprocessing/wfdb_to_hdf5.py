import wfdb
import h5py
import numpy as np
from pathlib import Path



def wdfb_to_hdf5(dataset_path, output_file):
    print(f"Processing: {dataset_path}")

    # Get all group folders in the dataset
    g_folders = sorted(dataset_path.glob("g*"))

    exam_ids = []
    signals = []
    max_n_samples = 0

    # Get the max n_samples across all records in the dataset
    for g in g_folders:
        for hea_file in sorted(g.glob("*.hea")):
            record_name = hea_file.stem # Get file name as exam id
            record_path = str(hea_file.with_suffix("")) # Get path without extension

            record = wfdb.rdrecord(record_path)
            # Decode ECG signal in mV, shape=(n_samples, n_leads)
            signal = record.p_signal.astype(np.float32)

            exam_ids.append(record_name)
            signals.append(signal)

            if signal.shape[0] > max_n_samples:
                max_n_samples = signal.shape[0]

    # Debugging info
    n_ecg = len(signals)
    n_leads = signals[0].shape[1]
    print(f"Number of ECGs: {n_ecg}")
    print(f"Max number samples in dataset: {max_n_samples}")
    print(f"Number of leads: {n_leads}")

    # Initialize tracings with zeros
    tracings = np.zeros(
        (n_ecg, max_n_samples, n_leads),
        dtype=np.float32
    )

    # Fill in the tracings with signals of varying lengths
    for i, sig in enumerate(signals):
        n_samples = sig.shape[0]
        tracings[i, :n_samples, :] = sig

    # Create hdf5 file
    with h5py.File(output_file, "w") as f:
        f.create_dataset(
            "exam_id",
            data=np.array(exam_ids, dtype=("S")) # p.s. in hdf5 the strings are stored as bytes  
        )
        f.create_dataset(
            "tracings",
            data=tracings
        )

    print(f"Saving hdf5 file: {out_file}")


TRAIN_ROOT = Path("PATH TO DATASETS") # e.g., data/physioNet/training
OUT_ROOT = Path("PATH TO OUTPUT FOLDER") # e.g., data/hdf5_output
OUT_ROOT.mkdir(exist_ok=True)

# Loop through each dataset 
for dataset_dir in sorted(TRAIN_ROOT.iterdir()):
    if not dataset_dir.is_dir():
        continue
    out_file = OUT_ROOT / f"{dataset_dir.name}.hdf5"
    wdfb_to_hdf5(dataset_dir, out_file)

