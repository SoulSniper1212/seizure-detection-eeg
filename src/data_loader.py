import os
import mne

def load_chbmit_patient(data_dir, patient_id='chb01'):
    patient_path = os.path.join(data_dir, patient_id)
    edf_files = [f for f in os.listdir(patient_path) if f.endswith('.edf')]
    
    eeg_data = []
    raw_objects = []
    
    for edf_file in sorted(edf_files):
        full_path = os.path.join(patient_path, edf_file)
        raw = mne.io.read_raw_edf(full_path, preload=True, verbose=False)
        raw.pick_types(eeg=True)
        raw.filter(0.5, 50, fir_design='firwin')
        eeg_data.append(raw.get_data())
        raw_objects.append(raw)
    
    return eeg_data, raw_objects
