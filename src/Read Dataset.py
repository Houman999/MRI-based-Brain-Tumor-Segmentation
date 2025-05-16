# Constants 

# Classes 
SEGMENT_CLASSES = {
    0: 'NOT tumor',
    1: 'NECROTIC/CORE',
    2: 'EDEMA',
    3: 'ENHANCING'
}

# Volume Slices
VOLUME_SLICES = 100

# Volume Start ...
VOLUME_START_AT = 22

# Load Sample Multi_modal Data 
def load_sample_images(case_id):
    case_path = os.path.join(TRAIN_PATH , case_id)
    def load_nii(name): return nib.load(os.path.join(case_path , f"{case_id}_{name}.nii")).get_fdata()
    return{mod: load_nii(mod) for mod in ['flair', 't1', 't1ce', 't2', 'seg']}

# Display Image Information
def display_image_info(images):
    for mod, img in images.items():
        print(f"Modality: {mod}")
        print(f"Shape: {img.shape} (Z, X, Y)")
        print(f"Number of slices: {img.shape[0]}")
        print(f"Resolution: {img.shape[1]} x {img.shape[2]}\n")

SAMPLE_ID = 'BraTS20_Training_001'

sample_imgs = load_sample_images(SAMPLE_ID)
display_image_info(sample_imgs)

# Dic and ID Handeling 
def get_valid_ids(path):
    all_dirs = [f.path for f in os.scandir(path) if f.is_dir()]
    all_dirs = [ d for d in all_dirs if not d.endswith("BraTS20_Training_355")] # this patient is bad case(file_name) for process 
    return [os.path.basename(d) for d in all_dirs]

train_val_ids = get_valid_ids(TRAIN_PATH)

# Spliting Train and Test
train_test_ids ,val_ids = train_test_split(train_val_ids , test_size =0.2 , random_state =42)
train_ids , test_ids = train_test_split(train_test_ids , test_size =0.15 , random_state =42)