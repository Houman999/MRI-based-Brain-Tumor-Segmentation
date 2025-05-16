# Using data through this pipeline for Model
class DataGenerator(Sequence):
    def __init__(self, list_IDs, dim=(128, 128), batch_size=1, n_channels=2, shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return len(self.list_IDs) // self.batch_size

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        batch_ids = [self.list_IDs[k] for k in indexes]
        X, Y = self.__data_generation(batch_ids)
        return X, Y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_ids):
        X = np.zeros((self.batch_size * VOLUME_SLICES, *self.dim, self.n_channels))
        y = np.zeros((self.batch_size * VOLUME_SLICES, 240, 240))

        for c, case_id in enumerate(batch_ids):
            case_path = os.path.join(TRAIN_PATH, case_id)
            flair = nib.load(os.path.join(case_path, f'{case_id}_flair.nii')).get_fdata()
            t1ce = nib.load(os.path.join(case_path, f'{case_id}_t1ce.nii')).get_fdata()
            seg = nib.load(os.path.join(case_path, f'{case_id}_seg.nii')).get_fdata()

            for j in range(VOLUME_SLICES):
                idx = j + VOLUME_SLICES * c
                X[idx, :, :, 0] = cv2.resize(flair[:, :, j + VOLUME_START_AT], self.dim)
                X[idx, :, :, 1] = cv2.resize(t1ce[:, :, j + VOLUME_START_AT], self.dim)
                y[idx] = seg[:, :, j + VOLUME_START_AT]

        y[y == 4] = 3  # Map label 4 to 3
        mask = tf.one_hot(y.astype(np.uint8), 4)
        Y = tf.image.resize(mask, self.dim)
        return X / np.max(X), Y

