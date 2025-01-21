import os
import shutil
from sklearn.model_selection import train_test_split


def split_dataset_into_train_val_test(input_dir, output_dir, train_size=0.85, val_size=0.05, test_size=0.1):
    os.makedirs(output_dir, exist_ok=True)

    for category in os.listdir(input_dir):
        category_path = os.path.join(input_dir, category)

        if os.path.isdir(category_path):
            files = [f for f in os.listdir(category_path) if os.path.isfile(os.path.join(category_path, f))]

            train_files, temp_files = train_test_split(files, train_size=train_size, random_state=42)
            val_files, test_files = train_test_split(
                temp_files, test_size=test_size / (test_size + val_size), random_state=42
            )

            for split, split_files in zip(['train', 'val', 'test'], [train_files, val_files, test_files]):
                split_dir = os.path.join(output_dir, split, category)
                os.makedirs(split_dir, exist_ok=True)
                for file in split_files:
                    src_file = os.path.join(category_path, file)
                    dest_file = os.path.join(split_dir, file)
                    shutil.copy(src_file, dest_file)

input_directory = 'hair'
output_directory = 'data'
split_dataset_into_train_val_test(input_directory, output_directory)
