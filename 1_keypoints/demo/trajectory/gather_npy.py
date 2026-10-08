import os
import shutil

def copy_and_rename_npy_files(parent_folder, destination_folder):

    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)


    for root, dirs, files in os.walk(parent_folder):

        for file in files:
            if file.endswith('tri_filter.npy'):

                file_path = os.path.join(root, file)


                parent_folder_name = os.path.basename(root)


                new_file_name = parent_folder_name + '.npy'
                new_file_path = os.path.join(destination_folder, new_file_name)


                print(f"Copying {file_path} to {new_file_path}")
                shutil.copy2(file_path, new_file_path)


parent_folder = os.environ['MOUSEGPT_DATA_ROOT']
destination_folder = os.environ['MOUSEGPT_KEYPOINTS_DIR']
copy_and_rename_npy_files(parent_folder, destination_folder)
