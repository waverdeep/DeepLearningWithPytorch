import os
import glob

# find all dataset filepath
def get_all_file_path(input_dir, file_extension):
    temp = glob.glob(os.path.join(input_dir, '**', '*.{}'.format(file_extension)), recursive=True)
    return temp

def create_directory(directory_path):
    try:
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
    except OSError:
        print ('Error: Creating directory. ' +  directory_path)


def create_new_path(original_path, new_path):
    if original_path[-1] != '/':
        original_path = '{}/'.format(original_path)

    return os.path.join(original_path, new_path)