import os
import shutil
import random
from pathlib import Path

def select_random_files(src_dir, dest_dir, percentage=0.1):


    Path(dest_dir).mkdir(parents=True, exist_ok=True)

    for domain in os.listdir(src_dir):
        domain_path = os.path.join(src_dir, domain)
        if os.path.isdir(domain_path):

            domain_dest_path = os.path.join(dest_dir, domain)
            Path(domain_dest_path).mkdir(parents=True, exist_ok=True)

            for category in os.listdir(domain_path):
                category_path = os.path.join(domain_path, category)
                if os.path.isdir(category_path):

                    category_dest_path = os.path.join(domain_dest_path, category)
                    Path(category_dest_path).mkdir(parents=True, exist_ok=True)

                    image_files = [f for f in os.listdir(category_path) if os.path.isfile(os.path.join(category_path, f))]

                    num_images_to_select = max(1, int(len(image_files) * percentage))

                    selected_images = random.sample(image_files, num_images_to_select)

                    for image in selected_images:
                        src_image_path = os.path.join(category_path, image)
                        dest_image_path = os.path.join(category_dest_path, image)
                        shutil.copy2(src_image_path, dest_image_path)
                    

src_dir = './data/office_home/'

dest_dir = './sub_data/office_home/'

select_random_files(src_dir, dest_dir, percentage=0.1)
