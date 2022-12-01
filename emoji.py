import settings
import os
from PIL import Image
import numpy as np

class Emoji:
    def __init__(self, data_folder: str, company_name: str="", output_size: tuple=(0,0)):
        """
        Takes an company name as an input, load the data from .data folder. The company name is default to the data folder name.
        """
        self.dir: str = data_folder
        self.name: str = company_name
        self.pic_file_names = self._load()
        self.output_size = output_size

        self._index: int = 0
    
    # API
    
    
    # Subprocess
    def _load(self) -> list:
        all_content = os.listdir(self.dir)
        valid_content = []
        for i in all_content:
            if i.endswith((".jpg", ".jpeg", ".png")) and \
                os.path.isfile(os.path.join(self.dir,i)):
                valid_content.append(i)
        return valid_content

    def __next__(self):
        self._index += 1
        if self._index > len(self.pic_file_names):
            raise StopIteration

        # Read image
        im = Image.open(os.path.join(self.dir, self.pic_file_names[self._index - 1]))
        if self.output_size == (0,0):
            pixels = np.array(im.getdata())
            return pixels
        else:
            pass
        return 

    def __iter__(self):
        return self

    # Helper
    @staticmethod
    def picture_to_list(picture):
        """
        Convert the picture to list of RGB value
        """
        pass

    @staticmethod
    def picture_resize(picture, target_size: tuple=(72,72), method: str= "closet neighbor"):
        """
        Convert the picture to target size.
        """
        pass


    