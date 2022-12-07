import settings
import os
from PIL import Image
import numpy as np

class EmojiOutputFormat:
    @staticmethod
    def original(image):
        return image
    
    @staticmethod
    def grayscale(image):
        return image.convert(mode="L")

    @staticmethod
    def RGB(image):
        return image.convert(mode="RGB")

    @staticmethod
    def RGBA(image):
        return image.convert(mode="RGBA")

class Emoji:
    def __init__(self, data_folder: str, company_name: str="", output_size: tuple=(0,0), output_format= EmojiOutputFormat.original,resize_mode: int=0):
        """
        Takes an company name as an input, load the data from .data folder. The company name is default to the data folder name.
        """
        self.dir: str = data_folder
        self.name: str = company_name
        self.pic_file_names = self._load()
        self.output_size = output_size
        self.post_process = output_format
        self.resize_mode = resize_mode
        
        self._index: int = 0
    
    # API
    def count(self) -> int:
        return len(self.pic_file_names)
    
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
        im = self.post_process(im)
        if self.output_size == (0,0) or self.output_size == im.size:
            pixels = np.array(list(im.getdata()))                
            # Returns ARGB value
        else:
            # Resize before return
            pixels = np.array(list(im.resize(self.output_size, mode=self.resize_mode).getdata()))                
            # Returns ARGB value
        return pixels, self.pic_file_names[self._index - 1]

    def __iter__(self):
        return self

    