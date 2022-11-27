import settings
import os

class Emoji:
    def __init__(self, data_folder: str, company_name: str=""):
        """
        Takes an company name as an input, load the data from .data folder. The company name is default to the data folder name.
        """
        pass
    
    # API
    
    
    # Subprocess
    def _load(self):
        pass

    def __next__(self):
        pass

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


    