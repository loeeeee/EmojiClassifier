# Emoji Classifier

## Intro

This is a course project for STATS 302. We want to classify the pictures of emoji from different companies. The project goal is sample. We want to create a model that could tell which company does the given emoji picture belongs to. The thing is that the emoji is picture represented by the UTF-8, and different devices from different companies would render them in a different manner. You could tell one emoji is from Apple, and another one is from Samsung by looking at their styles.

## Quick start

- Run settings.py. It will create necessary config and data folder for you.
- Extract the three emoji compressed file into the newly created .data folder.
- Change the name of the folder to *Microsoft, Meta, Apple*.
    - The folder structure: 

          --- .data
              --- Meta
                --- [images]
              --- Microsoft
              --- Apple              
                             
- **or** *(Advance)* Change the batch loading function inside helper function to point it to the location you want to.
- The environment is now set up
- Sorry for unable to provide a requirements.txt, but I did not use any wired package.

## License

MIT I guess?