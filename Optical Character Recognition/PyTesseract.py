try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract
import re
import numpy as np 
import os 
import pandas as pd


##############################################################################################################
#   User-defined variables
##############################################################################################################

image_directory = (r'C:\Users\rk613ke\Pictures')
excel_output_directory = (r'C:\Users\rk613ke\Desktop\2 - Ginnie\Python\TEXT_EXTRACT_SAMPLE.xlsx')

##############################################################################################################

##############################################################################################################

# Simple image to string
filename = 'loans.png'
image = (os.path.join(image_directory, filename))
extract = pytesseract.image_to_string(Image.open(image))

# Replace Line Breaks with comma's
extract = (re.sub("\n", ",", extract))

#Seperate strings into a list 
extract = extract.split(",")


#Convert the list into a df (This was based on personal preferance )
df = pd.DataFrame(extract)

# Replace blanks with nan values and drop the nulls
df = df.replace('', np.nan, regex=True)
df = df.dropna(how= 'all')


# Write the data frame to an excel file
df.to_excel(excel_output_directory, engine='xlsxwriter', index = False)
        
        
    
