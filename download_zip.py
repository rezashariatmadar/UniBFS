import urllib.request
import zipfile
import io
import os
from pathlib import Path

url = "https://ndownloader.figshare.com/files/13904870"
out_dir = Path("data")
out_dir.mkdir(exist_ok=True)

print("Downloading Microarray_data_MAT.zip from Figshare...")
try:
    req = urllib.request.Request(
        url, 
        headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    )
    with urllib.request.urlopen(req) as response:
        zip_data = response.read()
    
    print("Extracting zip to data/...")
    with zipfile.ZipFile(io.BytesIO(zip_data)) as z:
        z.extractall(out_dir)
        print("Extracted files:")
        for name in z.namelist():
            print(f"  {name}")
            
except Exception as e:
    print(f"Error: {e}")
