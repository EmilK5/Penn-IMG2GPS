import os
import shutil
import pandas as pd
import exifread
from PIL import Image
import pillow_heif  
from sklearn.model_selection import train_test_split

SOURCE_FOLDER = ""
OUTPUT_FOLDER = ""
SPLITS = {"train": 0.8, "test": 0.1, "val": 0.1}

# --- 1. COORDINATE CONVERSION ---
def convert_to_decimal_degrees(value):
    # Converts GPS degrees, minutes, seconds to decimal format
    d, m, s = value.values
    return d.num / d.den + (m.num / m.den) / 60 + (s.num / s.den) / 3600

def get_gps_metadata(image_path):
    # Extracts Lat/Lon using exifread logic
    with open(image_path, 'rb') as f:
        tags = exifread.process_file(f)
    
    lat_tag = tags.get('GPS GPSLatitude')
    lat_ref = tags.get('GPS GPSLatitudeRef')
    lon_tag = tags.get('GPS GPSLongitude')
    lon_ref = tags.get('GPS GPSLongitudeRef')

    if lat_tag and lon_tag:
        lat = convert_to_decimal_degrees(lat_tag)
        lon = convert_to_decimal_degrees(lon_tag)
        
        # Adjust for hemisphere
        if lat_ref.values[0] == 'S': lat = -lat
        if lon_ref.values[0] == 'W': lon = -lon
        return lat, lon
    return None

# --- 2. HEIC TO JPG CONVERSION ---
def convert_heic(file_path, target_path):
    heif_file = pillow_heif.read_heif(file_path)
    image = Image.frombytes(heif_file.mode, heif_file.size, heif_file.data, "raw")
    # Preserve the raw EXIF bytes for exifread to use later
    exif_data = heif_file.info.get("exif")
    image.save(target_path, "JPEG", exif=exif_data)

# --- 3. MAIN PROCESSING LOOP ---
def prepare_dataset():
    # Setup folder structure
    for split in SPLITS.keys():
        os.makedirs(os.path.join(OUTPUT_FOLDER, split), exist_ok=True)

    # Get all valid photo files
    all_files = [f for f in os.listdir(SOURCE_FOLDER) 
                 if f.lower().endswith(('.heic', '.jpg', '.jpeg'))]
    
    # Split the filenames
    train_files, temp_files = train_test_split(all_files, test_size=0.2, random_state=42)
    test_files, val_files = train_test_split(temp_files, test_size=0.5, random_state=42)
    
    file_map = {"train": train_files, "test": test_files, "val": val_files}

    for split, files in file_map.items():
        metadata = []
        split_dir = os.path.join(OUTPUT_FOLDER, split)
        
        print(f"Processing {split} split...")
        for filename in files:
            source_path = os.path.join(SOURCE_FOLDER, filename)
            target_name = filename.rsplit('.', 1)[0] + ".jpg"
            target_path = os.path.join(split_dir, target_name)
            
            # Convert or copy to JPG
            if filename.lower().endswith(".heic"):
                convert_heic(source_path, target_path)
            else:
                shutil.copy(source_path, target_path)
            
            # Extract GPS and add to metadata list
            coords = get_gps_metadata(target_path)
            if coords:
                metadata.append({
                    "file_name": target_name,
                    "latitude": coords[0],
                    "longitude": coords[1]
                })
        
        # Save metadata.csv for this split
        pd.DataFrame(metadata).to_csv(os.path.join(split_dir, "metadata.csv"), index=False)

if __name__ == "__main__":
    prepare_dataset()
    print(f"Done! Dataset is ready in the {OUTPUT_FOLDER} folder.")
