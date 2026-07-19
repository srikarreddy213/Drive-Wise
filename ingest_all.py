"""
ingest_all.py — Batch ingestion script for DriveWise.
Scans the 'new-brochuse' directory and processes all PDFs into the FAISS index.
"""

import os
import re
import shutil
from pathlib import Path
from urllib.parse import unquote

# Add workspace path to sys.path to allow imports from src
import sys
sys.path.append(str(Path(__file__).parent))

from src.config import BROCHURES_DIR
from src.ingestion import ingest_brochure
from src.vector_store import (
    load_vectorstore,
    build_vectorstore,
    add_documents_to_store,
    save_vectorstore,
    get_available_cars,
)

SRC_DIR = Path(__file__).parent / "new-brochuse"

def parse_filename(filename: str) -> tuple[str, str, str]:
    # Decode URL-encoded names
    name = unquote(filename)
    # Remove extension
    name, _ = os.path.splitext(name)
    
    # Pre-clean known weird characters
    name = name.replace('\xad', '-').replace('\u2013', '-')
    
    # 1. Determine brand
    brand = "Unknown"
    name_clean_lower = re.sub(r'[-_]', ' ', name.lower())
    
    if "mercedes" in name_clean_lower or re.search(r'\bmb\b', name_clean_lower) or "cle" in name_clean_lower or "class" in name_clean_lower:
        brand = "Mercedes-Benz"
    elif "bmw" in name_clean_lower:
        brand = "Bmw"
    elif "nissan" in name_clean_lower or "patrol" in name_clean_lower:
        brand = "Nissan"
    elif "nexon" in name_clean_lower or "safari" in name_clean_lower or "harrier" in name_clean_lower or "sierra" in name_clean_lower or "tata" in name_clean_lower:
        brand = "Tata"
    elif "creta" in name_clean_lower or "ioniq" in name_clean_lower:
        brand = "Hyundai"
    elif "multistrada" in name_clean_lower or "panigale" in name_clean_lower or "streetfighter" in name_clean_lower or "supersport" in name_clean_lower:
        brand = "Ducati"
    elif "suzuki" in name_clean_lower:
        brand = "Suzuki"
    elif "vw" in name_clean_lower or "volkswagen" in name_clean_lower:
        brand = "Volkswagen"
    elif "ford" in name_clean_lower:
        brand = "Ford"
    elif "lamborghini" in name_clean_lower:
        brand = "Lamborghini"
    elif "kia" in name_clean_lower:
        brand = "Kia"
    elif "mahindra" in name_clean_lower:
        brand = "Mahindra"

    # 3. Clean up model name
    # Replace separators with space
    clean_name = re.sub(r'[-_]', ' ', name)
    
    # Extract version (year)
    version = "2026"  # Default fallback version
    # Match 4 digit year (19xx or 20xx)
    year_match = re.search(r'\b(19\d{2}|20\d{2})\b', clean_name)
    if year_match:
        version = year_match.group(1)
        clean_name = re.sub(rf'\b{version}\b', '', clean_name, flags=re.IGNORECASE)
    else:
        # Match 2 digit year suffix after dash/underscore at the end (e.g. _25 or -25)
        short_year_match = re.search(r'\b(2[0-6]|1[0-9])\b', clean_name)
        if short_year_match:
            version = "20" + short_year_match.group(1)
            clean_name = re.sub(rf'\b{short_year_match.group(1)}\b', '', clean_name, flags=re.IGNORECASE)

    # Remove brand prefixes/tokens
    brand_tokens = ["mercedes benz", "mercedes-benz", "mercedes", "bmw", "tata", "hyundai", "ducati", "nissan", "suzuki", "volkswagen", "vw", "ford", "lamborghini", "kia", "mahindra", "mb"]
    
    # Remove brand names from the model string
    for bt in brand_tokens:
        clean_name = re.sub(rf'\b{bt}\b', '', clean_name, flags=re.IGNORECASE)
        
    # Remove common filler words
    filler_words = [
        "brochure", "digital", "desktop", "celebration", "catalogue", "flyer", "web",
        "all new", "new", "in", "id", "int", "uk", "usa", "fr", "sg", "mx", "ger", "tw", "nz", "au",
        "eng", "tmga", "nov", "june", "may", "march", "february", "my", "asset", "v2", "series"
    ]
    for fw in filler_words:
        clean_name = re.sub(rf'\b{fw}\b', '', clean_name, flags=re.IGNORECASE)

    # Clean double spaces and punctuation
    clean_name = re.sub(r'\s+', ' ', clean_name)
    clean_name = re.sub(r'\(\d+\)', '', clean_name) # Remove (1)
    clean_name = clean_name.strip()
    
    # Normalise model
    model = clean_name.title()
    
    # Edge case adjustments
    if brand == "Mercedes-Benz":
        model_upper_tokens = [t.upper() for t in model.split()]
        if "EQS" in model_upper_tokens:
            if "SUV" in model_upper_tokens:
                model = "EQS SUV"
            else:
                model = "EQS"
        elif "CLA" in model_upper_tokens:
            model = "CLA"
        elif "GLA" in model_upper_tokens:
            model = "GLA"
        elif "GLC" in model_upper_tokens:
            if "COUPE" in model_upper_tokens:
                model = "GLC Coupe"
            else:
                model = "GLC"
        elif "GLE" in model_upper_tokens:
            if "COUPE" in model_upper_tokens:
                model = "GLE Coupe"
            else:
                model = "GLE"
        elif "GLS" in model_upper_tokens:
            model = "GLS"
        elif "C" in model_upper_tokens:
            model = "C-Class"
        elif "E" in model_upper_tokens:
            model = "E-Class"
        elif "S" in model_upper_tokens:
            model = "S-Class"
        elif "V" in model_upper_tokens:
            model = "V-Class"
        elif "CLE" in model_upper_tokens:
            model = "CLE Cabriolet"
    elif brand == "Bmw":
        if "6" in name_clean_lower:
            model = "6 Series"
        elif "bikes" in name_clean_lower:
            model = "Bikes"
    elif brand == "Tata":
        if "nexon icng" in name_clean_lower:
            model = "Nexon iCNG"
        elif "nexon adas" in name_clean_lower:
            model = "Nexon ADAS"
        elif "nexon dark" in name_clean_lower:
            model = "Nexon Dark"
        elif "nexon" in name_clean_lower:
            model = "Nexon"
        elif "safari red dark" in name_clean_lower:
            model = "Safari Red Dark"
        elif "safari" in name_clean_lower:
            model = "Safari"
        elif "harrier red dark" in name_clean_lower:
            model = "Harrier Red Dark"
        elif "harrier" in name_clean_lower:
            model = "Harrier"
        elif "sierra" in name_clean_lower:
            model = "Sierra"
    elif brand == "Hyundai":
        if "creta ev" in name_clean_lower:
            model = "Creta EV"
        elif "creta n line" in name_clean_lower:
            model = "Creta N Line"
        elif "creta" in name_clean_lower:
            model = "Creta"
        elif "ioniq 5" in name_clean_lower:
            model = "Ioniq 5"
    elif brand == "Nissan":
        if "patrol super safari" in name_clean_lower:
            model = "Patrol Super Safari"
        elif "patrol" in name_clean_lower:
            model = "Patrol"
    elif brand == "Volkswagen":
        if "id polo" in name_clean_lower or "id. polo" in name_clean_lower:
            model = "ID. Polo"
            
    # Remove any empty model or default
    if not model or model == ".":
        model = "General"

    # Title-case normalizing (except for all caps model names)
    final_model = []
    for token in model.split():
        if token.upper() in ["EQS", "SUV", "CLA", "GLA", "GLC", "GLE", "GLS", "AMG", "EV", "ADAS", "CNG", "ICNG", "BMW", "VW", "GTi", "GT"]:
            final_model.append(token.upper() if token.upper() != "ICNG" else "iCNG")
        else:
            final_model.append(token)
    model = " ".join(final_model)

    return brand, model, version


def main():
    if sys.platform.startswith('win'):
        import io
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
        
    print("🚀 Starting DriveWise Batch Ingestion Pipeline ...")
    
    if not SRC_DIR.exists():
        print(f"❌ Source directory {SRC_DIR} not found.")
        sys.exit(1)
        
    pdf_files = [f for f in os.listdir(SRC_DIR) if f.endswith('.pdf')]
    total_files = len(pdf_files)
    print(f"🔍 Found {total_files} PDF files to process.")
    
    if total_files == 0:
        print("⚠️ No brochures to ingest.")
        return
        
    # Load or initialize vector store
    print("📥 Loading FAISS vector store ...")
    store = load_vectorstore()
    
    processed_count = 0
    success_count = 0
    
    for idx, f in enumerate(pdf_files, start=1):
        pdf_path = SRC_DIR / f
        brand, model, version = parse_filename(f)
        
        print(f"\n[{idx}/{total_files}] Processing: {f}")
        print(f"   -> Mapped to: Brand={brand} | Model={model} | Version={version}")
        
        try:
            # 1. Ingest PDF (clean, chunk, classify)
            docs = ingest_brochure(str(pdf_path), brand, model, version)
            
            if not docs:
                print(f"   ⚠️ No text chunks extracted. Skipping.")
                continue
                
            # 2. Add documents to store
            if store is None:
                store = build_vectorstore(docs)
            else:
                store = add_documents_to_store(docs, store)
                
            # 3. Save a clean copy of the brochure
            dest_filename = f"{brand.lower().strip().replace(' ', '_')}_{model.lower().strip().replace(' ', '_')}_{version}.pdf"
            dest_path = BROCHURES_DIR / dest_filename
            shutil.copy2(pdf_path, dest_path)
            
            success_count += 1
            processed_count += 1
            
            # Periodic save every 5 files to avoid losing progress
            if processed_count % 5 == 0:
                print(f"💾 [Auto-Save] Saving index to disk (processed {processed_count} files) ...")
                save_vectorstore(store)
                
        except Exception as e:
            print(f"   ❌ Error processing {f}: {e}")
            
    # Final save of the FAISS index
    if store is not None:
        print("\n💾 Saving final FAISS index ...")
        save_vectorstore(store)
        
        print("\n🎉 Ingestion Complete!")
        print(f"   - Total files processed: {total_files}")
        print(f"   - Successfully indexed: {success_count}")
        print(f"   - Total vectors in store: {store.index.ntotal}")
        
        available_cars = get_available_cars(store)
        print("\n🚘 Current Indexed Vehicles in Store:")
        for b, models in sorted(available_cars.items()):
            print(f"   - {b}: {', '.join(sorted(models))}")
    else:
        print("\n❌ Ingestion finished but no vector store was built.")

if __name__ == "__main__":
    main()
