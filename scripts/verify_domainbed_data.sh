#!/bin/bash
# scripts/verify_domainbed_data.sh
# Verifies presence and structure of DomainBed datasets.

DATA_ROOT="data/domainbed"

passed=true

check_dataset() {
    NAME=$1
    shift
    DOMAINS=("$@")
    
    echo "Checking $NAME..."
    if [ ! -d "$DATA_ROOT/$NAME" ]; then
        echo "  [FAIL] Directory $DATA_ROOT/$NAME missing."
        passed=false
        return
    fi
    
    for dom in "${DOMAINS[@]}"; do
        if [ ! -d "$DATA_ROOT/$NAME/$dom" ]; then
             echo "  [FAIL] Domain $dom missing in $NAME."
             passed=false
        else
            count=$(find "$DATA_ROOT/$NAME/$dom" -type f | wc -l)
            echo "  [OK] $NAME/$dom found ($count files)."
        fi
    done
}

# PACS
check_dataset "PACS" "art_painting" "cartoon" "photo" "sketch"

# VLCS
check_dataset "VLCS" "CALTECH" "LABELME" "PASCAL" "SUN"

# TerraIncognita
check_dataset "TerraIncognita" "location_38" "location_43" "location_46" "location_100"

# OfficeHome
check_dataset "OfficeHome" "Art" "Clipart" "Product" "RealWorld"

if [ "$passed" = true ]; then
    echo "SUCCESS: All required datasets found."
    exit 0
else
    echo "FAILURE: Missing datasets or domains."
    exit 1
fi
