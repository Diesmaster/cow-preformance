#!/bin/bash

# Create the data folder if it doesn't exist
mkdir -p data

# Unzip data.zip into the data folder
unzip -q data.zip -d data

# Check if the unzip was successful
if [ $? -eq 0 ]; then
    echo "Successfully extracted data.zip to data folder"
else
    echo "Error: Failed to extract data.zip"
    exit 1
fi
