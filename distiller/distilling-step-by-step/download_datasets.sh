dataset_url="https://github.com/google-research/distilling-step-by-step/raw/main/datasets.zip"
output_file="datasets.zip"
output_dir="./"

if command -v wget &> /dev/null; then
    echo "Using wget for downloading dataset..."
    wget "$dataset_url" -O "$output_file"
elif command -v curl &> /dev/null; then
    echo "Using curl for downloading dataset..."
    curl -L "$dataset_url" -o "$output_file"
else
    echo "Error: wget atau curl not found."
    exit 1
fi


if [ -f "$output_file" ]; then
    echo "Extracting dataset..."
    unzip "$output_file" -d "$output_dir"
else
    echo "Error: File zip tidak ditemukan."
    exit 1
fi