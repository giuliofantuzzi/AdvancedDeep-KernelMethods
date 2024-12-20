#!/bin/bash
echo "...Cleaning log files..."

# Remove all log files in current directory and all subdirectories
find . -name "*.aux" -type f -delete
find . -name "*.fdb_latexmk" -type f -delete
find . -name "*.fls" -type f -delete
find . -name "*.log" -type f -delete
find . -name "*.out" -type f -delete
find . -name "*.synctex.gz" -type f -delete
find . -name "*.toc" -type f -delete

echo "Log files cleaned successfully!"
