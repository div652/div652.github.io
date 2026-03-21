import os
import glob
import re
from datetime import datetime
import shutil

# Configuration
EXPORT_DIR = "../slite_exports"
POSTS_DIR = "../_posts"

def slugify(text):
    text = re.sub(r'[^\w\s-]', '', text).strip().lower()
    return re.sub(r'[-\s]+', '-', text)

def main():
    if not os.path.exists(EXPORT_DIR):
        print(f"Error: {EXPORT_DIR} directory not found.")
        print("Please export your Slite Markdown files into the 'slite_exports' folder.")
        # create the folder for convenience
        os.makedirs(EXPORT_DIR, exist_ok=True)
        return

    os.makedirs(POSTS_DIR, exist_ok=True)
    md_files = glob.glob(os.path.join(EXPORT_DIR, "*.md"))
    
    if not md_files:
        print(f"No markdown files found in {EXPORT_DIR}.")
        print("Please place your Slite exports there and re-run.")
        return

    for file_path in md_files:
        filename = os.path.basename(file_path)
        base_name, _ = os.path.splitext(filename)
        
        # Determine Title
        title = base_name.replace("-", " ").title()
        
        # Read file content
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        # Get creation time for date
        stat = os.stat(file_path)
        # fallback to modification time if ctime isn't meaningful on linux
        date_time = datetime.fromtimestamp(stat.st_mtime)
        date_str = date_time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Generate new filename
        slugified_title = slugify(base_name)
        new_filename = f"{date_time.strftime('%Y-%m-%d')}-{slugified_title}.md"
        dest_path = os.path.join(POSTS_DIR, new_filename)
        
        # Prepare Jekyll Frontmatter
        frontmatter = f"""---
layout: post
title: "{title}"
date: {date_str}
description: Migrated from Slite
tags: [architecture, machine-learning, notes]
categories: [slite-export]
---
"""
        # Prefix the frontmatter to the content
        final_content = frontmatter + "\n" + content
        
        # Write to _posts
        with open(dest_path, "w", encoding="utf-8") as f:
            f.write(final_content)
            
        print(f"Migrated: {filename} -> {new_filename}")

if __name__ == "__main__":
    main()
