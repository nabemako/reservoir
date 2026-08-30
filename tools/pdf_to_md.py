#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract text and images from the target PDF and produce a verbatim Markdown file.

Outputs are written to ./extracted_pdf/
- paper_full.md      : full verbatim transcription (very long)
- images/...png      : extracted images from the PDF

This script is intended to be run in a GitHub Actions runner and commit outputs
back to the repository.
"""
import fitz  # PyMuPDF
import os
import sys
import re

PDF_FILENAME = "Reservoir_Computing_Model_for_Liquid_Crystal_Cell_Responsive_to_Design_Parameter_and_Its_Augmentation_With_Novel_Transfer_Learning_Framework.pdf"
OUT_DIR = "extracted_pdf"
IMAGES_DIR = os.path.join(OUT_DIR, "images")
MD_PATH = os.path.join(OUT_DIR, "paper_full.md")

os.makedirs(IMAGES_DIR, exist_ok=True)

print("Opening PDF:", PDF_FILENAME)
try:
    doc = fitz.open(PDF_FILENAME)
except Exception as e:
    print("Failed to open PDF:", e)
    sys.exit(2)

meta = doc.metadata

with open(MD_PATH, "w", encoding="utf-8") as md:
    md.write(f"# Full transcription: {meta.get('title','(no title)')}\n\n")
    if meta.get('author'):
        md.write(f"**Authors:** {meta.get('author')}\n\n")
    # write all metadata
    md.write("## PDF metadata\n\n")
    md.write('```
')
    for k, v in meta.items():
        md.write(f"{k}: {v}\n")
    md.write('```\n\n')

    page_count = doc.page_count
    md.write(f"## Page count: {page_count}\n\n")

    img_index = 0
    for pno in range(page_count):
        page = doc.load_page(pno)
        md.write(f"---\n\n")
        md.write(f"## Page {pno+1}\n\n")
        # Extract and write page text verbatim
        text = page.get_text("text")
        # Normalize CRLF
        text = text.replace('\r\n', '\n')
        md.write("```")
        md.write(text)
        if not text.endswith('\n'):
            md.write('\n')
        md.write("```\n\n")

        # Extract images embedded in page
        image_list = page.get_images(full=True)
        if image_list:
            md.write(f"### Images on page {pno+1}\n\n")
        for img in image_list:
            xref = img[0]
            base_name = f"page_{pno+1}_img_{img_index}.png"
            imgpath = os.path.join(IMAGES_DIR, base_name)
            try:
                pix = fitz.Pixmap(doc, xref)
                # Handle CMYK or alpha
                if pix.n - pix.alpha < 4:
                    pix.save(imgpath)
                else:
                    # convert to RGB
                    pix1 = fitz.Pixmap(fitz.csRGB, pix)
                    pix1.save(imgpath)
                    pix1 = None
                pix = None
                md.write(f"![{base_name}]({os.path.join('images', base_name)})\n\n")
                print('Saved image', imgpath)
            except Exception as e:
                print('Failed to save image', xref, e)
            img_index += 1

print('Wrote markdown to', MD_PATH)
print('Done')
