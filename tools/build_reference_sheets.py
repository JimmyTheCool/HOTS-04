#!/usr/bin/env python3
from pathlib import Path
from collections import defaultdict
import csv
import json
import re
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.pagesizes import A4, landscape
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

root = Path('.')
image_paths = sorted(
    p for p in root.iterdir()
    if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'}
)
groups = defaultdict(list)
pattern = re.compile(r'^(.*?)(\d+)$')
for path in image_paths:
    match = pattern.match(path.stem.strip())
    hero = match.group(1).strip(' _-') if match else path.stem.strip()
    order = int(match.group(2)) if match else 0
    groups[hero].append((order, path))
for hero in groups:
    groups[hero].sort(key=lambda item: (item[0], item[1].name.casefold()))

out = Path('reference_analysis')
sheets = out / 'contact_sheets'
sheets.mkdir(parents=True, exist_ok=True)
inventory = {
    'hero_count': len(groups),
    'image_count': len(image_paths),
    'heroes': [
        {'hero': hero, 'images': [p.name for _, p in items]}
        for hero, items in sorted(groups.items(), key=lambda kv: kv[0].casefold())
    ],
}
(out / 'inventory.json').write_text(json.dumps(inventory, indent=2), encoding='utf-8')
with (out / 'inventory.csv').open('w', newline='', encoding='utf-8-sig') as handle:
    writer = csv.writer(handle)
    writer.writerow(['Hero', 'Image Count', 'Images'])
    for hero, items in sorted(groups.items(), key=lambda kv: kv[0].casefold()):
        writer.writerow([hero, len(items), ' | '.join(p.name for _, p in items)])

page_w, page_h = landscape(A4)
pdf_path = out / 'HOTS04_Reference_Sheets.pdf'
c = canvas.Canvas(str(pdf_path), pagesize=(page_w, page_h))
margin = 28
title_h = 44
gap = 12
cell_w = (page_w - 2 * margin - gap) / 2
cell_h = (page_h - 2 * margin - title_h - gap) / 2
font = ImageFont.load_default()

for hero, items in sorted(groups.items(), key=lambda kv: kv[0].casefold()):
    c.setFont('Helvetica-Bold', 22)
    c.drawString(margin, page_h - margin - 22, hero)
    c.setFont('Helvetica', 9)
    c.drawRightString(page_w - margin, page_h - margin - 18, f'{len(items)} reference views')

    sheet = Image.new('RGB', (1600, 1100), 'white')
    draw = ImageDraw.Draw(sheet)
    draw.text((24, 15), hero, fill='black', font=font)
    thumb_w, thumb_h = 760, 490

    for index, (_, path) in enumerate(items[:4]):
        row, col = divmod(index, 2)
        x0 = 20 + col * 790
        y0 = 55 + row * 515
        image = Image.open(path).convert('RGB')
        image.thumbnail((thumb_w, thumb_h - 28), Image.Resampling.LANCZOS)
        frame = Image.new('RGB', (thumb_w, thumb_h), '#eeeeee')
        fx = (thumb_w - image.width) // 2
        fy = 22 + (thumb_h - 22 - image.height) // 2
        frame.paste(image, (fx, fy))
        fd = ImageDraw.Draw(frame)
        fd.text((8, 5), path.name, fill='black', font=font)
        sheet.paste(frame, (x0, y0))

        pdf_x = margin + col * (cell_w + gap)
        pdf_y = page_h - margin - title_h - (row + 1) * cell_h - row * gap
        iw, ih = Image.open(path).size
        scale = min(cell_w / iw, (cell_h - 14) / ih)
        dw, dh = iw * scale, ih * scale
        c.drawImage(
            ImageReader(str(path)),
            pdf_x + (cell_w - dw) / 2,
            pdf_y + (cell_h - 14 - dh) / 2,
            width=dw,
            height=dh,
            preserveAspectRatio=True,
            mask='auto',
        )
        c.setFont('Helvetica', 8)
        c.drawCentredString(pdf_x + cell_w / 2, pdf_y + 2, path.name)

    safe = re.sub(r'[^A-Za-z0-9._-]+', '_', hero).strip('_') or 'hero'
    sheet.save(sheets / f'{safe}.jpg', quality=92, optimize=True)
    c.showPage()

c.save()
print(json.dumps(inventory, indent=2))
