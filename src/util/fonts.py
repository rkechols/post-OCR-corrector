import os


ALL_FONTS = list()
FONTS_FILE = os.path.join("data", "fonts.txt")
with open(FONTS_FILE, "r", encoding="utf-8") as file:
    for line in file:
        font_path = line.split(":")[0].strip()
        ALL_FONTS.append(font_path)
