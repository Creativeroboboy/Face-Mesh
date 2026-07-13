from PIL import Image
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

import matplotlib.pyplot as plt
import matplotlib.patches as patches

img = Image.open("face.jpg")

data = pytesseract.image_to_data(
    img,
    lang="ara",   # 👈 Arabic ON
    output_type=pytesseract.Output.DICT
)

fig, ax = plt.subplots(1)
ax.imshow(img)

for i in range(len(data['text'])):
    text = data['text'][i].strip()

    if text:
        x, y, w, h = (
            data['left'][i],
            data['top'][i],
            data['width'][i],
            data['height'][i]
        )

        rect = patches.Rectangle(
            (x, y),
            w,
            h,
            linewidth=1,
            edgecolor='red',
            facecolor='none'
        )

        ax.add_patch(rect)
        ax.text(x, y - 5, text, fontsize=8)

plt.show()