import os

from pypdf import PdfReader

for filename in os.listdir("pdfs"):

    if not filename.endswith(".pdf"):
        continue

    path = os.path.join(
        "pdfs",
        filename
    )

    reader = PdfReader(path)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    output_name = (
        filename.replace(
            ".pdf",
            ".txt"
        )
    )

    output_path = os.path.join(
        "data",
        output_name
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(text)

    print(
        f"Converted {filename}"
    )
    print(
        f"Pages: {len(reader.pages)}"
    )
    print(
        f"Characters: {len(text)}\n"
    )