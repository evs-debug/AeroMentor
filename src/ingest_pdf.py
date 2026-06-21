from pypdf import PdfReader

reader = PdfReader(
    "pdfs/phak.pdf"
)

text = ""

for page in reader.pages:

    page_text = page.extract_text()

    if page_text:
        text += page_text + "\n"

with open(
    "docs/phak.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write(text)

print(
    f"Extracted {len(reader.pages)} pages."
)
print(
    f"Characters extracted: {len(text)}"
)