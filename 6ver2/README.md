# 6ver2 - Object Removal (No Deep Learning)

Project nay chi dung OpenCV inpainting de xoa doi tuong va phuc hoi nen anh.
Khong can model `.pth`, khong train, RAM nhe hon ban deep learning.

## Cai dat (Ubuntu)

```bash
cd 6ver2
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Chay app

```bash
python3 app.py
```

Mac dinh app chay o:
- http://127.0.0.1:7861

## Cach dung

1. Upload anh.
2. To mau do len vung can xoa.
3. Chon thuat toan:
   - `TELEA`: nhanh, ket qua tu nhien trong nhieu truong hop.
   - `NS`: on dinh voi mot so ket cau phuc tap.
4. Bam `Run`.
5. Neu can, tang `Inpaint Radius` de bo vung xoa rong hon.
