import zipfile
import re
import os

src_path = "outputs/Ke_hoach_do_an_10_tuan.xlsx"
fixed_path = "outputs/Ke_hoach_do_an_10_tuan_fixed.xlsx"

print(f"Reading from {src_path}...")
with zipfile.ZipFile(src_path, "r") as zin:
    with zipfile.ZipFile(fixed_path, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            content = zin.read(item.filename)
            if item.filename in ["xl/worksheets/sheet2.xml", "xl/worksheets/sheet3.xml"]:
                text = content.decode("utf-8")
                # Thay the pane: bo xSplit (duong doc), chi giu lai ySplit=6 (co dinh dong tieu de) de khong con duong doc chan man hinh
                # hoac co the unfreeze hoan toan neu muon
                # O day thay bang ySplit="6" topLeftCell="A7" activePane="bottomLeft" state="frozen"
                new_text = re.sub(
                    r'<x:pane\s+[^>]*/>',
                    '<x:pane ySplit="6" topLeftCell="A7" activePane="bottomLeft" state="frozen" />',
                    text
                )
                content = new_text.encode("utf-8")
                print(f"Removed vertical freeze from {item.filename}")
            zout.writestr(item, content)

print(f"Successfully generated: {fixed_path}")

# Thu ghi de vao file goc neu file goc khong bi lock
try:
    with open(src_path, "r+b") as f:
        pass
    import shutil
    shutil.copyfile(fixed_path, src_path)
    os.remove(fixed_path)
    print(f"Overwritten directly into {src_path}!")
except PermissionError:
    print(f"Note: {src_path} is currently open in Excel. Kept {fixed_path}.")
