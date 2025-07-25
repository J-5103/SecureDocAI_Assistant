import os
import shutil
import uuid

def save_uploaded_file(uploaded_file, subfolder=""):
    ext = os.path.splitext(uploaded_file.filename)[-1]
    filename = f"{uuid.uuid4().hex}{ext}"
    upload_dir = os.path.join("uploads", subfolder)
    os.makedirs(upload_dir, exist_ok=True)

    file_path = os.path.join(upload_dir, filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(uploaded_file.file, buffer)

    return file_path
