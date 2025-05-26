import sys
import os
from PIL import Image, ImageEnhance
import subprocess

def process_image(input_path):
    try:
        # Path referensi melody_img, sesuaikan sesuai foldermu
        melody_path = "data_app/reference/reference.jpg"
        melody_img = Image.open(melody_path)
        new_img = Image.open(input_path)

        melody_height = melody_img.height
        new_aspect_ratio = new_img.width / round(new_img.height/2)
        new_width = int(melody_height * new_aspect_ratio)

        new_resized = new_img.resize((new_width, melody_height), resample=Image.LANCZOS)
        new_bright = ImageEnhance.Brightness(new_resized).enhance(1.7)
        new_contrasted = ImageEnhance.Contrast(new_bright).enhance(1.7)
        new_sharp = ImageEnhance.Sharpness(new_contrasted).enhance(1.6)

        # Folder hasil output
        output_dir = "data_app/result"
        os.makedirs(output_dir, exist_ok=True)  # buat folder jika belum ada

        output_path = os.path.join(output_dir, "result.jpg")
        new_sharp.save(output_path)

        return output_path
    except Exception as e:
        print(f"Error saat proses gambar: {e}", file=sys.stderr)
        return None

def main():
    if len(sys.argv) < 2:
        print("Usage: python img_convert.py <path_to_image>", file=sys.stderr)
        sys.exit(1)

    input_path = sys.argv[1]

    processed_path = process_image(input_path)
    if not processed_path:
        sys.exit(1)

    command = ['python', 'data_app/predict.py', '--model', 'data_app/Final/model.h5', '--image', processed_path]

    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error saat menjalankan predict.py:\n{e.stderr}", file=sys.stderr)
        sys.exit(1)

    # Hapus file sementara **tidak dilakukan lagi** karena ingin simpan hasil di folder result
    # Jadi, bagian hapus file dihilangkan

if __name__ == "__main__":
    main()
