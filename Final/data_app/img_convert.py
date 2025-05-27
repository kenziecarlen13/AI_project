import sys
import os
from PIL import Image, ImageEnhance
import subprocess

def process_image(input_path):
    try:
        # Path referensi reference_img, sesuaikan sesuai foldermu
        reference_path = "data_app/reference/reference.jpg"
        reference_img = Image.open(reference_path)
        new_img = Image.open(input_path)

        # Cek apakah input image lebih besar dari reference image
        reference_width = reference_img.width
        reference_height = reference_img.height
        input_width = new_img.width
        input_height = new_img.height

        # Jika input image lebih besar dari reference, resize dengan mempertahankan aspect ratio
        if input_width > reference_width or input_height > reference_height:
            # Hitung ratio untuk resize sambil mempertahankan aspect ratio
            width_ratio = reference_width / input_width
            height_ratio = reference_height / input_height
            
            # Pilih ratio yang lebih kecil agar gambar tidak melebihi ukuran reference
            resize_ratio = min(width_ratio, height_ratio)
            
            # Hitung ukuran baru
            new_width = int(input_width * resize_ratio)
            new_height = int(input_height * resize_ratio)
            
            # Resize dengan mempertahankan aspect ratio
            new_resized = new_img.resize((new_width, new_height), resample=Image.LANCZOS)
            
            # Jika ukuran hasil resize tidak sama persis dengan reference, buat canvas baru
            if new_width != reference_width or new_height != reference_height:
                # Buat canvas putih dengan ukuran reference
                canvas = Image.new('RGB', (reference_width, reference_height), 'white')
                
                # Hitung posisi untuk center gambar
                x_offset = (reference_width - new_width) // 2
                y_offset = (reference_height - new_height) // 2
                
                # Paste gambar ke center canvas
                canvas.paste(new_resized, (x_offset, y_offset))
                new_resized = canvas
        else:
            # Jika ukuran input <= reference, langsung proses tanpa resize
            new_resized = new_img

        # Proses enhancement
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