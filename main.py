import tkinter as tk
from tkinter import filedialog, messagebox
import subprocess
import sys
import os
from data_app.img_convert import process_image

def is_image_file(path):
    valid_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".gif")
    return path.lower().endswith(valid_extensions)

def browse_file():
    filepath = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.gif")]
    )
    if filepath:
        entry_path.delete(0, tk.END)
        entry_path.insert(0, filepath)

def show_result_window(output_text):
    result_window = tk.Toplevel(root)
    result_window.title("Hasil Proses")
    result_window.geometry("800x500")

    text_frame = tk.Frame(result_window)
    text_frame.pack(expand=True, fill=tk.BOTH)

    # Mengatur font yang lebih jelas
    font_setting = ("Consolas", 16)

    text_box = tk.Text(text_frame, wrap=tk.WORD, font=font_setting)
    text_box.insert(tk.END, output_text)
    text_box.config(state=tk.DISABLED)
    text_box.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)

    scrollbar = tk.Scrollbar(text_frame, command=text_box.yview)
    text_box.config(yscrollcommand=scrollbar.set)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

def start_generate():
    input_path = entry_path.get()
    if not input_path:
        messagebox.showwarning("Peringatan", "Silakan pilih atau masukkan path gambar terlebih dahulu.")
        return

    if not is_image_file(input_path):
        messagebox.showwarning("Peringatan", "File yang dipilih bukan gambar yang valid.")
        return

    command = [sys.executable, "data_app/img_convert.py", input_path]

    try:
        if sys.platform == "win32":
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True,
                creationflags=subprocess.CREATE_NO_WINDOW
            )
        else:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True
            )

        show_result_window(result.stdout)

    except subprocess.CalledProcessError as e:
        messagebox.showerror("Error", f"Terjadi kesalahan saat menjalankan proses:\n\n{e.stderr}")

# GUI Setup
root = tk.Tk()
root.title("Image Path Input GUI")
root.geometry("1200x400")

frame_center = tk.Frame(root)
frame_center.pack(expand=True)

frame_top = tk.Frame(frame_center)
frame_top.pack(pady=20)

entry_path = tk.Entry(frame_top, width=70)
entry_path.pack(side=tk.LEFT, padx=(0, 10))

btn_browse = tk.Button(frame_top, text="Choose File", command=browse_file)
btn_browse.pack(side=tk.LEFT)

btn_start = tk.Button(frame_center, text="Start Generate", command=start_generate, bg="green", fg="white", height=2, width=20)
btn_start.pack(pady=20)

root.mainloop()
