import os
import cv2
import numpy as np
import sys
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageEnhance, ImageTk
import threading
import time
import json
from datetime import datetime

import tensorflow as tf
from keras import backend as K
from keras.models import Model, load_model
from keras.layers import Input, Conv2D, MaxPooling2D, Reshape, Bidirectional, LSTM, Dense, Lambda, Activation, BatchNormalization, Dropout

def get_base_path():
    """Get the base path for the application (works for both script and exe)"""
    if getattr(sys, 'frozen', False):
        # Running as compiled executable
        base_path = os.path.dirname(sys.executable)
    else:
        # Running as script
        base_path = os.path.dirname(os.path.abspath(__file__))
    return base_path

def get_resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = get_base_path()
    
    return os.path.join(base_path, relative_path)

class HandwritingPredictor:
    def __init__(self, model_path):
        """
        Initialize the handwriting predictor
        
        Args:
            model_path (str): Path to the saved model (.h5 file)
        """
        self.model_path = model_path
        self.alphabets = u"ABCDEFGHIJKLMNOPQRSTUVWXYZ-' "
        self.num_of_characters = len(self.alphabets) + 1  # +1 for CTC blank
        self.model = None
        self.is_model_loaded = False
        
        # Load the model
        self.load_model()
    
    def build_prediction_model(self):
        """
        Build the prediction model architecture (same as training but without CTC loss)
        """
        input_data = Input(shape=(256, 64, 1), name='input')

        inner = Conv2D(32, (3, 3), padding='same', name='conv1', kernel_initializer='he_normal')(input_data)  
        inner = BatchNormalization()(inner)
        inner = Activation('relu')(inner)
        inner = MaxPooling2D(pool_size=(2, 2), name='max1')(inner)

        inner = Conv2D(64, (3, 3), padding='same', name='conv2', kernel_initializer='he_normal')(inner)
        inner = BatchNormalization()(inner)
        inner = Activation('relu')(inner)
        inner = MaxPooling2D(pool_size=(2, 2), name='max2')(inner)
        inner = Dropout(0.3)(inner)

        inner = Conv2D(128, (3, 3), padding='same', name='conv3', kernel_initializer='he_normal')(inner)
        inner = BatchNormalization()(inner)
        inner = Activation('relu')(inner)
        inner = MaxPooling2D(pool_size=(1, 2), name='max3')(inner)
        inner = Dropout(0.3)(inner)

        # CNN to RNN
        inner = Reshape(target_shape=((64, 1024)), name='reshape')(inner)
        inner = Dense(64, activation='relu', kernel_initializer='he_normal', name='dense1')(inner)

        ## RNN
        inner = Bidirectional(LSTM(256, return_sequences=True), name = 'lstm1')(inner)
        inner = Bidirectional(LSTM(256, return_sequences=True), name = 'lstm2')(inner)

        ## OUTPUT
        inner = Dense(self.num_of_characters, kernel_initializer='he_normal', name='dense2')(inner)
        y_pred = Activation('softmax', name='softmax')(inner)

        model = Model(inputs=input_data, outputs=y_pred)
        return model
    
    def ctc_lambda_func(self, args):
        """CTC loss function"""
        y_pred, labels, input_length, label_length = args
        y_pred = y_pred[:, 2:, :]
        return K.ctc_batch_cost(labels, y_pred, input_length, label_length)
        
    def load_model(self):
        """Load the trained CRNN model with improved error handling"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
        try:
            # Method 1: Try loading with custom objects
            custom_objects = {
                'ctc_lambda_func': self.ctc_lambda_func,
                'ctc': lambda y_true, y_pred: y_pred
            }
            
            full_model = load_model(self.model_path, custom_objects=custom_objects, compile=False)
            
            # Extract the prediction model (input to softmax layer)
            input_layer = full_model.input
            if isinstance(input_layer, list):
                input_layer = input_layer[0]  # Get the first input (image input)
            
            softmax_layer = full_model.get_layer('softmax').output
            self.model = Model(inputs=input_layer, outputs=softmax_layer)
            
            print(f"Model loaded successfully from: {self.model_path}")
            self.is_model_loaded = True
            
        except Exception as e:
            print(f"Method 1 failed: {e}")
            print("Trying method 2: Building model and loading weights...")
            
            try:
                # Method 2: Build model architecture and load weights
                self.model = self.build_prediction_model()
                
                # Load the full model to extract weights
                full_model = load_model(self.model_path, custom_objects={
                    'ctc_lambda_func': self.ctc_lambda_func, 
                    'ctc': lambda y_true, y_pred: y_pred
                }, compile=False)
                
                # Transfer weights from loaded model to prediction model
                for layer in self.model.layers:
                    if layer.name in [l.name for l in full_model.layers]:
                        try:
                            layer.set_weights(full_model.get_layer(layer.name).get_weights())
                        except Exception as weight_error:
                            print(f"Could not transfer weights for layer {layer.name}: {weight_error}")
                            
                print("Model loaded successfully using method 2")
                self.is_model_loaded = True
                
            except Exception as e2:
                print(f"Method 2 also failed: {e2}")
                print("Trying method 3: Build model and load weights directly...")
                
                try:
                    # Method 3: Load weights file directly if available
                    self.model = self.build_prediction_model()
                    weights_path = self.model_path.replace('.h5', '_weights.h5')
                    
                    if os.path.exists(weights_path):
                        self.model.load_weights(weights_path)
                        print("Model loaded using separate weights file")
                        self.is_model_loaded = True
                    else:
                        raise Exception("Could not load model with any method")
                        
                except Exception as e3:
                    print(f"All methods failed. Last error: {e3}")
                    self.is_model_loaded = False
                    raise e3
    
    def preprocess_image(self, img_path):
        """
        Preprocess the input image for prediction with improved error handling
        
        Args:
            img_path (str): Path to the input image
            
        Returns:
            np.array: Preprocessed image ready for prediction
        """
        try:
            # Read image as grayscale
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if image is None:
                raise ValueError(f"Could not read image from: {img_path}")
            
            # Apply the same preprocessing as in training
            (h, w) = image.shape
            
            # Create blank white image
            final_img = np.ones([64, 256]) * 255
            
            # Crop if larger than expected dimensions
            if w > 256:
                image = image[:, :256]
                
            if h > 64:
                image = image[:64, :]
            
            # Place image on blank canvas
            final_img[:h, :w] = image
            
            # Rotate clockwise
            final_img = cv2.rotate(final_img, cv2.ROTATE_90_CLOCKWISE)
            
            # Normalize to [0, 1]
            final_img = final_img / 255.0
            
            # Reshape for model input
            final_img = final_img.reshape(1, 256, 64, 1)
            
            return final_img
            
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            return None
    
    def num_to_label(self, num_array):
        """
        Convert numerical prediction back to text
        
        Args:
            num_array: Array of predicted character indices
            
        Returns:
            str: Predicted text
        """
        result = ""
        for ch in num_array:
            if ch == -1:  # CTC Blank
                break
            elif ch < len(self.alphabets):
                result += self.alphabets[ch]
        return result
    
    def predict_text(self, img_path, verbose=True):
        """
        Predict text from handwriting image with improved error handling
        
        Args:
            img_path (str): Path to the input image
            verbose (bool): Whether to print detailed information
            
        Returns:
            str: Predicted text
        """
        if not self.is_model_loaded:
            print("Model is not loaded properly")
            return None
            
        if verbose:
            print(f"Processing image: {img_path}")
        
        # Preprocess image
        processed_img = self.preprocess_image(img_path)
        if processed_img is None:
            return None
        
        try:
            # Make prediction
            pred = self.model.predict(processed_img, verbose=0)
            
            # Decode CTC output - Updated for newer TensorFlow versions
            input_length = np.ones(pred.shape[0]) * pred.shape[1]
            
            # Use tf.keras.backend.ctc_decode instead of K.ctc_decode
            try:
                # Method 1: TensorFlow 2.x approach
                decoded, _ = tf.keras.backend.ctc_decode(pred, 
                                                       input_length=input_length, 
                                                       greedy=True)
                # Convert tensor to numpy
                decoded_dense = tf.sparse.to_dense(decoded[0], default_value=-1).numpy()
                predicted_text = self.num_to_label(decoded_dense[0])
                
            except Exception as decode_error:
                # Method 2: Alternative approach using tf.nn.ctc_greedy_decoder
                try:
                    # Transpose pred to shape [time_steps, batch_size, num_classes]
                    pred_transposed = tf.transpose(pred, [1, 0, 2])
                    input_length_tensor = tf.cast(input_length, tf.int32)
                    
                    decoded, _ = tf.nn.ctc_greedy_decoder(pred_transposed, input_length_tensor)
                    # Convert sparse tensor to dense
                    decoded_dense = tf.sparse.to_dense(decoded[0], default_value=-1).numpy()
                    predicted_text = self.num_to_label(decoded_dense[0])
                    
                except Exception as e2:
                    # Method 3: Manual greedy decoding
                    if verbose:
                        print(f"Using manual decoding due to: {e2}")
                    predicted_text = self.manual_ctc_decode(pred[0])
            
            if verbose:
                print(f"Predicted text: '{predicted_text}'")
            
            return predicted_text
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            return None
    
    def manual_ctc_decode(self, prediction):
        """
        Manual CTC greedy decoding as fallback
        
        Args:
            prediction: Model prediction output [time_steps, num_classes]
            
        Returns:
            str: Decoded text
        """
        # Get the most likely character at each time step
        input_len = prediction.shape[0]
        results = []
        
        for i in range(input_len):
            # Get the character with highest probability
            char_idx = np.argmax(prediction[i])
            results.append(char_idx)
        
        # Remove consecutive duplicates and blanks
        final_result = []
        previous = -1
        
        for char_idx in results:
            # Skip blanks (last character in alphabet) and consecutive duplicates
            if char_idx != len(self.alphabets) and char_idx != previous:
                final_result.append(char_idx)
            previous = char_idx
        
        # Convert to text
        decoded_text = ""
        for char_idx in final_result:
            if char_idx < len(self.alphabets):
                decoded_text += self.alphabets[char_idx]
        
        return decoded_text


class ImageProcessor:
    def __init__(self):
        base_path = get_base_path()
        self.reference_path = os.path.join(base_path, "data_app", "reference", "reference.jpg")
        self.output_dir = os.path.join(base_path, "data_app", "result")
        
    def process_image(self, input_path):
        """
        Process the input image with improved contrast and sizing logic
        
        Args:
            input_path (str): Path to input image
            
        Returns:
            str: Path to processed image or None if failed
        """
        try:
            # Check if reference image exists
            if not os.path.exists(self.reference_path):
                print(f"Warning: Reference image not found at {self.reference_path}")
                # Use default processing without reference
                return self._process_without_reference(input_path)
            
            reference_img = Image.open(self.reference_path)
            new_img = Image.open(input_path)

            # Get dimensions
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

            # Enhanced processing untuk kontras area gelap yang lebih baik
            # 1. Tingkatkan kontras terlebih dahulu untuk memisahkan area gelap dan terang
            new_contrasted = ImageEnhance.Contrast(new_resized).enhance(1.5)
            
            # 2. Sedikit tingkatkan brightness untuk area gelap tidak terlalu gelap
            new_bright = ImageEnhance.Brightness(new_contrasted).enhance(1.8)
            
            # 3. Tingkatkan kontras lagi setelah brightness adjustment
            new_contrasted2 = ImageEnhance.Contrast(new_bright).enhance(1.3)
            
            # 4. Tingkatkan sharpness untuk detail yang lebih tajam
            new_sharp = ImageEnhance.Sharpness(new_contrasted2).enhance(1.4)

            # Create output directory
            os.makedirs(self.output_dir, exist_ok=True)

            output_path = os.path.join(self.output_dir, "result.jpg")
            new_sharp.save(output_path)

            return output_path
            
        except Exception as e:
            print(f"Error processing image: {e}")
            return None
    
    def _process_without_reference(self, input_path):
        """
        Process image without reference (fallback method)
        """
        try:
            new_img = Image.open(input_path)
            
            # Enhanced processing untuk kontras area gelap yang lebih baik
            # 1. Tingkatkan kontras terlebih dahulu untuk memisahkan area gelap dan terang
            new_contrasted = ImageEnhance.Contrast(new_img).enhance(1.5)
            
            # 2. Sedikit tingkatkan brightness untuk area gelap tidak terlalu gelap
            new_bright = ImageEnhance.Brightness(new_contrasted).enhance(1.8)
            
            # 3. Tingkatkan kontras lagi setelah brightness adjustment
            new_contrasted2 = ImageEnhance.Contrast(new_bright).enhance(1.3)
            
            # 4. Tingkatkan sharpness untuk detail yang lebih tajam
            new_sharp = ImageEnhance.Sharpness(new_contrasted2).enhance(1.4)

            # Create output directory
            os.makedirs(self.output_dir, exist_ok=True)

            output_path = os.path.join(self.output_dir, "result.jpg")
            new_sharp.save(output_path)

            return output_path
            
        except Exception as e:
            print(f"Error in fallback processing: {e}")
            return None


class HandwritingRecognitionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Handwriting Recognition Application")
        self.root.geometry("1200x800")
        
        # Initialize components
        self.predictor = None
        self.processor = ImageProcessor()
        
        # Use relative path for model
        base_path = get_base_path()
        self.model_path = os.path.join(base_path, "data_app", "Final", "model.h5")
        
        # Setup GUI
        self.setup_gui()
        
        # Try to load model on startup
        self.load_model_async()
    
    def setup_gui(self):
        """Setup the GUI components"""
        # Main frame
        main_frame = tk.Frame(self.root)
        main_frame.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)
        
        # Title
        title_label = tk.Label(main_frame, text="Handwriting Recognition", 
                              font=("Arial", 18, "bold"))
        title_label.pack(pady=(0, 20))
        
        # File selection frame
        file_frame = tk.Frame(main_frame)
        file_frame.pack(fill=tk.X, pady=(0, 20))
        
        tk.Label(file_frame, text="Select Image:", font=("Arial", 12)).pack(anchor=tk.W)
        
        path_frame = tk.Frame(file_frame)
        path_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.entry_path = tk.Entry(path_frame, width=70, font=("Arial", 10))
        self.entry_path.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 10))
        
        self.btn_browse = tk.Button(path_frame, text="Browse", command=self.browse_file,
                                   bg="#4CAF50", fg="white", font=("Arial", 10))
        self.btn_browse.pack(side=tk.RIGHT)
        
        # Process button frame
        button_frame = tk.Frame(main_frame)
        button_frame.pack(pady=20)
        
        self.btn_process = tk.Button(button_frame, text="Process Image", 
                                    command=self.start_processing,
                                    bg="#2196F3", fg="white", height=2, width=20,
                                    font=("Arial", 12, "bold"))
        self.btn_process.pack()
        
        # Progress bar
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.pack(fill=tk.X, pady=(10, 20))
        
        # Status label
        self.status_label = tk.Label(main_frame, text="Ready", 
                                    font=("Arial", 10), fg="green")
        self.status_label.pack()
        
        # Results frame
        result_frame = tk.LabelFrame(main_frame, text="Results", 
                                    font=("Arial", 12, "bold"))
        result_frame.pack(expand=True, fill=tk.BOTH, pady=(20, 0))
        
        # Image display frame
        image_frame = tk.Frame(result_frame)
        image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 5), pady=10)
        
        tk.Label(image_frame, text="Processed Image:", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        
        self.image_label = tk.Label(image_frame, text="No image processed", 
                                   bg="white", relief=tk.SUNKEN)
        self.image_label.pack(expand=True, fill=tk.BOTH, pady=(5, 0))
        
        # Text result frame
        text_frame = tk.Frame(result_frame)
        text_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 10), pady=10)
        
        tk.Label(text_frame, text="Recognized Text:", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        
        text_container = tk.Frame(text_frame)
        text_container.pack(expand=True, fill=tk.BOTH, pady=(5, 0))
        
        self.text_result = tk.Text(text_container, wrap=tk.WORD, font=("Consolas", 14),
                                  state=tk.DISABLED)
        self.text_result.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
        
        scrollbar = tk.Scrollbar(text_container, command=self.text_result.yview)
        self.text_result.config(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def load_model_async(self):
        """Load model in background thread"""
        def load_model():
            try:
                self.update_status("Loading model...", "orange")
                if os.path.exists(self.model_path):
                    self.predictor = HandwritingPredictor(self.model_path)
                    if self.predictor.is_model_loaded:
                        self.update_status("Model loaded successfully", "green")
                    else:
                        self.update_status("Failed to load model properly", "red")
                else:
                    self.update_status(f"Model file not found: {self.model_path}", "red")
            except Exception as e:
                self.update_status(f"Error loading model: {str(e)}", "red")
        
        thread = threading.Thread(target=load_model, daemon=True)
        thread.start()
    
    def update_status(self, message, color="black"):
        """Update status label safely from any thread"""
        def update():
            self.status_label.config(text=message, fg=color)
        
        self.root.after(0, update)
    
    def browse_file(self):
        """Browse for image file"""
        filepath = filedialog.askopenfilename(
            title="Select Handwriting Image",
            filetypes=[
                ("Image Files", "*.png *.jpg *.jpeg *.bmp *.gif"),
                ("All Files", "*.*")
            ]
        )
        if filepath:
            self.entry_path.delete(0, tk.END)
            self.entry_path.insert(0, filepath)
    
    def is_image_file(self, path):
        """Check if file is a valid image"""
        valid_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".gif")
        return path.lower().endswith(valid_extensions)
    
    def start_processing(self):
        """Start image processing in background thread"""
        input_path = self.entry_path.get().strip()
        
        if not input_path:
            messagebox.showwarning("Warning", "Please select an image file first.")
            return
        
        if not os.path.exists(input_path):
            messagebox.showerror("Error", "Selected file does not exist.")
            return
        
        if not self.is_image_file(input_path):
            messagebox.showwarning("Warning", "Please select a valid image file.")
            return
        
        if self.predictor is None or not self.predictor.is_model_loaded:
            messagebox.showerror("Error", "Model is not loaded properly. Please check the model path and try restarting the application.")
            return
        
        # Disable button and start progress
        self.btn_process.config(state=tk.DISABLED)
        self.progress.start()
        
        # Start processing in background thread
        thread = threading.Thread(target=self.process_image, args=(input_path,), daemon=True)
        thread.start()
    
    def process_image(self, input_path):
        """Process image and predict text"""
        try:
            self.update_status("Processing image...", "orange")
            
            # Process image
            processed_path = self.processor.process_image(input_path)
            if not processed_path:
                raise Exception("Failed to process image")
            
            self.update_status("Predicting text...", "orange")
            
            # Predict text
            predicted_text = self.predictor.predict_text(processed_path, verbose=False)
            if predicted_text is None:
                raise Exception("Failed to predict text")
            
            # Save result to file
            self.save_result_to_file(predicted_text)
            
            # Update GUI with results
            self.root.after(0, self.update_results, processed_path, predicted_text)
            
        except Exception as e:
            self.root.after(0, self.handle_error, str(e))
    
    def save_result_to_file(self, result):
        """Save prediction result to text file"""
        try:
            # Create folder relative to executable location
            base_path = get_base_path()
            folder = os.path.join(base_path, "image_to_text")
            os.makedirs(folder, exist_ok=True)
            
            # Create safe filename
            safe_result = "".join(c for c in result if c.isalnum() or c in (' ', '-', '_')).rstrip()
            if not safe_result:
                safe_result = f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            file_name = f'{safe_result}.txt'
            file_path = os.path.join(folder, file_name)
            
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(f'{result}\n')
            
            print(f"Result saved to: {file_path}")
            
        except Exception as e:
            print(f"Error saving result: {e}")
    
    def update_results(self, processed_path, predicted_text):
        """Update GUI with processing results"""
        try:
            # Display processed image
            img = Image.open(processed_path)
            img.thumbnail((400, 300), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            
            self.image_label.config(image=photo, text="")
            self.image_label.image = photo  # Keep a reference
            
            # Display predicted text
            self.text_result.config(state=tk.NORMAL)
            self.text_result.delete(1.0, tk.END)
            self.text_result.insert(tk.END, predicted_text)
            self.text_result.config(state=tk.DISABLED)
            
            self.update_status("Processing completed successfully!", "green")
            
        except Exception as e:
            self.handle_error(f"Error updating results: {str(e)}")
        finally:
            self.progress.stop()
            self.btn_process.config(state=tk.NORMAL)
    
    def handle_error(self, error_message):
        """Handle errors during processing"""
        self.update_status(f"Error: {error_message}", "red")
        self.progress.stop()
        self.btn_process.config(state=tk.NORMAL)
        messagebox.showerror("Processing Error", error_message)


def main():
    """Main function to run the application"""
    root = tk.Tk()
    app = HandwritingRecognitionGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()