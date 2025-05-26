import os
import cv2
import numpy as np
import argparse
import sys
from pathlib import Path

import tensorflow as tf
from keras import backend as K
from keras.models import Model, load_model
from keras.layers import Input, Conv2D, MaxPooling2D, Reshape, Bidirectional, LSTM, Dense, Lambda, Activation, BatchNormalization, Dropout

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
        """Load the trained CRNN model"""
        try:
            # Method 1: Try loading with custom objects
            custom_objects = {
                'ctc_lambda_func': self.ctc_lambda_func,
                'ctc': lambda y_true, y_pred: y_pred
            }
            
            full_model = load_model(self.model_path, custom_objects=custom_objects)
            
            # Extract the prediction model (input to softmax layer)
            input_layer = full_model.input
            if isinstance(input_layer, list):
                input_layer = input_layer[0]  # Get the first input (image input)
            
            softmax_layer = full_model.get_layer('softmax').output
            self.model = Model(inputs=input_layer, outputs=softmax_layer)
            
            print(f"Model loaded successfully from: {self.model_path}")
            
        except Exception as e:
            print(f"Method 1 failed: {e}")
            print("Trying method 2: Building model and loading weights...")
            
            try:
                # Method 2: Build model architecture and load weights
                self.model = self.build_prediction_model()
                
                # Load the full model to extract weights
                full_model = load_model(self.model_path, custom_objects={'ctc_lambda_func': self.ctc_lambda_func, 'ctc': lambda y_true, y_pred: y_pred})
                
                # Transfer weights from loaded model to prediction model
                for layer in self.model.layers:
                    if layer.name in [l.name for l in full_model.layers]:
                        try:
                            layer.set_weights(full_model.get_layer(layer.name).get_weights())
                        except:
                            print(f"Could not transfer weights for layer: {layer.name}")
                            
                print("Model loaded successfully using method 2")
                
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
                    else:
                        print("Please save your model weights separately using model.save_weights('model_weights.h5')")
                        print("Or provide a model file that can be loaded without custom CTC functions")
                        sys.exit(1)
                        
                except Exception as e3:
                    print(f"All methods failed. Last error: {e3}")
                    sys.exit(1)
    
    def preprocess_image(self, img_path):
        """
        Preprocess the input image for prediction
        
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
            else:
                result += self.alphabets[ch]
        return result
    
    def predict_text(self, img_path, verbose=True):
        """
        Predict text from handwriting image
        
        Args:
            img_path (str): Path to the input image
            verbose (bool): Whether to print detailed information
            
        Returns:
            str: Predicted text
        """
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
                
            except:
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

def main():
    """Main function to handle command line arguments and run prediction"""
    parser = argparse.ArgumentParser(description='CRNN Handwriting Recognition Predictor')
    parser.add_argument('--model', '-m', type=str, required=True,
                       help='Path to the trained model (.h5 file)')
    parser.add_argument('--image', '-i', type=str, required=True,
                       help='Path to the input image')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Path to save the prediction result (optional)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Validate input files
    if not os.path.exists(args.model):
        print(f"Error: Model file not found at {args.model}")
        sys.exit(1)
    
    if not os.path.exists(args.image):
        print(f"Error: Image file not found at {args.image}")
        sys.exit(1)
    
    # Initialize predictor
    predictor = HandwritingPredictor(args.model)
    
    # Make prediction
    result = predictor.predict_text(args.image, verbose=args.verbose)
    
    if result is not None:
        print(f"\n{'='*50}")
        print(f"PREDICTION RESULT: '{result}'")
        print(f"{'='*50}")

        folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '../image_to_text'))
        file_name = f'{result}.txt'
        file_path = os.path.join(folder, file_name)

        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)  # ini lebih aman daripada os.mkdir()
            
        with open(file_path, 'w') as file:
            file.write(f'{result}\n')
        
        # Save result to file if output path is provided
        if args.output:
            try:
                with open(args.output, 'w') as f:
                    f.write(result)
                print(f"Result saved to: {args.output}")
            except Exception as e:
                print(f"Error saving result: {e}")
    else:
        print("Prediction failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()

# Example usage:
# python predict_handwriting.py --model /path/to/model.h5 --image /path/to/image.jpg
# python predict_handwriting.py -m model.h5 -i test_image.png -o result.txt --verbose