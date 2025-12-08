from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import numpy as np
import os
import pickle
import base64
from feature_extraction import extract_features
from watermarking import embed_watermark, extract_watermark, verify_watermark

app = Flask(__name__, static_folder='../frontend', static_url_path='/')
CORS(app)  # Enable CORS for all routes

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../models/model.pkl')

# Helper functions for text/binary conversion with length prefixing
def text_to_bits(text):
    """Convert text string to binary bits array with length prefix"""
    text_bytes = text.encode('utf-8')
    text_length = len(text_bytes)
    
    # Create length prefix (2 bytes = up to 65535 bytes of text)
    length_bytes = text_length.to_bytes(2, byteorder='big')
    
    # Combine length prefix with actual text
    full_bytes = length_bytes + text_bytes
    bits = np.unpackbits(np.frombuffer(full_bytes, dtype=np.uint8))
    
    print(f"[text_to_bits] Text: '{text}' -> {text_length} bytes -> {len(bits)} bits (with 16-bit length prefix)")
    return bits

def bits_to_text(bits):
    """Convert binary bits array back to text using length prefix"""
    # Ensure we have at least 16 bits for the length prefix
    if len(bits) < 16:
        print(f"[bits_to_text] ERROR: Not enough bits ({len(bits)}) for length prefix")
        return ""
    
    # Pad to multiple of 8
    if len(bits) % 8 != 0:
        bits = np.concatenate([bits, np.zeros(8 - len(bits) % 8, dtype=np.uint8)])
    
    # Convert all bits to bytes
    all_bytes = np.packbits(bits).tobytes()
    
    # Extract length prefix (first 2 bytes)
    text_length = int.from_bytes(all_bytes[:2], byteorder='big')
    
    # Validate length
    if text_length > len(all_bytes) - 2 or text_length < 0:
        print(f"[bits_to_text] WARNING: Invalid length prefix {text_length}, using fallback")
        # Fallback: try to decode without length prefix
        try:
            return all_bytes[2:].decode('utf-8', errors='ignore').rstrip('\x00')
        except:
            return ""
    
    # Extract actual text bytes using the length
    text_bytes = all_bytes[2:2+text_length]
    
    try:
        decoded_text = text_bytes.decode('utf-8')
        print(f"[bits_to_text] Decoded {text_length} bytes -> '{decoded_text}'")
        return decoded_text
    except UnicodeDecodeError as e:
        print(f"[bits_to_text] UTF-8 decode error: {e}")
        return text_bytes.decode('utf-8', errors='replace')

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/predict', methods=['POST'])
def predict_route():
    # Accept either 'file' or 'audio' as the key for the uploaded audio
    if 'file' in request.files:
        audio_file = request.files['file']
    elif 'audio' in request.files:
        audio_file = request.files['audio']
    else:
        return jsonify({'error': 'audio file required'}), 400

    input_path = os.path.join(UPLOAD_FOLDER, 'predict.wav')
    audio_file.save(input_path)

    if not os.path.exists(MODEL_PATH):
        return jsonify({'error': 'Model file not found'}), 500

    try:
        with open(MODEL_PATH, 'rb') as f:
            model_data = pickle.load(f)
        # Handle both dict format {'model': ..., 'scaler': ...} and direct model
        if isinstance(model_data, dict):
            model = model_data.get('model')
            scaler = model_data.get('scaler')
        else:
            model = model_data
            scaler = None
        
        features = extract_features(input_path).reshape(1, -1)
        if scaler is not None:
            features = scaler.transform(features)
        
        prediction = model.predict(features)[0]
        confidence = (
            np.max(model.predict_proba(features)) * 100
            if hasattr(model, 'predict_proba') else 0.0
        )
        # Extract accuracy if available
        model_accuracy = model_data.get('accuracy', 0.0) if isinstance(model_data, dict) else 0.0
        
        result_str = 'REAL' if prediction == 1 else 'FAKE'
        return jsonify({
            'result': result_str, 
            'confidence': round(confidence, 2),
            'model_accuracy': round(model_accuracy * 100, 2)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/embed', methods=['POST'])
def embed_route():
    try:
        # Accept either 'audio' or 'audio_file' for the uploaded audio
        if 'audio' in request.files:
            audio_file = request.files['audio']
        elif 'audio_file' in request.files:
            audio_file = request.files['audio_file']
        else:
            return jsonify({'error': 'audio file required'}), 400

        # Accept watermark text from either 'bits' or 'watermark_data'
        watermark_text = request.form.get('bits') or request.form.get('watermark_data')
        if not watermark_text:
            return jsonify({'error': 'watermark text required'}), 400

        # Convert text to binary bits
        bits = text_to_bits(watermark_text)
        print(f"[embed_route] Watermark text: '{watermark_text}' -> {len(bits)} bits")

        input_path = os.path.join(UPLOAD_FOLDER, 'input.wav')
        output_path = os.path.join(UPLOAD_FOLDER, 'output_watermarked.wav')
        audio_file.save(input_path)

        ok = embed_watermark(input_path, bits, output_path)
        if not ok:
            return jsonify({'success': False, 'error': 'Embedding failed'}), 500

        # Return base64 of watermarked audio for download/display
        try:
            with open(output_path, 'rb') as f:
                encoded = base64.b64encode(f.read()).decode('utf-8')
        except Exception as e:
            print(f"[embed_route] Base64 encoding error: {e}")
            encoded = None
        
        # Generate a proper filename - sanitize the watermark text
        safe_text = watermark_text[:20].replace(' ', '_').replace('/', '_').replace('\\', '_')
        safe_text = ''.join(c for c in safe_text if c.isalnum() or c == '_')
        filename = f"watermarked_{safe_text}.wav"
        
        return jsonify({
            'success': True,
            'message': f'Watermark "{watermark_text}" embedded successfully',
            'output_path': output_path,
            'watermark_text': watermark_text,
            'bits_embedded': len(bits),
            'embedded_audio_base64': encoded,
            'filename': filename
        })
    except Exception as e:
        import traceback
        print(f"[embed_route] ERROR: {type(e).__name__}: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/extract', methods=['POST'])
def extract_route():
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'audio file required'}), 400
        audio_file = request.files['audio']
        input_path = os.path.join(UPLOAD_FOLDER, 'extract.wav')
        audio_file.save(input_path)
        
        extracted_bits = extract_watermark(input_path)
        if extracted_bits is None:
            return jsonify({'success': False, 'watermark_text': None})
        
        # Convert bits back to text
        watermark_text = bits_to_text(extracted_bits)
        return jsonify({
            'success': True,
            'watermark_text': watermark_text,
            'bits_count': len(extracted_bits)
        })
    except Exception as e:
        import traceback
        print(f"[extract_route] ERROR: {type(e).__name__}: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/verify', methods=['POST'])
def verify_route():
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'audio file required'}), 400
        audio_file = request.files['audio']
        input_path = os.path.join(UPLOAD_FOLDER, 'verify.wav')
        audio_file.save(input_path)
        
        watermark_text = request.form.get('bits') or request.form.get('watermark_data')
        if not watermark_text:
            # No original watermark provided; just extract and return text
            extracted_bits = extract_watermark(input_path)
            if extracted_bits is None:
                return jsonify({
                    'result': 'FAKE',
                    'confidence': 0.0,
                    'extracted_bits': None,
                    'error': 'Failed to extract watermark - audio may not be watermarked'
                })
            
            extracted_text = bits_to_text(extracted_bits)
            return jsonify({
                'result': 'REAL',
                'confidence': 100.0,
                'extracted_watermark': extracted_text,
                'extracted_bits': extracted_bits.tolist(),
                'message': f'Watermark detected: "{extracted_text}"'
            })
        
        # Convert original text to bits for verification
        original_bits = text_to_bits(watermark_text)
        result = verify_watermark(input_path, original_bits)
        
        # Add the original text and extracted text to the result
        result['original_watermark'] = watermark_text
        if result.get('extracted_bits'):
            extracted_text = bits_to_text(np.array(result['extracted_bits'], dtype=np.uint8))
            result['extracted_watermark'] = extracted_text
            result['message'] = f'Original: "{watermark_text}" | Extracted: "{extracted_text}"'
        
        return jsonify(result)
    except Exception as e:
        import traceback
        print(f"[verify_route] ERROR: {type(e).__name__}: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e), 'result': 'FAKE', 'confidence': 0.0}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
