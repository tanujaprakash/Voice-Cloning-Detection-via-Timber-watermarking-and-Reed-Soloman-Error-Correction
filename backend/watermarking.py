# watermarking.py - IMPROVED VERSION
import numpy as np
import librosa
import soundfile as sf
from scipy.fft import fft, ifft
import reedsolo

# ------------------ Configuration ------------------
# BALANCED: Good error correction with less overhead
RS_N = 15   # codeword length in bytes (reduced from 31)
RS_K = 11   # message bytes (increased from 9, was 21)
rs_codec = reedsolo.RSCodec(RS_N - RS_K)  # Can correct 2 byte errors

CHUNK_SIZE = 1024
TARGET_BINS = [200, 201, 202, 203, 204, 205, 206]
NEIGHBOR_BINS = [220, 221, 222, 223, 224, 225, 226] # Moved to ~3kHz range

# IMPROVED: Stronger amplitude for more robust embedding
AMPLITUDE_FACTOR = 6.0  # Increased to 6.0 for better robustness

SIMILARITY_THRESHOLD = 0.75  # Lowered to account for some errors
MIN_VALID_BITS = RS_K * 8

# IMPROVED: Redundancy for reliability
REDUNDANCY = 5  # Increased from 2 to 5 for better majority voting


# ------------------ Utilities ------------------
def bits_to_bytes(bits: np.ndarray) -> bytes:
    if len(bits) % 8 != 0:
        pad = 8 - (len(bits) % 8)
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    return np.packbits(bits).tobytes()


def bytes_to_bits(b: bytes, count_bits: int) -> np.ndarray:
    bits = np.unpackbits(np.frombuffer(b, dtype=np.uint8))
    return bits[:count_bits]


# ------------------ Embedding ------------------
def embed_watermark(input_wav_path: str, watermark_bits: np.ndarray, output_wav_path: str) -> bool:
    try:
        # Force 16kHz to ensure consistent frequency bins
        y, sr = librosa.load(input_wav_path, sr=16000, mono=True)
        y_out = np.copy(y)

        msg_bytes = bits_to_bytes(np.array(watermark_bits, dtype=np.uint8))
        
        # Pad to RS_K bytes to ensure fixed length block
        if len(msg_bytes) < RS_K:
            msg_bytes += b'\x00' * (RS_K - len(msg_bytes))
            
        encoded = rs_codec.encode(msg_bytes)
        if isinstance(encoded, tuple):
            encoded = encoded[0]  # Handle tuple return
        encoded_bits = np.unpackbits(np.frombuffer(encoded, dtype=np.uint8))

        total_bits = len(encoded_bits)
        max_chunks = len(y) // CHUNK_SIZE

        # IMPROVED: Adaptive redundancy based on audio length
        redundancy = REDUNDANCY
        if total_bits * redundancy > max_chunks:
            redundancy = max(1, max_chunks // total_bits)
            print(f"[embed_watermark] Adjusted redundancy to {redundancy}x (audio length constraint)")
        
        total_bits_with_redundancy = total_bits * redundancy

        if total_bits_with_redundancy > max_chunks:
            print(f"[embed_watermark] WARNING: Audio too short even with {redundancy}x redundancy")
            print(f"  Need {total_bits_with_redundancy} chunks, have {max_chunks}")
            return False

        bit_index = 0
        chunk_index = 0
        
        # Embed the watermark with adaptive redundancy
        for copy in range(redundancy):
            bit_index = 0
            for i in range(chunk_index * CHUNK_SIZE, len(y) - CHUNK_SIZE, CHUNK_SIZE):
                if bit_index >= total_bits:
                    break

                chunk = y[i:i+CHUNK_SIZE]
                fft_chunk = fft(chunk)
                mag = np.abs(fft_chunk)
                ph = np.angle(fft_chunk)

                # IMPROVED: Push-Pull Strategy
                # We calculate a baseline energy and then push Target/Neighbor apart
                # This ensures a relative difference regardless of absolute volume
                
                current_target = np.mean(mag[TARGET_BINS])
                current_neighbor = np.mean(mag[NEIGHBOR_BINS])
                baseline = (current_target + current_neighbor) / 2
                
                # Ensure baseline is not too close to zero (silence)
                baseline = max(baseline, 0.1) 
                
                # Define High and Low values based on baseline
                # Using a ratio instead of additive margin is more robust to volume changes
                val_high = baseline * (1 + AMPLITUDE_FACTOR)
                val_low = baseline * 0.1 # Suppress significantly
                
                if encoded_bits[bit_index] == 1:
                    # Bit 1: Target HIGH, Neighbor LOW
                    mag[TARGET_BINS] = val_high
                    mag[NEIGHBOR_BINS] = val_low
                    # Force phase to 0 for stability
                    ph[TARGET_BINS] = 0
                    ph[NEIGHBOR_BINS] = 0
                else:
                    # Bit 0: Target LOW, Neighbor HIGH
                    mag[TARGET_BINS] = val_low
                    mag[NEIGHBOR_BINS] = val_high
                    # Force phase to 0 for stability
                    ph[TARGET_BINS] = 0
                    ph[NEIGHBOR_BINS] = 0

                # Reconstruct chunk
                out_chunk = np.real(ifft(mag * np.exp(1j * ph)))
                
                # Check for clipping and normalize this chunk if needed
                max_amp = np.max(np.abs(out_chunk))
                if max_amp > 1.0:
                    out_chunk = out_chunk / max_amp
                    # print(f"  [embed] Prevented clipping in chunk {chunk_index}")
                
                y_out[i:i+CHUNK_SIZE] = out_chunk

                bit_index += 1
                chunk_index += 1

        # Write with explicit WAV format specification
        sf.write(output_wav_path, y_out, sr, format='WAV', subtype='PCM_16')
        print(f"[embed_watermark] Successfully embedded {total_bits} bits x {redundancy} copies")
        return True

    except Exception as e:
        print(f"[embed_watermark] error: {e}")
        import traceback
        traceback.print_exc()
        return False


# ------------------ Extraction ------------------
def extract_watermark(input_wav_path: str, expected_num_bits: int = None):
    try:
        # Force 16kHz to match embedding
        y, sr = librosa.load(input_wav_path, sr=16000, mono=True)
        max_chunks = len(y) // CHUNK_SIZE
        needed_bits = RS_N * 8

        # IMPROVED: Detect actual redundancy used (adaptive)
        redundancy = min(REDUNDANCY, max_chunks // needed_bits)
        if redundancy < 1:
            redundancy = 1
        
        print(f"[extract_watermark] Attempting to extract with {redundancy}x redundancy")

        # IMPROVED: Extract all redundant copies
        all_copies = []
        
        for copy in range(redundancy):
            observed = []
            start_chunk = copy * needed_bits
            
            for i in range(needed_bits):
                chunk_idx = start_chunk + i
                if chunk_idx >= max_chunks:
                    break
                    
                start = chunk_idx * CHUNK_SIZE
                chunk = y[start:start+CHUNK_SIZE]
                if len(chunk) < CHUNK_SIZE:
                    break

                mag = np.abs(fft(chunk))
                # Compare energy in Target vs Neighbor bins
                target_mag = np.mean(mag[TARGET_BINS])
                neighbor_mag = np.mean(mag[NEIGHBOR_BINS])
                
                if target_mag > neighbor_mag:
                    observed.append(1)
                else:
                    observed.append(0)
            
            all_copies.append(np.array(observed[:needed_bits], dtype=np.uint8))

        # IMPROVED: Majority voting across copies (or use single copy if redundancy=1)
        if len(all_copies) > 1:
            # Stack all copies and take majority vote
            stacked = np.stack(all_copies)
            observed = np.round(np.mean(stacked, axis=0)).astype(np.uint8)
            print(f"[extract_watermark] Used majority voting from {len(all_copies)} copies")
        elif len(all_copies) == 1:
            observed = all_copies[0]
            print(f"[extract_watermark] Using single copy (no redundancy)")
        else:
            print(f"[extract_watermark] ERROR: No copies extracted")
            return None

        observed_bytes = np.packbits(observed).tobytes()

        try:
            # reedsolo.decode() returns (decoded_bytes, decoded_bytes, errata_pos)
            # We only need the decoded bytes
            decode_result = rs_codec.decode(observed_bytes)
            if isinstance(decode_result, tuple):
                decoded_bytes = decode_result[0]  # Extract bytes from tuple
            else:
                decoded_bytes = decode_result
            print(f"[extract_watermark] RS decode successful!")
        except Exception as e:
            print(f"[extract_watermark] RS decode failed: {e}")
            # Try to return raw bits anyway for debugging
            print(f"[extract_watermark] RS decode failed. Returning None.")
            return None

        decoded_bits = np.unpackbits(np.frombuffer(decoded_bytes, dtype=np.uint8))

        if expected_num_bits:
            if len(decoded_bits) >= expected_num_bits:
                return decoded_bits[:expected_num_bits]
            else:
                pad = expected_num_bits - len(decoded_bits)
                return np.concatenate([decoded_bits, np.zeros(pad, dtype=np.uint8)])

        return decoded_bits

    except Exception as e:
        print(f"[extract_watermark] error: {e}")
        import traceback
        traceback.print_exc()
        return None


# ------------------ Verification ------------------
def verify_watermark(input_wav_path: str, original_bits: np.ndarray) -> dict:
    try:
        extracted = extract_watermark(input_wav_path, expected_num_bits=len(original_bits))
        if extracted is None:
            return {"result": "FAKE", "confidence": 0.0, "extracted_bits": None}

        similarity = float(np.mean(extracted == original_bits))
        confidence_pct = round(similarity * 100, 2)

        if similarity >= SIMILARITY_THRESHOLD:
            return {"result": "REAL", "confidence": confidence_pct, "extracted_bits": extracted.tolist()}
        else:
            return {"result": "FAKE", "confidence": confidence_pct, "extracted_bits": extracted.tolist()}

    except Exception as e:
        print(f"[verify_watermark] error: {e}")
        import traceback
        traceback.print_exc()
        return {"result": "FAKE", "confidence": 0.0, "extracted_bits": None}
