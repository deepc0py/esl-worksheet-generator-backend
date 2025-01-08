import asyncio
import websockets
from websockets.server import WebSocketServerProtocol
import wave
import numpy as np
from faster_whisper import WhisperModel
from dataclasses import dataclass
from typing import Optional, Dict, List
import logging
import time
import threading
from queue import Queue, Empty
import json
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ClientState:
    """Represents the state of a connected client"""
    websocket: WebSocketServerProtocol
    audio_queue: Queue
    start_time: float
    recording_data: List[bytes]
    is_active: bool
    source_lang: str = "es"   # Set Spanish as source language

class TranscriptionServer:
    def __init__(
        self,
        host: str,
        port: int,
        lang: str = "es",
        model: str = "large-v3",
        use_vad: bool = False,
        save_output_recording: bool = True,
        output_recording_filename: str = "./recordings/recording.wav",
        max_clients: int = 8,
        max_connection_time: int = 600,
        run_audio_server: bool = True,
        audio_server_port: int = 8765,
        device: str = "cpu",
        compute_type: str = "float32"
    ):
        """Initialize the transcription server with the given parameters."""
        self.host = host
        self.port = port
        self.lang = lang
        self.use_vad = use_vad
        self.save_output_recording = save_output_recording
        self.output_recording_filename = output_recording_filename
        self.max_clients = max_clients
        self.max_connection_time = max_connection_time
        self.run_audio_server = run_audio_server
        self.audio_server_port = audio_server_port
        
        # Create recordings directory if it doesn't exist
        os.makedirs(os.path.dirname(output_recording_filename), exist_ok=True)
        
        # Initialize the Whisper model
        logger.info(f"Loading Whisper model {model}...")
        self.model = WhisperModel(
            model,
            device=device,
            compute_type=compute_type
        )
        
        # Client management
        self.clients: Dict[str, ClientState] = {}
        self.lock = threading.Lock()

    def _preprocess_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """Preprocess audio data with high-pass filter and compression."""
        try:
            # Ensure we're working with float32 (not double/float64)
            audio_data = audio_data.astype(np.float32)
            
            # Apply a high-pass filter to remove low frequency noise
            try:
                from scipy import signal
                nyq = 0.5 * 16000
                cutoff = 50 / nyq
                b, a = signal.butter(5, cutoff, btype='high', analog=False)
                audio_data = signal.filtfilt(b, a, audio_data).astype(np.float32)  # Ensure float32
            except ImportError:
                logger.warning("scipy not available, skipping high-pass filter")
            
            # Normalize before compression
            max_abs = np.max(np.abs(audio_data))
            if max_abs > 1e-10:  # Avoid division by zero
                audio_data = audio_data / max_abs
            
            # Apply compression
            threshold = 0.5
            ratio = 2.0
            mask_above = np.abs(audio_data) > threshold
            mask_positive = audio_data > 0
            
            compressed = np.copy(audio_data)
            compressed[mask_above & mask_positive] = threshold + (audio_data[mask_above & mask_positive] - threshold) / ratio
            compressed[mask_above & ~mask_positive] = -(threshold + (np.abs(audio_data[mask_above & ~mask_positive]) - threshold) / ratio)
            
            # Final normalization to ensure [-1, 1] range
            compressed = np.clip(compressed, -1.0, 1.0)
            
            return compressed.astype(np.float32)  # Ensure float32 output
            
        except Exception as e:
            logger.error(f"Error in audio preprocessing: {str(e)}")
            return audio_data.astype(np.float32)  # Return original data as float32 if processing fails

    async def process_audio_stream(self, client_state: ClientState):
        """Process incoming audio stream for a client."""
        accumulated_audio = np.array([], dtype=np.float32)
        last_processed_length = 0
        
        while client_state.is_active:
            try:
                # Get audio chunk from queue with timeout
                try:
                    audio_chunk = await asyncio.get_event_loop().run_in_executor(
                        None, 
                        lambda: client_state.audio_queue.get(timeout=1.0)
                    )
                except Empty:
                    # If we have accumulated audio, process it even if no new audio is coming
                    if len(accumulated_audio) > 0:
                        logger.info("Processing remaining audio...")
                        try:
                            segments, info = await asyncio.get_event_loop().run_in_executor(
                                None,
                                lambda: self.model.transcribe(
                                    accumulated_audio,
                                    language="es",
                                    vad_filter=True,
                                    vad_parameters=dict(
                                        min_silence_duration_ms=500,
                                        speech_pad_ms=100,
                                    )
                                )
                            )
                            
                            segments_list = list(segments)
                            if segments_list:
                                for segment in segments_list:
                                    response = {
                                        "type": "transcription",
                                        "start": segment.start,
                                        "end": segment.end,
                                        "text": segment.text
                                    }
                                    logger.info(f"Final transcription: {segment.text}")
                                    try:
                                        await client_state.websocket.send(json.dumps(response))
                                    except websockets.exceptions.ConnectionClosed:
                                        logger.info("Client disconnected during transcription send")
                                        client_state.is_active = False
                                        break
                            
                            # Clear the accumulated audio after processing
                            accumulated_audio = np.array([], dtype=np.float32)
                            last_processed_length = 0
                        except Exception as e:
                            logger.error(f"Error processing final audio: {str(e)}", exc_info=True)
                    continue
                
                # Convert bytes to numpy array and preprocess
                audio_data = np.frombuffer(audio_chunk, dtype=np.float32)
                
                # Skip processing if the audio chunk is empty
                if len(audio_data) == 0:
                    continue
                
                # Save raw audio chunk before preprocessing
                if self.save_output_recording:
                    client_state.recording_data.append(audio_chunk)
                
                audio_data = self._preprocess_audio(audio_data)
                accumulated_audio = np.concatenate([accumulated_audio, audio_data])
                
                # Process when we have enough audio (2 seconds)
                if len(accumulated_audio) >= 32000:
                    logger.info("Starting transcription of audio segment...")
                    try:
                        segments, info = await asyncio.get_event_loop().run_in_executor(
                            None,
                            lambda: self.model.transcribe(
                                accumulated_audio,
                                language="es",
                                vad_filter=True,
                                vad_parameters=dict(
                                    min_silence_duration_ms=500,
                                    speech_pad_ms=100,
                                )
                            )
                        )
                        
                        segments_list = list(segments)
                        if segments_list:
                            for segment in segments_list:
                                response = {
                                    "type": "transcription",
                                    "start": segment.start,
                                    "end": segment.end,
                                    "text": segment.text
                                }
                                logger.info(f"Transcription: {segment.text}")
                                try:
                                    await client_state.websocket.send(json.dumps(response))
                                except websockets.exceptions.ConnectionClosed:
                                    logger.info("Client disconnected during transcription send")
                                    client_state.is_active = False
                                    break
                        
                        # Keep a small overlap for context
                        accumulated_audio = accumulated_audio[-8000:]  # Keep last 0.5 seconds
                        last_processed_length = 0
                        
                    except Exception as e:
                        logger.error(f"Error during transcription: {str(e)}", exc_info=True)
                        continue
                    
            except Exception as e:
                logger.error(f"Error processing audio stream: {str(e)}", exc_info=True)
                continue

        # Final cleanup and process any remaining audio
        if len(accumulated_audio) > 0:
            logger.info("Processing final audio buffer before cleanup...")
            try:
                segments, info = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.model.transcribe(
                        accumulated_audio,
                        language="es",
                        vad_filter=True,
                        vad_parameters=dict(
                            min_silence_duration_ms=500,
                            speech_pad_ms=100,
                        )
                    )
                )
                
                segments_list = list(segments)
                if segments_list:
                    for segment in segments_list:
                        response = {
                            "type": "transcription",
                            "start": segment.start,
                            "end": segment.end,
                            "text": segment.text
                        }
                        logger.info(f"Final cleanup transcription: {segment.text}")
                        try:
                            await client_state.websocket.send(json.dumps(response))
                        except websockets.exceptions.ConnectionClosed:
                            logger.info("Client disconnected during final transcription send")
                            break
            except Exception as e:
                logger.error(f"Error processing final cleanup audio: {str(e)}", exc_info=True)

        if self.save_output_recording and client_state.recording_data:
            logger.info("Saving final recording...")

    async def handle_client(self, websocket: WebSocketServerProtocol, path: str):
        """Handle individual client connections."""
        if len(self.clients) >= self.max_clients:
            await websocket.send(json.dumps({
                "type": "error",
                "message": "Server at maximum capacity"
            }))
            return

        client_id = str(id(websocket))
        client_state = ClientState(
            websocket=websocket,
            audio_queue=Queue(),
            start_time=time.time(),
            recording_data=[],
            is_active=True
        )
        
        with self.lock:
            self.clients[client_id] = client_state
        
        try:
            # Start audio processing task
            process_task = asyncio.create_task(
                self.process_audio_stream(client_state)
            )
            
            async for message in websocket:
                current_time = time.time()
                if current_time - client_state.start_time > self.max_connection_time:
                    await websocket.send(json.dumps({
                        "type": "error",
                        "message": "Maximum connection time exceeded"
                    }))
                    break
                
                if isinstance(message, bytes):
                    client_state.audio_queue.put(message)
                else:
                    try:
                        data = json.loads(message)
                        if data.get("type") == "close":
                            break
                        elif data.get("type") == "config":
                            if "source_lang" in data:
                                client_state.source_lang = data["source_lang"]
                            await websocket.send(json.dumps({
                                "type": "config_ack",
                                "source_lang": client_state.source_lang
                            }))
                    except json.JSONDecodeError:
                        logger.error("Invalid JSON message received")
                        continue
            
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client {client_id} connection closed")
        finally:
            # Cleanup
            client_state.is_active = False
            
            # Wait for any remaining audio to be processed
            while not client_state.audio_queue.empty():
                await asyncio.sleep(0.1)
            
            # Wait for final processing
            await asyncio.sleep(10)  # Give time for final processing
            
            # Save the recording if enabled
            if self.save_output_recording and client_state.recording_data:
                try:
                    self._save_recording(client_state.recording_data, client_id)
                    logger.info(f"Successfully saved recording for client {client_id}")
                except Exception as e:
                    logger.error(f"Failed to save recording: {str(e)}", exc_info=True)
            
            with self.lock:
                del self.clients[client_id]
            
            await process_task

    def _save_recording(self, recording_data: List[bytes], client_id: str):
        """Save the recorded audio to a WAV file."""
        filename = self.output_recording_filename.replace(
            ".wav", f"_{client_id}.wav"
        )
        
        try:
            # Convert the bytes data to float32 numpy array
            audio_data = np.concatenate([
                np.frombuffer(chunk, dtype=np.float32) for chunk in recording_data
            ])
            
            # Ensure the data is in float32 range [-1, 1]
            audio_data = np.clip(audio_data, -1.0, 1.0)
            
            # Convert to 16-bit PCM with proper scaling
            audio_data = (audio_data * 32767.0).clip(-32768, 32767).astype(np.int16)
            
            with wave.open(filename, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(16000)  # 16kHz
                wav_file.writeframes(audio_data.tobytes())
            
            logger.info(f"Saved recording to {filename}")
            
        except Exception as e:
            logger.error(f"Error saving recording: {str(e)}", exc_info=True)

    async def start(self):
        """Start the WebSocket server."""
        if self.run_audio_server:
            async def wrapped_handler(websocket):
                await self.handle_client(websocket, "/")
            
            server = await websockets.serve(
                wrapped_handler,
                self.host,
                self.audio_server_port,
                ping_interval=30,
                ping_timeout=10
            )
            
            logger.info(
                f"WebSocket server started on "
                f"ws://{self.host}:{self.audio_server_port}"
            )
            
            await server.wait_closed()

if __name__ == "__main__":
    # Create and run the server
    server = TranscriptionServer(
        host="localhost",
        port=8765,
        model="large-v3",
        device="cpu",
        compute_type="float32"
    )
    
    # Run the server in the event loop
    asyncio.run(server.start())
