import asyncio
import websockets
from websockets.server import WebSocketServerProtocol
import wave
import numpy as np
from faster_whisper import WhisperModel
from dataclasses import dataclass
from typing import Optional, Dict, List, Any
import logging
import time
import threading
from queue import Queue, Empty
import json
import os
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
class AudioConstants:
    SAMPLE_RATE = 16000
    CHUNK_SIZE = 80000  # 5 seconds of audio at 16kHz
    OVERLAP_SIZE = 16000  # 1 second of audio for context
    BIT_DEPTH = 16
    CHANNELS = 1
    MAX_AMPLITUDE = 32767.0

class ProcessingConstants:
    HIGH_PASS_CUTOFF = 50
    COMPRESSION_THRESHOLD = 0.5
    COMPRESSION_RATIO = 2.0
    NOISE_FLOOR = 1e-10

class VADConstants:
    MIN_SILENCE_DURATION_MS = 500
    SPEECH_PAD_MS = 100

@dataclass
class ServerConfig:
    """Server configuration parameters"""
    host: str
    port: int
    max_clients: int = 8
    max_connection_time: int = 600
    ping_interval: int = 30
    ping_timeout: int = 10

@dataclass
class ModelConfig:
    """Whisper model configuration"""
    model_name: str = "large-v3"
    device: str = "cpu"
    compute_type: str = "float32"
    source_lang: str = "es"
    use_vad: bool = True

@dataclass
class RecordingConfig:
    """Recording configuration"""
    save_output: bool = True
    output_dir: str = "./recordings"
    filename_prefix: str = "recording"

class MessageType(Enum):
    """WebSocket message types"""
    TRANSCRIPTION = "transcription"
    ERROR = "error"
    CONFIG = "config"
    CONFIG_ACK = "config_ack"
    CLOSE = "close"

class TranscriptionError(Exception):
    """Base exception for transcription errors"""
    pass

class AudioProcessingError(TranscriptionError):
    """Raised when audio processing fails"""
    pass

class ModelError(TranscriptionError):
    """Raised when model inference fails"""
    pass

@dataclass
class ClientState:
    """Represents the state of a connected client"""
    websocket: WebSocketServerProtocol
    audio_queue: Queue
    start_time: float
    recording_data: List[bytes]
    is_active: bool
    source_lang: str = "es"

class TranscriptionServer:
    def __init__(
        self,
        server_config: ServerConfig,
        model_config: ModelConfig,
        recording_config: Optional[RecordingConfig] = None
    ):
        """Initialize the transcription server with the given configurations.
        
        Args:
            server_config: Server configuration parameters
            model_config: Whisper model configuration
            recording_config: Optional recording configuration
        """
        self.server_config = server_config
        self.model_config = model_config
        self.recording_config = recording_config or RecordingConfig()
        
        # Create recordings directory if saving is enabled
        if self.recording_config.save_output:
            os.makedirs(self.recording_config.output_dir, exist_ok=True)
        
        # Initialize the Whisper model
        logger.info(f"Loading Whisper model {model_config.model_name}...")
        try:
            self.model = WhisperModel(
                model_config.model_name,
                device=model_config.device,
                compute_type=model_config.compute_type
            )
        except Exception as e:
            raise ModelError(f"Failed to initialize Whisper model: {str(e)}") from e
        
        # Client management
        self.clients: Dict[str, ClientState] = {}
        self.lock = threading.Lock()

    def _preprocess_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """Preprocess audio data with high-pass filter and compression.
        
        Args:
            audio_data: Raw audio data as numpy array
            
        Returns:
            Preprocessed audio data
            
        Raises:
            AudioProcessingError: If preprocessing fails
        """
        try:
            # Ensure we're working with float32
            audio_data = audio_data.astype(np.float32)
            
            # Apply a high-pass filter
            try:
                from scipy import signal
                nyq = 0.5 * AudioConstants.SAMPLE_RATE
                cutoff = ProcessingConstants.HIGH_PASS_CUTOFF / nyq
                b, a = signal.butter(5, cutoff, btype='high', analog=False)
                audio_data = signal.filtfilt(b, a, audio_data).astype(np.float32)
            except ImportError:
                logger.warning("scipy not available, skipping high-pass filter")
            
            # Normalize before compression
            max_abs = np.max(np.abs(audio_data))
            if max_abs > ProcessingConstants.NOISE_FLOOR:
                audio_data = audio_data / max_abs
            
            # Apply compression
            threshold = ProcessingConstants.COMPRESSION_THRESHOLD
            ratio = ProcessingConstants.COMPRESSION_RATIO
            mask_above = np.abs(audio_data) > threshold
            mask_positive = audio_data > 0
            
            compressed = np.copy(audio_data)
            compressed[mask_above & mask_positive] = threshold + (audio_data[mask_above & mask_positive] - threshold) / ratio
            compressed[mask_above & ~mask_positive] = -(threshold + (np.abs(audio_data[mask_above & ~mask_positive]) - threshold) / ratio)
            
            # Final normalization
            return np.clip(compressed, -1.0, 1.0).astype(np.float32)
            
        except Exception as e:
            raise AudioProcessingError(f"Error in audio preprocessing: {str(e)}") from e

    async def _transcribe_audio(self, audio_data: np.ndarray) -> List[Dict[str, Any]]:
        """Transcribe audio data using the Whisper model.
        
        Args:
            audio_data: Preprocessed audio data
            
        Returns:
            List of transcription segments
            
        Raises:
            ModelError: If transcription fails
        """
        try:
            segments, info = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.model.transcribe(
                    audio_data,
                    language=self.model_config.source_lang,
                    vad_filter=self.model_config.use_vad,
                    vad_parameters=dict(
                        min_silence_duration_ms=VADConstants.MIN_SILENCE_DURATION_MS,
                        speech_pad_ms=VADConstants.SPEECH_PAD_MS,
                    )
                )
            )
            
            result = []
            for segment in segments:
                response = {
                    "type": MessageType.TRANSCRIPTION.value,
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text
                }
                logger.info(f"Transcription [{segment.start:.2f}s -> {segment.end:.2f}s]: {segment.text}")
                result.append(response)
            return result
            
        except Exception as e:
            raise ModelError(f"Error during transcription: {str(e)}") from e

    async def process_audio_stream(self, client_state: ClientState):
        """Process incoming audio stream for a client.
        
        Args:
            client_state: State object for the connected client
            
        Raises:
            AudioProcessingError: If audio processing fails
            ModelError: If transcription fails
        """
        accumulated_audio = np.array([], dtype=np.float32)
        
        while client_state.is_active:
            try:
                # Get audio chunk from queue with timeout
                try:
                    audio_chunk = await asyncio.get_event_loop().run_in_executor(
                        None, 
                        lambda: client_state.audio_queue.get(timeout=1.0)
                    )
                except Empty:
                    # Process remaining audio if any
                    if len(accumulated_audio) > 0:
                        logger.info("Processing remaining audio...")
                        try:
                            segments = await self._transcribe_audio(accumulated_audio)
                            for segment in segments:
                                try:
                                    await client_state.websocket.send(json.dumps(segment))
                                except websockets.exceptions.ConnectionClosed:
                                    logger.info("Client disconnected during transcription send")
                                    client_state.is_active = False
                                    break
                            accumulated_audio = np.array([], dtype=np.float32)
                        except ModelError as e:
                            logger.error(f"Error processing final audio: {str(e)}")
                    continue
                
                # Convert bytes to numpy array
                audio_data = np.frombuffer(audio_chunk, dtype=np.float32)
                
                # Skip empty chunks
                if len(audio_data) == 0:
                    continue
                
                # Save raw audio if enabled
                if self.recording_config.save_output:
                    client_state.recording_data.append(audio_chunk)
                
                # Preprocess and accumulate audio
                try:
                    audio_data = self._preprocess_audio(audio_data)
                    accumulated_audio = np.concatenate([accumulated_audio, audio_data])
                except AudioProcessingError as e:
                    logger.error(f"Audio preprocessing failed: {str(e)}")
                    continue
                
                # Process when we have enough audio
                if len(accumulated_audio) >= AudioConstants.CHUNK_SIZE:
                    try:
                        segments = await self._transcribe_audio(accumulated_audio)
                        for segment in segments:
                            try:
                                await client_state.websocket.send(json.dumps(segment))
                            except websockets.exceptions.ConnectionClosed:
                                logger.info("Client disconnected during transcription send")
                                client_state.is_active = False
                                break
                        
                        # Keep overlap for context
                        accumulated_audio = accumulated_audio[-AudioConstants.OVERLAP_SIZE:]
                        
                    except ModelError as e:
                        logger.error(f"Transcription failed: {str(e)}")
                        continue
                    
            except Exception as e:
                logger.error(f"Error processing audio stream: {str(e)}", exc_info=True)
                continue

        # Process any remaining audio before cleanup
        if len(accumulated_audio) > 0:
            logger.info("Processing final audio buffer before cleanup...")
            try:
                segments = await self._transcribe_audio(accumulated_audio)
                for segment in segments:
                    try:
                        await client_state.websocket.send(json.dumps(segment))
                    except websockets.exceptions.ConnectionClosed:
                        logger.info("Client disconnected during final transcription send")
                        break
            except ModelError as e:
                logger.error(f"Error processing final cleanup audio: {str(e)}")

        if self.recording_config.save_output and client_state.recording_data:
            logger.info("Saving final recording...")

    async def handle_client(self, websocket: WebSocketServerProtocol, path: str):
        """Handle individual client connections."""
        if len(self.clients) >= self.server_config.max_clients:
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
                if current_time - client_state.start_time > self.server_config.max_connection_time:
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
            if self.recording_config.save_output and client_state.recording_data:
                try:
                    self._save_recording(client_state.recording_data, client_id)
                    logger.info(f"Successfully saved recording for client {client_id}")
                except Exception as e:
                    logger.error(f"Failed to save recording: {str(e)}", exc_info=True)
            
            with self.lock:
                del self.clients[client_id]
            
            await process_task

    def _save_recording(self, recording_data: List[bytes], client_id: str):
        """Save the recorded audio to a WAV file.
        
        Args:
            recording_data: List of audio chunks in bytes
            client_id: Unique identifier for the client
            
        Raises:
            Exception: If saving fails
        """
        # Construct the full path for the recording
        filename = os.path.join(
            self.recording_config.output_dir,
            f"{self.recording_config.filename_prefix}_{client_id}.wav"
        )
        
        try:
            # Convert the bytes data to float32 numpy array
            audio_data = np.concatenate([
                np.frombuffer(chunk, dtype=np.float32) for chunk in recording_data
            ])
            
            # Ensure the data is in float32 range [-1, 1]
            audio_data = np.clip(audio_data, -1.0, 1.0)
            
            # Convert to 16-bit PCM with proper scaling
            audio_data = (audio_data * AudioConstants.MAX_AMPLITUDE).clip(
                -AudioConstants.MAX_AMPLITUDE, 
                AudioConstants.MAX_AMPLITUDE - 1
            ).astype(np.int16)
            
            with wave.open(filename, 'wb') as wav_file:
                wav_file.setnchannels(AudioConstants.CHANNELS)
                wav_file.setsampwidth(AudioConstants.BIT_DEPTH // 8)
                wav_file.setframerate(AudioConstants.SAMPLE_RATE)
                wav_file.writeframes(audio_data.tobytes())
            
            logger.info(f"Saved recording to {filename}")
            
        except Exception as e:
            logger.error(f"Error saving recording: {str(e)}", exc_info=True)
            raise

    async def start(self):
        """Start the WebSocket server."""
        async def wrapped_handler(websocket):
            await self.handle_client(websocket, "/")
        
        server = await websockets.serve(
            wrapped_handler,
            self.server_config.host,
            self.server_config.port,
            ping_interval=self.server_config.ping_interval,
            ping_timeout=self.server_config.ping_timeout
        )
        
        logger.info(
            f"WebSocket server started on "
            f"ws://{self.server_config.host}:{self.server_config.port}"
        )
        
        await server.wait_closed()

if __name__ == "__main__":
    # Create and run the server
    server = TranscriptionServer(
        server_config=ServerConfig(
            host="localhost",
            port=8765,
            max_clients=8,
            max_connection_time=600,
            ping_interval=30,
            ping_timeout=10
        ),
        model_config=ModelConfig(
            model_name="large-v3",
            device="cpu",
            compute_type="float32",
            source_lang="es",
            use_vad=True
        ),
        recording_config=RecordingConfig(
            save_output=True,
            output_dir="./recordings",
            filename_prefix="recording"
        )
    )
    
    # Run the server in the event loop
    asyncio.run(server.start())
