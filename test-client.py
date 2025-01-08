import asyncio
import websockets
import wave
import json
import argparse
from pathlib import Path

async def stream_audio(websocket, audio_file: Path, chunk_size: int = 4000):
    """Stream audio file in chunks to the websocket server."""
    with wave.open(str(audio_file), 'rb') as wav_file:
        while True:
            data = wav_file.readframes(chunk_size)
            if not data:
                break
            await websocket.send(data)
            await asyncio.sleep(0.1)  # Simulate real-time streaming

async def receive_transcriptions(websocket):
    """Receive and print transcriptions from the server."""
    try:
        while True:
            try:
                message = await websocket.recv()
                response = json.loads(message)
                
                if response.get("type") == "transcription":
                    print(f"[{response['start']:.2f}s -> {response['end']:.2f}s] {response['text']}")
                elif response.get("type") == "config_ack":
                    print(f"Configuration acknowledged: {response}")
                elif response.get("type") == "error":
                    print(f"Error: {response['message']}")
                    break
                elif response.get("type") == "status":
                    print(f"Status: {response['message']}")
            except websockets.exceptions.ConnectionClosed:
                print("Connection closed by server")
                break
            except Exception as e:
                print(f"Error receiving message: {e}")
                break
    except Exception as e:
        print(f"Error in receive_transcriptions: {e}")

async def main(audio_file: Path, target_lang: str = None):
    uri = "ws://localhost:8765"
    
    async with websockets.connect(uri) as websocket:
        print(f"Connected to {uri}")
        
        # Configure language if specified
        if target_lang:
            config = {
                "type": "config",
                "source_lang": target_lang  # Changed from translate_to to source_lang
            }
            await websocket.send(json.dumps(config))
            print(f"Configured source language: {target_lang}")
        
        # Start receiving transcriptions in the background
        receive_task = asyncio.create_task(receive_transcriptions(websocket))
        
        # Stream the audio file
        print("Starting audio stream...")
        await stream_audio(websocket, audio_file)
        print("Finished streaming audio")
        
        # Wait longer for final transcriptions
        print("Waiting for final transcriptions...")
        await asyncio.sleep(10)  # Increased from 2 to 10 seconds
        
        # Wait for receive task to complete
        try:
            await receive_task
        except Exception as e:
            print(f"Error waiting for receive task: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WebSocket Audio Client")
    parser.add_argument("audio_file", type=Path, help="Path to WAV audio file")
    parser.add_argument("--lang", type=str, help="Source language (e.g., 'es' for Spanish)", default="es")
    args = parser.parse_args()
    
    asyncio.run(main(args.audio_file, args.lang))
