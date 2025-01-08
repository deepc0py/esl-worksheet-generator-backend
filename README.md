# ESL Conversation Transcription Server

A real-time speech transcription server designed for ESL (English as a Second Language) conversation practice. This server uses OpenAI's Whisper model (via faster-whisper) to provide accurate Spanish speech transcription for ESL curriculum generation.

## Features

- Real-time Spanish speech transcription
- WebSocket-based streaming for low-latency processing
- Audio preprocessing for improved transcription quality:
  - High-pass filtering to remove background noise
  - Dynamic range compression
  - Automatic gain control
- Automatic recording saving for curriculum development
- Multi-client support (configurable, default 8 simultaneous connections)
- Configurable model size and processing parameters
- Detailed logging of transcriptions with timestamps

## Prerequisites

- Python 3.9+
- CUDA-capable GPU (optional, for faster processing)
- 8GB+ RAM (16GB+ recommended for large model)

## Installation

1. Create and activate a conda environment:
```bash
conda create -n whisperlive python=3.9
conda activate whisperlive
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Clone the repository:
```bash
git clone [repository-url]
cd esl-worksheet-generator-backend
```

## Configuration

The server can be configured through three main configuration classes:

### Server Configuration
```python
ServerConfig(
    host="localhost",          # Server host
    port=8765,                # WebSocket port
    max_clients=8,            # Maximum simultaneous connections
    max_connection_time=600,  # Maximum session duration (seconds)
    ping_interval=30,         # WebSocket ping interval
    ping_timeout=10          # WebSocket ping timeout
)
```

### Model Configuration
```python
ModelConfig(
    model_name="large-v3",    # Whisper model size
    device="cpu",             # Processing device ("cpu" or "cuda")
    compute_type="float32",   # Computation precision
    source_lang="es",         # Source language
    use_vad=True             # Voice Activity Detection
)
```

### Recording Configuration
```python
RecordingConfig(
    save_output=True,         # Whether to save recordings
    output_dir="./recordings", # Recording output directory
    filename_prefix="recording" # Recording filename prefix
)
```

## Usage

### Starting the Server

```bash
python transcription_server.py
```

The server will start on `ws://localhost:8765` by default.

### Audio Processing Parameters

- Sample Rate: 16kHz
- Chunk Size: 5 seconds of audio (configurable)
- Context Overlap: 1 second between chunks
- Channels: Mono
- Format: WAV
- Bit Depth: 16-bit

### WebSocket API

Clients can connect to the server using WebSockets and:

1. Send audio data as binary messages
2. Receive JSON messages with transcriptions:
```json
{
    "type": "transcription",
    "text": "transcribed text",
    "start": 0.0,
    "end": 5.0
}
```

### Configuration Messages

Clients can send configuration messages:
```json
{
    "type": "config",
    "source_lang": "es"
}
```

## Performance Considerations

- The large-v3 model provides the best transcription quality but requires more processing power
- CPU processing is slower but more widely compatible
- Each client connection requires approximately 2GB of RAM
- Audio is processed in 5-second segments with 1-second overlap for optimal context

## Limitations

- Spanish-only transcription currently supported
- Maximum session duration of 10 minutes per connection
- Requires stable network connection for real-time streaming

## Directory Structure

```
.
├── transcription_server.py    # Main server implementation
├── requirements.txt          # Python dependencies
├── recordings/               # Saved audio recordings (gitignored)
└── README.md                # This file
```

## Error Handling

The server implements robust error handling for:
- Audio processing errors
- Model inference failures
- Network disconnections
- Invalid message formats
- Resource exhaustion

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
