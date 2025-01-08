# ESL Conversation Transcription Server

A real-time speech transcription server designed for ESL (English as a Second Language) conversation practice. This server uses OpenAI's Whisper model (via faster-whisper) to provide accurate Spanish speech transcription for ESL curriculum generation.

## Features

- Real-time Spanish speech transcription
- WebSocket-based streaming for low-latency processing
- Audio preprocessing for improved transcription quality:
  - High-pass filtering to remove background noise
  - Audio compression for better speech clarity
  - Automatic gain control
- Automatic recording saving for curriculum development
- Multi-client support (up to 8 simultaneous connections)
- Configurable model size and processing parameters

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
pip install faster-whisper websockets numpy scipy
```

3. Clone the repository:
```bash
git clone [repository-url]
cd esl-worksheet-generator-backend
```

## Usage

### Starting the Server

```bash
python transcription_server.py
```

The server will start on `ws://localhost:8765` by default.

### Configuration Options

The server can be configured with various parameters:

- `model`: Whisper model size (default: "large-v3")
- `device`: Processing device ("cpu" or "cuda", default: "cpu")
- `compute_type`: Computation precision ("float32" or "float16", default: "float32")
- `max_clients`: Maximum simultaneous connections (default: 8)
- `save_output_recording`: Whether to save audio recordings (default: True)

### Testing

A test client is provided to verify server functionality:

```bash
python test-client.py path/to/audio.wav
```

The test client supports the following options:
- `--lang`: Source language (default: "es" for Spanish)

### Audio Requirements

- Sample Rate: 16kHz
- Channels: Mono
- Format: WAV
- Bit Depth: 16-bit

### Directory Structure

```
.
├── transcription_server.py    # Main server implementation
├── test-client.py            # Test client for server verification
├── recordings/               # Saved audio recordings (gitignored)
└── README.md                # This file
```

## Integration with ESL Curriculum Generator

This server is designed to work with our ESL curriculum generator by:
1. Recording and transcribing Spanish conversations
2. Providing accurate transcriptions for worksheet generation
3. Maintaining an archive of conversation recordings for reference

### WebSocket API

Clients can connect to the server using WebSockets and:

1. Send audio data as binary messages
2. Receive JSON messages with transcriptions:
```json
{
    "type": "transcription",
    "text": "transcribed text",
    "start": 0.0,
    "end": 2.0
}
```

## Performance Considerations

- The large-v3 model provides the best transcription quality but requires more processing power
- CPU processing is slower but more widely compatible
- Each client connection requires approximately 2GB of RAM
- Audio is processed in 2-second segments for real-time performance

## Limitations

- Spanish-only transcription currently supported
- Maximum audio length of 10 minutes per session
- Requires stable network connection for real-time streaming

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
