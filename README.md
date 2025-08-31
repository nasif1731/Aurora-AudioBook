# Aurora AudioBook 

Transform any PDF into an audiobook using your own voice! This application combines OCR, text-to-speech, and voice cloning to create personalized audiobooks.

## Features

- **PDF Processing**: Automatically detects scanned vs. searchable PDFs
- **OCR Support**: Converts scanned PDFs to searchable text using OCR
- **Text-to-Speech**: Uses Deepgram's high-quality TTS models
- **Voice Cloning**: Clones your voice using OpenVoice technology
- **Smart Chunking**: Intelligently splits text into optimal audio segments
- **Modern UI**: Clean, professional Gradio interface

## Quick Start

### Prerequisites

- Python 3.8+
- FFmpeg installed on your system
- Deepgram API key

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd genAI_task_01
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install system dependencies:**
   ```bash
   # On Ubuntu/Debian:
   sudo apt-get install ocrmypdf ffmpeg tesseract-ocr tesseract-ocr-eng
   
   # On macOS:
   brew install ocrmypdf ffmpeg tesseract
   
   # On Windows:
   # Download and install FFmpeg and Tesseract from their official websites
   ```

4. **Set up environment variables:**
   ```bash
   # Copy the example file
   cp .env.example .env
   
   # Edit .env and add your Deepgram API key
   DEEPGRAM_API_KEY=your_actual_api_key_here
   ```

### Usage

1. **Run the application:**
   ```bash
   python app.py
   ```

2. **Open your browser** and navigate to the local URL (usually `http://localhost:7860`)

3. **Upload files:**
   - PDF document (any size, scanned or searchable)
   - Voice sample (15-45 seconds of clear speech in WAV or MP3 format)

4. **Configure settings:**
   - Max characters to process (for quick testing)
   - Number of chunks to synthesize

5. **Click "Generate Audiobook"** and wait for processing

## Configuration

### Environment Variables

- `DEEPGRAM_API_KEY`: Your Deepgram API key (required)
- `DG_MODEL`: TTS model to use (defaults to "aura-2-thalia-en")

### Deepgram Models

Available models include:
- `aura-2-thalia-en`: High-quality English voice
- `aura-2-asteria-en`: Alternative English voice
- `aura-2-orion-en`: Male English voice

## Project Structure

```
genAI_task_01/
├── app.py              # Main application
├── requirements.txt    # Python dependencies
├── packages.txt        # System dependencies
├── README.md          # This file
├── .env.example       # Environment variables template
└── data/              # Generated during runtime
    ├── inputs/        # Uploaded files
    ├── outputs/       # Generated audio and metadata
    ├── audio_chunks/  # Individual TTS chunks
    └── meta/          # Processing metadata
```

## How It Works

1. **PDF Analysis**: Detects if PDF is scanned or searchable
2. **OCR Processing**: Converts scanned PDFs to text (if needed)
3. **Text Extraction**: Extracts and cleans text content
4. **Smart Chunking**: Splits text into optimal audio segments
5. **TTS Generation**: Creates audio using Deepgram's API
6. **Voice Cloning**: Applies your voice characteristics using OpenVoice
7. **Audio Assembly**: Combines all chunks into final audiobook

## Audio Quality

- **Input**: Supports WAV and MP3 voice samples
- **Output**: High-quality MP3 with 192kbps bitrate
- **Processing**: Automatic audio normalization to 16kHz mono
- **Chunking**: Optimal segment lengths (800-1800 characters)

## Troubleshooting

### Common Issues

1. **Missing Deepgram API Key**
   - Ensure your `.env` file contains the correct API key
   - Verify the key is valid in Deepgram console

2. **OCR Failures**
   - Ensure Tesseract is properly installed
   - Check that `ocrmypdf` is available in PATH

3. **Audio Processing Errors**
   - Verify FFmpeg is installed and accessible
   - Check audio file formats are supported

4. **OpenVoice Issues**
   - Ensure CUDA is available for GPU acceleration
   - Check OpenVoice CLI installation

### Performance Tips

- Use shorter voice samples (15-30 seconds) for faster processing
- Limit character count for quick testing
- Process fewer chunks initially to verify setup

## Security Notes

- Never commit your `.env` file with API keys
- Keep your Deepgram API key secure
- The application runs locally and doesn't upload files to external servers

## License

This project is open source. Please check individual dependency licenses.

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## Support

For issues related to:
- **Deepgram**: Check their [documentation](https://developers.deepgram.com/)
- **OpenVoice**: Visit their [GitHub repository](https://github.com/myshell-ai/OpenVoice)
- **OCR**: Review [OCRmyPDF documentation](https://ocrmypdf.readthedocs.io/)

---

**Happy audiobook creation!**

