import os
import shutil
import re
import json
import subprocess
import requests
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import tempfile
import logging
import sys

import gradio as gr
import fitz 
from dotenv import load_dotenv
from pydub import AudioSegment


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


load_dotenv()

class AudioBookConfig:
    
    def __init__(self):
        self.deepgram_api_key = os.getenv("DEEPGRAM_API_KEY", "").strip()
        self.voice_model = os.getenv("VOICE_MODEL", "aura-2-thalia-en")
        self.workspace_root = Path("./workspace")
        self.setup_directories()
    
    def setup_directories(self):
        self.source_files = self.workspace_root / "source_files"
        self.generated_content = self.workspace_root / "generated_content"
        self.audio_segments = self.generated_content / "audio_segments"
        self.metadata_storage = self.generated_content / "metadata"
        
        for directory in [self.source_files, self.generated_content, 
                         self.audio_segments, self.metadata_storage]:
            directory.mkdir(parents=True, exist_ok=True)


config = AudioBookConfig()

class FileUtilities:
    """Utility class for file operations and path management"""
    
    @staticmethod
    def compare_file_paths(path_a: Path, path_b: Path) -> bool:
        """Compare if two paths point to the same location"""
        try:
            return path_a.resolve() == path_b.resolve()
        except (OSError, ValueError):
            return str(path_a) == str(path_b)
    
    @staticmethod
    def secure_copy(source: Path, destination: Path) -> bool:
        """Safely copy file from source to destination"""
        try:
            if not FileUtilities.compare_file_paths(source, destination):
                shutil.copy2(source, destination)
                logger.info(f"File copied: {source} -> {destination}")
                return True
            else:
                logger.info(f"File already at destination: {destination}")
                return True
        except Exception as e:
            logger.error(f"Failed to copy file: {e}")
            return False

class PDFProcessor:
   
    
    def __init__(self, sample_pages: int = 3):
        self.sample_pages = sample_pages
    
    def detect_scanned_document(self, pdf_file: Path) -> bool:
        try:
            document = fitz.open(pdf_file)
            total_pages = min(self.sample_pages, len(document))
            
            text_found_pages = 0
            for page_num in range(total_pages):
                page = document.load_page(page_num)
                page_text = page.get_text().strip()
                if page_text:
                    text_found_pages += 1
            
            document.close()
            return text_found_pages == 0
            
        except Exception as e:
            logger.error(f"Error analyzing PDF: {e}")
            return True  
    
    def perform_ocr_conversion(self, input_file: Path, output_file: Path) -> bool:
        """Convert scanned PDF to searchable PDF using OCR"""
        try:
            ocr_command = [
                'ocrmypdf',
                '--force-ocr',
                '--skip-text',
                '--output-type', 'pdf',
                str(input_file),
                str(output_file)
            ]
            
            result = subprocess.run(
                ocr_command, 
                capture_output=True, 
                text=True, 
                timeout=300
            )
            
            if result.returncode == 0:
                logger.info(f"OCR conversion successful: {output_file}")
                return True
            else:
                logger.error(f"OCR conversion failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("OCR process timed out")
            return False
        except Exception as e:
            logger.error(f"OCR conversion error: {e}")
            return False
    
    def extract_document_text(self, pdf_file: Path) -> str:
        try:
            document = fitz.open(pdf_file)
            extracted_pages = []
            
            for page_index in range(len(document)):
                page = document.load_page(page_index)
                page_content = page.get_text("text")
                extracted_pages.append(page_content)
            
            document.close()
            return "\n".join(extracted_pages)
            
        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
            return ""

class TextProcessor:
    
    def __init__(self, min_segment_length: int = 600, max_segment_length: int = 1500):
        self.min_segment_length = min_segment_length
        self.max_segment_length = max_segment_length
    
    def clean_extracted_text(self, raw_text: str) -> str:
        if not raw_text:
            return ""
        
        text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', raw_text)
        
        text = re.sub(r'[ \t]*\n[ \t]*', ' ', text)
        
        text = re.sub(r'\s+', ' ', text)
        
        text = re.sub(r'([.!?]){2,}', r'\1', text)
        
        return text.strip()
    
    def create_text_segments(self, content: str) -> List[Tuple[str, str]]:
        if not content:
            return []
        
        sentence_pattern = r'([.!?])\s+'
        parts = re.split(sentence_pattern, content)
        
        sentences = []
        for i in range(0, len(parts), 2):
            sentence = parts[i]
            if i + 1 < len(parts):
                sentence += parts[i + 1]
            sentences.append(sentence.strip())
        
        segments = []
        current_segment = ""
        
        for sentence in sentences:
            if not sentence:
                continue
                
            potential_length = len(current_segment) + len(sentence) + 1
            
            if potential_length <= self.max_segment_length:
                current_segment += (" " + sentence) if current_segment else sentence
            elif len(current_segment) >= self.min_segment_length:
                segments.append(current_segment.strip())
                current_segment = sentence
            else:
                current_segment += (" " + sentence) if current_segment else sentence
        
        if current_segment.strip():
            segments.append(current_segment.strip())
        
        indexed_segments = []
        for index, segment in enumerate(segments):
            segment_id = f"segment_{index:04d}"
            indexed_segments.append((segment_id, segment))
        
        return indexed_segments

class AudioProcessor:
    
    def __init__(self, config: AudioBookConfig):
        self.config = config
    
    def standardize_audio_format(self, input_file: Path, output_file: Path) -> bool:
        try:
            ffmpeg_cmd = [
                'ffmpeg', '-y',
                '-i', str(input_file),
                '-ac', '1',  
                '-ar', '16000', 
                '-vn',  
                str(output_file)
            ]
            
            result = subprocess.run(
                ffmpeg_cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                logger.info(f"Audio normalization successful: {output_file}")
                return True
            else:
                logger.error(f"Audio normalization failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Audio processing error: {e}")
            return False
    
    def generate_speech_from_text(self, text_content: str, output_file: Path) -> bool:
        if not self.config.deepgram_api_key:
            raise ValueError("Deepgram API key is required for text-to-speech conversion")
        
        try:
            api_endpoint = "https://api.deepgram.com/v1/speak"
            request_headers = {
                "Authorization": f"Token {self.config.deepgram_api_key}",
                "Content-Type": "application/json"
            }
            
            api_parameters = {
                "model": self.config.voice_model,
                "encoding": "mp3",
                "bit_rate": 48000
            }
            
            request_data = {"text": text_content}
            
            response = requests.post(
                api_endpoint,
                headers=request_headers,
                params=api_parameters,
                json=request_data,
                stream=True,
                timeout=180
            )
            
            if response.status_code == 200:
                with open(output_file, "wb") as audio_file:
                    for data_chunk in response.iter_content(chunk_size=8192):
                        if data_chunk:
                            audio_file.write(data_chunk)
                
                logger.info(f"TTS generation successful: {output_file}")
                return True
            else:
                logger.error(f"TTS API request failed: {response.status_code} - {response.text[:200]}")
                return False
                
        except requests.RequestException as e:
            logger.error(f"TTS API request error: {e}")
            return False
        except Exception as e:
            logger.error(f"TTS generation error: {e}")
            return False
    
    def process_text_segments(self, text_segments: List[Tuple[str, str]], 
                            max_segments: Optional[int] = None, 
                            progress_callback=None) -> List[Dict[str, Any]]:
        """Process multiple text segments into audio files"""
        if max_segments:
            segments_to_process = text_segments[:max_segments]
        else:
            segments_to_process = text_segments
        
        audio_metadata = []
        total_segments = len(segments_to_process)
        
        for index, (segment_id, text_content) in enumerate(segments_to_process, 1):
            output_file = self.config.audio_segments / f"{segment_id}.mp3"
            
            if progress_callback:
                progress_callback(f"AUDIO: Generating audio {index}/{total_segments}: {segment_id} ({len(text_content)} characters)")
            
            if self.generate_speech_from_text(text_content, output_file):
                audio_metadata.append({
                    "segment_id": segment_id,
                    "audio_file": str(output_file),
                    "character_count": len(text_content),
                    "processing_order": index
                })
            else:
                logger.warning(f"Failed to generate audio for segment: {segment_id}")
        
        return audio_metadata
    
    def combine_audio_segments(self, audio_metadata: List[Dict[str, Any]], 
                             output_file: Path, silence_duration: int = 750) -> bool:
        try:
            combined_audio = AudioSegment.silent(duration=0)
            silence_gap = AudioSegment.silent(duration=silence_duration)
            
            for segment_info in audio_metadata:
                audio_file = segment_info["audio_file"]
                if Path(audio_file).exists():
                    audio_segment = AudioSegment.from_file(audio_file)
                    combined_audio += audio_segment + silence_gap
                else:
                    logger.warning(f"Audio file not found: {audio_file}")
            
            # Export combined audio
            combined_audio.export(output_file, format="mp3", bitrate="192k")
            logger.info(f"Audio combination successful: {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Audio combination error: {e}")
            return False
    
    def apply_voice_cloning(self, source_audio: Path, reference_voice: Path, 
                        output_file: Path) -> bool:
        try:
            # Normalize the reference voice to 16kHz mono WAV
            normalized_voice = self.config.generated_content / "reference_voice_normalized.wav"
            if not self.standardize_audio_format(reference_voice, normalized_voice):
                return False

            # Pick device
            device_type = "cuda" if os.path.exists("/proc/driver/nvidia/version") else "cpu"

            # OpenVoice CLI command
            openvoice_cmd = [
                sys.executable, "-m", "openvoice_cli", "single",
                "-i", str(source_audio),
                "-r", str(normalized_voice),
                "-o", str(output_file),
                "-d", device_type
            ]

            result = subprocess.run(
                openvoice_cmd,
                capture_output=True,
                text=True,
                timeout=600  # longer for big audiobooks
            )

            if result.returncode == 0:
                logger.info(f"✅ Voice cloning successful: {output_file}")
                return True
            else:
                logger.error(f"❌ Voice cloning failed: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            logger.error("❌ Voice cloning process timed out")
            return False
        except Exception as e:
            logger.error(f"❌ Voice cloning error: {e}")
            return False



class AudioBookGenerator:
    
    def __init__(self):
        self.config = config
        self.file_utils = FileUtilities()
        self.pdf_processor = PDFProcessor()
        self.text_processor = TextProcessor()
        self.audio_processor = AudioProcessor(self.config)
        self.processing_log = []
    
    def log_progress(self, message: str):
        """Add message to processing log"""
        self.processing_log.append(message)
        logger.info(message)
    
    def validate_inputs(self, pdf_file, voice_file) -> bool:
        if pdf_file is None:
            self.log_progress("ERROR: PDF file is required")
            return False
        if voice_file is None:
            self.log_progress("ERROR: Voice sample file is required")
            return False
        return True
    
    def prepare_input_files(self, pdf_file, voice_file) -> Tuple[Optional[Path], Optional[Path]]:
        """Copy input files to workspace"""
        try:
            source_pdf = Path(pdf_file.name)
            source_voice = Path(voice_file.name)
            
            pdf_destination = self.config.source_files / source_pdf.name
            voice_destination = self.config.source_files / source_voice.name
            
            if self.file_utils.secure_copy(source_pdf, pdf_destination):
                self.log_progress(f"PDF document prepared: {pdf_destination.name}")
            else:
                self.log_progress("ERROR: Failed to prepare PDF file")
                return None, None
            
          
            if self.file_utils.secure_copy(source_voice, voice_destination):
                self.log_progress(f"Voice sample prepared: {voice_destination.name}")
            else:
                self.log_progress("ERROR: Failed to prepare voice file")
                return None, None
            
            return pdf_destination, voice_destination
            
        except Exception as e:
            self.log_progress(f"ERROR: File preparation error: {e}")
            return None, None
    
    def extract_and_process_text(self, pdf_file: Path, character_limit: Optional[int] = None) -> str:
        """Extract and process text from PDF"""
        try:
            
            self.log_progress("ANALYZING: PDF document structure...")
            needs_ocr = self.pdf_processor.detect_scanned_document(pdf_file)
            
            if needs_ocr:
                self.log_progress("PROCESSING: Document requires OCR processing (this may take several minutes)...")
                ocr_output = self.config.generated_content / f"{pdf_file.stem}_ocr_processed.pdf"
                
                if self.pdf_processor.perform_ocr_conversion(pdf_file, ocr_output):
                    text_content = self.pdf_processor.extract_document_text(ocr_output)
                else:
                    self.log_progress("ERROR: OCR processing failed")
                    return ""
            else:
                self.log_progress("SUCCESS: Document contains searchable text - extracting...")
                text_content = self.pdf_processor.extract_document_text(pdf_file)
            
          
            processed_text = self.text_processor.clean_extracted_text(text_content)
            
           
            if character_limit and len(processed_text) > character_limit:
                self.log_progress(f"LIMITING: Applying character limit: {len(processed_text):,} -> {character_limit:,} characters")
                processed_text = processed_text[:character_limit]
            
            self.log_progress(f"COMPLETE: Text processing complete: {len(processed_text):,} characters")
            return processed_text
            
        except Exception as e:
            self.log_progress(f"ERROR: Text extraction error: {e}")
            return ""
    
    def generate_complete_audiobook(self, pdf_file, voice_file, 
                                  max_characters: int = 4000, 
                                  max_segments: int = 2) -> Tuple[str, Optional[str]]:
        
        self.processing_log = []  
        
        try:
          
            if not self.validate_inputs(pdf_file, voice_file):
                return "\n".join(self.processing_log), None
            
            
            pdf_path, voice_path = self.prepare_input_files(pdf_file, voice_file)
            if not pdf_path or not voice_path:
                return "\n".join(self.processing_log), None
            
            
            text_content = self.extract_and_process_text(pdf_path, max_characters)
            if not text_content:
                self.log_progress("ERROR: No text content extracted from PDF")
                return "\n".join(self.processing_log), None
            
            
            text_segments = self.text_processor.create_text_segments(text_content)
            total_segments = len(text_segments)
            segments_to_process = min(total_segments, max_segments) if max_segments else total_segments
            
            self.log_progress(f"SEGMENTS: Created {total_segments} text segments (processing {segments_to_process})")
            
            if not text_segments:
                self.log_progress("ERROR: No valid text segments created")
                return "\n".join(self.processing_log), None
            
            
            self.log_progress("GENERATING: Starting text-to-speech generation...")
            audio_metadata = self.audio_processor.process_text_segments(
                text_segments, 
                max_segments, 
                self.log_progress
            )
            
            if not audio_metadata:
                self.log_progress("ERROR: No audio segments generated")
                return "\n".join(self.processing_log), None
            
            
            self.log_progress("COMBINING: Merging audio segments...")
            combined_audio_file = self.config.generated_content / "combined_audiobook.mp3"
            
            if not self.audio_processor.combine_audio_segments(audio_metadata, combined_audio_file):
                self.log_progress("ERROR: Failed to combine audio segments")
                return "\n".join(self.processing_log), None
            
            
            self.log_progress("CLONING: Applying voice cloning transformation...")
            final_audiobook = self.config.generated_content / "final_audiobook_with_cloned_voice.mp3"
            
            if self.audio_processor.apply_voice_cloning(combined_audio_file, voice_path, final_audiobook):
                self.log_progress("SUCCESS: Audiobook generation completed successfully!")
                return "\n".join(self.processing_log), str(final_audiobook)
            else:
                self.log_progress("WARNING: Voice cloning failed - using original TTS voice")
                return "\n".join(self.processing_log), str(combined_audio_file)
                
        except Exception as e:
            self.log_progress(f"ERROR: Unexpected error: {e}")
            return "\n".join(self.processing_log), None

# Initialize the main generator
audiobook_generator = AudioBookGenerator()
class AuroraInterface:
    def __init__(self, generator: AudioBookGenerator):
        self.generator = generator
        self.setup_interface()

    def setup_interface(self):
        """Create Aurora dashboard-style UI"""
        with gr.Blocks(
            css=self.get_custom_styles(),
            theme=gr.themes.Soft(),
            title="Aurora Audiobook Studio"
        ) as self.interface:

            # --- Header banner ---
            gr.HTML(
                """
                <div class="header">
                    <h1>🌌 Aurora Audiobook Studio</h1>
                    <p>Transform your books into immersive audio — with your own voice.</p>
                </div>
                """
            )

            # --- Main layout ---
            with gr.Row():
                # Sidebar inputs
                with gr.Column(scale=1, elem_classes=["sidebar"]):
                    gr.Markdown("### Upload Files")
                    pdf_input = gr.File(label="📄 PDF File", file_types=[".pdf"])
                    voice_input = gr.File(label="🎤 Voice Sample", file_types=[".wav", ".mp3"])

                    gr.Markdown("### Settings")
                    char_limit = gr.Slider(
                        1000, 100000, value=5000, step=1000, label="Max Characters"
                    )
                    seg_limit = gr.Slider(
                        1, 50, value=3, step=1, label="Max Segments"
                    )

                    generate_btn = gr.Button("🚀 Generate Audiobook", elem_classes=["generate-btn"])

                # Content outputs
                with gr.Column(scale=2, elem_classes=["content-area"]):
                    status_output = gr.Label(label="Progress", value="Idle")

                    gr.Markdown("### 🎵 Your Audiobook")
                    audio_output = gr.Audio(type="filepath", label="Play Result")

                    download_output = gr.File(label="⬇️ Download MP3")

            # --- Footer ---
            gr.HTML(
                """
                <div class="footer">
                    <p>⚡ Powered by Aurora Studio | Built with Gradio + OpenVoice</p>
                </div>
                """
            )

            # Wire events
            generate_btn.click(
                fn=self.process_audiobook_request,
                inputs=[pdf_input, voice_input, char_limit, seg_limit],
                outputs=[status_output, audio_output, download_output],
                api_name="aurora_generate"
            )

    def process_audiobook_request(self, pdf_file, voice_file, char_limit, seg_limit):
        """Handle audiobook generation request with live status updates"""
        try:
            if pdf_file is None or voice_file is None:
                return "❌ Missing input", None, None

            # --- Stage 1: Extract text ---
            status = "📖 Extracting text from PDF..."
            yield status, None, None

            text_content = self.generator.extract_and_process_text(Path(pdf_file.name), char_limit)
            if not text_content:
                yield "❌ Failed to extract text", None, None
                return

            # --- Stage 2: Segment + TTS ---
            status = "🔊 Generating TTS segments..."
            yield status, None, None

            text_segments = self.generator.text_processor.create_text_segments(text_content)
            audio_metadata = self.generator.audio_processor.process_text_segments(
                text_segments, max_segments=seg_limit
            )
            if not audio_metadata:
                yield "❌ Failed during TTS", None, None
                return

            # --- Stage 3: Combine ---
            status = "🎼 Combining audio segments..."
            yield status, None, None

            combined_audio = self.generator.config.generated_content / "combined_audiobook.mp3"
            if not self.generator.audio_processor.combine_audio_segments(audio_metadata, combined_audio):
                yield "❌ Failed combining audio", None, None
                return

            # --- Stage 4: Clone voice ---
            status = "🧬 Cloning voice..."
            yield status, None, None

            final_audiobook = self.generator.config.generated_content / "final_audiobook.mp3"
            if self.generator.audio_processor.apply_voice_cloning(combined_audio, Path(voice_file.name), final_audiobook):
                yield "✅ Done", str(final_audiobook), str(final_audiobook)
            else:
                yield "⚠️ Voice cloning failed, using TTS output", str(combined_audio), str(combined_audio)

        except Exception as e:
            logger.error(f"Aurora UI error: {e}")
            yield f"❌ Error: {str(e)}", None, None

    def get_custom_styles(self):
        """Aurora theme styles"""
        return """
        .header {
            background: linear-gradient(90deg, #4f46e5, #9333ea);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            margin-bottom: 20px;
        }
        .sidebar {
            background: #f3f4f6;
            padding: 15px;
            border-radius: 8px;
        }
        .content-area {
            background: white;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #e5e7eb;
        }
        .generate-btn {
            background: #4f46e5;
            color: white;
            font-weight: bold;
            padding: 12px;
            border-radius: 6px;
            width: 100%;
        }
        .generate-btn:hover {
            background: #3730a3;
        }
        .footer {
            margin-top: 20px;
            text-align: center;
            font-size: 0.9rem;
            color: #6b7280;
        }
        """

    def launch_application(self, **kwargs):
        """Launch Aurora UI"""
        default_kwargs = {
            "server_name": "0.0.0.0",
            "server_port": 7860,
            "show_error": True,
            "quiet": False
        }
        default_kwargs.update(kwargs)
        self.interface.queue().launch(**default_kwargs)  # queue() needed for yield




def main():
    """Main application entry point"""
    try:
        
        app_interface = AuroraInterface(audiobook_generator)
        
        
        app_interface.launch_application()
        
    except Exception as e:
        logger.error(f"Application startup error: {e}")
        print(f"ERROR: Failed to start application: {e}")

if __name__ == "__main__":
    main()
