

import os
from pathlib import Path


from app import (
    AudioBookConfig, 
    FileUtilities, 
    PDFProcessor, 
    TextProcessor, 
    AudioProcessor,
    logger
)


PDF_FILE_PATH = r"C:\Users\nasif\Downloads\genAI_task_01\Life_30_Being_Human_-_Max_Tegmark.pdf" 
VOICE_FILE_PATH = r"C:\Users\nasif\Downloads\genAI_task_01\New-Recording-4.wav"  
MAX_CHARACTERS = 5000  
MAX_SEGMENTS = 3       


SAVE_LOG = True       
LOG_FILE = "test_processing_log.txt"  


class SimpleAudioBookTester:
    """Simple audiobook generator for testing with direct file paths"""
    
    def __init__(self):
        self.config = AudioBookConfig()
        self.file_utils = FileUtilities()
        self.pdf_processor = PDFProcessor()
        self.text_processor = TextProcessor()
        self.audio_processor = AudioProcessor(self.config)
        self.processing_log = []
    
    def log_message(self, message: str):
        """Log message and print to console"""
        self.processing_log.append(message)
        print(message)
        logger.info(message)
    
    def validate_files(self, pdf_path: str, voice_path: str):
        """Check if input files exist and are valid"""
        pdf_file = Path(pdf_path)
        voice_file = Path(voice_path)
        
        if not pdf_file.exists():
            self.log_message(f"ERROR: PDF file not found: {pdf_path}")
            return False
        
        if not pdf_file.suffix.lower() == '.pdf':
            self.log_message(f"ERROR: File is not a PDF: {pdf_path}")
            return False
        
        if not voice_file.exists():
            self.log_message(f"ERROR: Voice file not found: {voice_path}")
            return False
        
        if voice_file.suffix.lower() not in ['.wav', '.mp3']:
            self.log_message(f"ERROR: Voice file must be WAV or MP3: {voice_path}")
            return False
        
        self.log_message(f"SUCCESS: PDF file validated - {pdf_file.name}")
        self.log_message(f"SUCCESS: Voice file validated - {voice_file.name}")
        return True
    
    def process_audiobook(self, pdf_path: str, voice_path: str, 
                         max_chars=None, max_segments=None):
        """Complete audiobook generation process"""
        
        self.log_message("=" * 60)
        self.log_message("STARTING AUDIOBOOK GENERATION")
        self.log_message("=" * 60)
        
        try:
            
            if not self.validate_files(pdf_path, voice_path):
                return None
            
            
            pdf_file = Path(pdf_path)
            voice_file = Path(voice_path)
            
            
            self.log_message("PREPARING: Copying files to workspace...")
            workspace_pdf = self.config.source_files / f"test_{pdf_file.name}"
            workspace_voice = self.config.source_files / f"test_{voice_file.name}"
            
            if not self.file_utils.secure_copy(pdf_file, workspace_pdf):
                self.log_message("ERROR: Failed to copy PDF to workspace")
                return None
            
            if not self.file_utils.secure_copy(voice_file, workspace_voice):
                self.log_message("ERROR: Failed to copy voice file to workspace")
                return None
            
            
            self.log_message("ANALYZING: Processing PDF document...")
            text_content = self.extract_text_from_pdf(workspace_pdf, max_chars)
            
            if not text_content:
                self.log_message("ERROR: No text extracted from PDF")
                return None
            
           
            self.log_message("SEGMENTING: Creating text chunks...")
            text_segments = self.text_processor.create_text_segments(text_content)
            total_segments = len(text_segments)
            
            if max_segments:
                segments_to_process = min(total_segments, max_segments)
                text_segments = text_segments[:segments_to_process]
            else:
                segments_to_process = total_segments
            
            self.log_message(f"SEGMENTS: Created {total_segments} total, processing {segments_to_process}")
            
            if not text_segments:
                self.log_message("ERROR: No text segments created")
                return None
            
            
            self.log_message("GENERATING: Converting text to speech...")
            audio_files = self.generate_audio_segments(text_segments)
            
            if not audio_files:
                self.log_message("ERROR: No audio generated")
                return None
            
            
            self.log_message("COMBINING: Merging audio files...")
            combined_file = self.combine_audio_files(audio_files, pdf_file.stem)
            
            if not combined_file:
                self.log_message("ERROR: Failed to combine audio")
                return None
            
            
            self.log_message("CLONING: Applying voice transformation...")
            final_file = self.apply_voice_cloning(combined_file, workspace_voice, pdf_file.stem)
            
            
            if final_file:
                self.log_message("=" * 60)
                self.log_message("SUCCESS: AUDIOBOOK GENERATION COMPLETED!")
                self.log_message(f"OUTPUT FILE: {final_file}")
                self.log_message(f"FILE SIZE: {Path(final_file).stat().st_size / (1024*1024):.1f} MB")
                self.log_message("=" * 60)
                return final_file
            else:
                self.log_message("ERROR: Final processing failed")
                return None
                
        except Exception as e:
            self.log_message(f"ERROR: Unexpected error occurred: {e}")
            return None
    
    def extract_text_from_pdf(self, pdf_file, max_chars=None):
        """Extract and clean text from PDF"""
        try:
            
            needs_ocr = self.pdf_processor.detect_scanned_document(pdf_file)
            
            if needs_ocr:
                self.log_message("OCR: Document is scanned, applying OCR...")
                ocr_file = self.config.generated_content / f"{pdf_file.stem}_ocr.pdf"
                
                if self.pdf_processor.perform_ocr_conversion(pdf_file, ocr_file):
                    text = self.pdf_processor.extract_document_text(ocr_file)
                else:
                    self.log_message("ERROR: OCR processing failed")
                    return None
            else:
                self.log_message("TEXT: Extracting text from searchable PDF...")
                text = self.pdf_processor.extract_document_text(pdf_file)
            
            
            cleaned_text = self.text_processor.clean_extracted_text(text)
            
            
            if max_chars and len(cleaned_text) > max_chars:
                self.log_message(f"LIMITING: Truncating from {len(cleaned_text):,} to {max_chars:,} characters")
                cleaned_text = cleaned_text[:max_chars]
            
            self.log_message(f"TEXT: Extracted {len(cleaned_text):,} characters")
            return cleaned_text
            
        except Exception as e:
            self.log_message(f"ERROR: Text extraction failed: {e}")
            return None
    
    def generate_audio_segments(self, text_segments):
        """Generate audio for each text segment"""
        audio_files = []
        total = len(text_segments)
        
        for i, (segment_id, text) in enumerate(text_segments, 1):
            self.log_message(f"AUDIO: Generating segment {i}/{total} - {segment_id}")
            
            output_file = self.config.audio_segments / f"{segment_id}.mp3"
            
            if self.audio_processor.generate_speech_from_text(text, output_file):
                audio_files.append({
                    "segment_id": segment_id,
                    "audio_file": str(output_file),
                    "character_count": len(text)
                })
            else:
                self.log_message(f"WARNING: Failed to generate audio for {segment_id}")
        
        self.log_message(f"AUDIO: Generated {len(audio_files)} audio segments")
        return audio_files
    
    def combine_audio_files(self, audio_files, file_stem):
        """Combine all audio segments into one file"""
        combined_file = self.config.generated_content / f"combined_{file_stem}.mp3"
        
        if self.audio_processor.combine_audio_segments(audio_files, combined_file):
            self.log_message(f"COMBINED: Audio saved to {combined_file.name}")
            return str(combined_file)
        else:
            self.log_message("ERROR: Failed to combine audio segments")
            return None
    
    def apply_voice_cloning(self, audio_file, voice_file, file_stem):
        """Apply voice cloning to the combined audio"""
        final_file = self.config.generated_content / f"final_{file_stem}_cloned.mp3"
        
        if self.audio_processor.apply_voice_cloning(Path(audio_file), voice_file, final_file):
            self.log_message(f"CLONED: Final audiobook saved to {final_file.name}")
            return str(final_file)
        else:
            self.log_message("WARNING: Voice cloning failed, returning original TTS")
            return audio_file
    
    def save_log_file(self, log_filename):
        """Save processing log to file"""
        try:
            with open(log_filename, 'w', encoding='utf-8') as f:
                f.write("AudioBook Creator - Processing Log\n")
                f.write("=" * 60 + "\n")
                for message in self.processing_log:
                    f.write(f"{message}\n")
            
            self.log_message(f"LOG: Saved to {log_filename}")
        except Exception as e:
            self.log_message(f"ERROR: Failed to save log: {e}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function - runs the audiobook generation"""
    
    print("AudioBook Creator - Simple Test Script")
    print("=" * 60)
    print(f"PDF File: {PDF_FILE_PATH}")
    print(f"Voice File: {VOICE_FILE_PATH}")
    print(f"Character Limit: {MAX_CHARACTERS if MAX_CHARACTERS else 'No limit'}")
    print(f"Segment Limit: {MAX_SEGMENTS if MAX_SEGMENTS else 'No limit'}")
    print("=" * 60)
    
    
    if not os.getenv("DEEPGRAM_API_KEY"):
        print("ERROR: DEEPGRAM_API_KEY environment variable not found")
        print("Please create a .env file with your Deepgram API key")
        return
    
    
    processor = SimpleAudioBookTester()
    
    result = processor.process_audiobook(
        pdf_path=PDF_FILE_PATH,
        voice_path=VOICE_FILE_PATH,
        max_chars=MAX_CHARACTERS,
        max_segments=MAX_SEGMENTS
    )
    
    
    if SAVE_LOG:
        processor.save_log_file(LOG_FILE)
    
    
    if result:
        print(f"\nSUCCESS! Audiobook saved to: {result}")
        print(f"Processing log saved to: {LOG_FILE}")
    else:
        print("\nFAILED! Check the log for details.")

if __name__ == "__main__":
    main()
