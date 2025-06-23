import os
import sys
import asyncio
import aiohttp
import magic
import pytesseract
from PIL import Image
from io import BytesIO
from typing import List, Dict, Any
from dotenv import load_dotenv
from .logger import get_logger
from .utils import clean_text

# Load environment variables
load_dotenv()

logger = get_logger(__name__)

# Force load unstructured with proper error handling for multiprocessing
def _ensure_unstructured_available():
    """Ensure unstructured dependencies are properly loaded in child processes."""
    try:
        # Force import with specific error handling
        from unstructured.partition.auto import partition
        from unstructured.partition.docx import partition_docx
        from unstructured.partition.pdf import partition_pdf
        logger.info("Successfully loaded unstructured dependencies")
        return True
    except ImportError as e:
        logger.error(f"Failed to import unstructured dependencies: {e}")
        
        # Try to diagnose the issue
        try:
            import unstructured
            logger.info(f"unstructured version: {unstructured.__version__}")
        except:
            logger.error("unstructured package not found")
        
        # Check if specific modules are available
        try:
            from unstructured.partition import docx
            logger.info("docx partition module found")
        except ImportError as docx_error:
            logger.error(f"docx partition module missing: {docx_error}")
        
        try:
            from unstructured.partition import pdf  
            logger.info("pdf partition module found")
        except ImportError as pdf_error:
            logger.error(f"pdf partition module missing: {pdf_error}")
            
        return False

# Test dependency availability at module load
_UNSTRUCTURED_AVAILABLE = _ensure_unstructured_available()

def _check_tesseract_available():
    """Check if Tesseract OCR is properly installed and available."""
    try:
        import pytesseract
        # Try to get Tesseract version to verify it's working
        version = pytesseract.get_tesseract_version()
        logger.info(f"Tesseract OCR available, version: {version}")
        return True
    except Exception as e:
        logger.error(f"Tesseract OCR not available: {e}")
        logger.error("Install Tesseract: brew install tesseract (macOS) or apt-get install tesseract-ocr (Ubuntu)")
        return False

# Test OCR availability at module load
_TESSERACT_AVAILABLE = _check_tesseract_available()

class FileProcessor:
    """Handles downloading and processing of file attachments."""
    
    def __init__(self):
        self.tesseract_languages = os.getenv("TESSERACT_LANGUAGES", "eng")
        self.session = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=120)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def process_attachments(self, attachment_urls: List[str]) -> List[str]:
        """
        Process multiple attachment URLs and extract text content.
        
        Args:
            attachment_urls: List of attachment URLs to process
            
        Returns:
            List of extracted text content
        """
        if not attachment_urls:
            return []
        
        logger.info(f"Processing {len(attachment_urls)} attachments")
        
        # Process attachments concurrently
        tasks = [self._process_single_attachment(url) for url in attachment_urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and empty results
        extracted_texts = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to process attachment {attachment_urls[i]}: {result}")
                self._log_to_dead_letter_queue(attachment_urls[i], str(result), "processing")
            elif result:
                extracted_texts.append(result)
        
        logger.info(f"Successfully processed {len(extracted_texts)} attachments")
        return extracted_texts
    
    async def _process_single_attachment(self, url: str) -> str:
        """
        Process a single attachment URL.
        
        Args:
            url: Attachment URL to process
            
        Returns:
            Extracted text content
        """
        try:
            # Download file
            file_content = await self._download_file(url)
            if not file_content:
                return ""
            
            # Determine MIME type
            mime_type = self._get_mime_type(file_content)
            logger.debug(f"Detected MIME type for {url}: {mime_type}")
            
            # Process based on MIME type
            if mime_type.startswith('image/'):
                return await self._process_image(file_content, url)
            elif mime_type in ['application/pdf', 'text/plain', 'application/msword',
                             'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                             'application/vnd.ms-powerpoint',
                             'application/vnd.openxmlformats-officedocument.presentationml.presentation']:
                return await self._process_document(file_content, url)
            else:
                logger.warning(f"Unsupported MIME type {mime_type} for {url}")
                return ""
                
        except Exception as e:
            logger.error(f"Error processing attachment {url}: {e}")
            self._log_to_dead_letter_queue(url, str(e), "single_processing")
            return ""
    
    async def _download_file(self, url: str) -> bytes:
        """
        Download file from URL.
        
        Args:
            url: File URL to download
            
        Returns:
            File content as bytes
        """
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    content = await response.read()
                    logger.debug(f"Downloaded {len(content)} bytes from {url}")
                    return content
                else:
                    logger.error(f"Failed to download {url}: HTTP {response.status}")
                    return b""
                    
        except Exception as e:
            logger.error(f"Download error for {url}: {e}")
            self._log_to_dead_letter_queue(url, str(e), "download")
            return b""
    
    def _get_mime_type(self, content: bytes) -> str:
        """
        Determine MIME type of file content.
        
        Args:
            content: File content as bytes
            
        Returns:
            MIME type string
        """
        try:
            return magic.from_buffer(content, mime=True)
        except Exception as e:
            logger.error(f"Failed to determine MIME type: {e}")
            return "application/octet-stream"
    
    async def _process_image(self, content: bytes, url: str) -> str:
        """
        Process image file using OCR.
        
        Args:
            content: Image content as bytes
            url: Original URL for logging
            
        Returns:
            Extracted text from image
        """
        try:
            # Check if Tesseract is available
            global _TESSERACT_AVAILABLE
            if not _TESSERACT_AVAILABLE:
                logger.warning("Tesseract OCR not available, retesting...")
                _TESSERACT_AVAILABLE = _check_tesseract_available()
                
                if not _TESSERACT_AVAILABLE:
                    error_msg = "Tesseract OCR not available for image processing"
                    logger.error(error_msg)
                    self._log_to_dead_letter_queue(url, error_msg, "tesseract_unavailable")
                    return ""
            
            # Validate image size (50MB limit)
            if len(content) > 50 * 1024 * 1024:
                logger.warning(f"Image too large ({len(content)} bytes): {url}")
                self._log_to_dead_letter_queue(url, "Image exceeds 50MB limit", "size_check")
                return ""
            
            # Validate minimum size (avoid tiny images)
            if len(content) < 100:
                logger.warning(f"Image too small ({len(content)} bytes): {url}")
                return ""
            
            # Open image with PIL with robust error handling
            try:
                image = Image.open(BytesIO(content))
                logger.debug(f"Successfully opened image: {image.size} pixels, mode: {image.mode}")
            except Exception as e:
                logger.error(f"Failed to open image {url}: {e}")
                self._log_to_dead_letter_queue(url, f"PIL error: {str(e)}", "pil_open")
                return ""
            
            # Validate image dimensions
            width, height = image.size
            if width < 10 or height < 10:
                logger.warning(f"Image dimensions too small ({width}x{height}): {url}")
                return ""
            
            if width > 10000 or height > 10000:
                logger.warning(f"Image dimensions too large ({width}x{height}), resizing: {url}")
                # Resize large images to prevent memory issues
                max_dimension = 4000
                if width > height:
                    new_width = max_dimension
                    new_height = int(height * max_dimension / width)
                else:
                    new_height = max_dimension
                    new_width = int(width * max_dimension / height)
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                logger.info(f"Resized image to {new_width}x{new_height}")
            
            # Convert to RGB if necessary (ensure compatibility with Tesseract)
            if image.mode not in ['RGB', 'L']:
                try:
                    logger.debug(f"Converting image from {image.mode} to RGB")
                    image = image.convert('RGB')
                except Exception as e:
                    logger.error(f"Failed to convert image to RGB {url}: {e}")
                    self._log_to_dead_letter_queue(url, f"RGB conversion error: {str(e)}", "rgb_conversion")
                    return ""
            
            # Enhance image quality for better OCR results
            try:
                from PIL import ImageEnhance, ImageFilter
                
                # Enhance contrast and sharpness for better OCR
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(1.2)  # Slight contrast boost
                
                enhancer = ImageEnhance.Sharpness(image)
                image = enhancer.enhance(1.1)  # Slight sharpness boost
                
                logger.debug("Applied image enhancements for better OCR")
            except Exception as enhance_error:
                logger.debug(f"Image enhancement failed, proceeding with original: {enhance_error}")
                # Continue with original image if enhancement fails
            
            # Extract text using OCR with multiple attempts
            extracted_text = ""
            tesseract_languages = self.tesseract_languages
            
            # Try with specified languages first
            try:
                logger.debug(f"Running Tesseract OCR with languages: {tesseract_languages}")
                extracted_text = pytesseract.image_to_string(
                    image, 
                    lang=tesseract_languages,
                    config='--psm 6 --oem 3'  # Assume uniform block of text, use LSTM OCR engine
                )
                logger.debug(f"Tesseract extracted {len(extracted_text)} characters")
            except Exception as e:
                logger.warning(f"Tesseract OCR failed with specified languages ({tesseract_languages}): {e}")
                
                # Fallback to English only
                try:
                    logger.debug("Retrying OCR with English only")
                    extracted_text = pytesseract.image_to_string(
                        image, 
                        lang='eng',
                        config='--psm 6 --oem 3'
                    )
                    logger.debug(f"Fallback OCR extracted {len(extracted_text)} characters")
                except Exception as fallback_error:
                    logger.error(f"All OCR attempts failed for {url}: {fallback_error}")
                    self._log_to_dead_letter_queue(url, f"Tesseract error: {str(fallback_error)}", "tesseract_ocr")
                    return ""
            
            # Clean and validate extracted text
            if not extracted_text or not extracted_text.strip():
                logger.info(f"No text detected in image: {url}")
                return ""
            
            cleaned_text = clean_text(extracted_text)
            
            # Filter out very short or nonsensical extractions
            if len(cleaned_text.strip()) < 3:
                logger.debug(f"Extracted text too short, likely noise: {url}")
                return ""
            
            logger.info(f"Successfully extracted {len(cleaned_text)} characters from image {url}")
            return cleaned_text
            
        except Exception as e:
            error_msg = f"Unexpected OCR error for {url}: {e}"
            logger.error(error_msg)
            self._log_to_dead_letter_queue(url, f"Unexpected error: {str(e)}", "ocr_general")
            return ""
    
    async def _process_document(self, content: bytes, url: str) -> str:
        """
        Process document file using unstructured.
        
        Args:
            content: Document content as bytes
            url: Original URL for logging
            
        Returns:
            Extracted text from document
        """
        try:
            # Ensure dependencies are available in this process
            global _UNSTRUCTURED_AVAILABLE
            if not _UNSTRUCTURED_AVAILABLE:
                logger.warning("Attempting to reload unstructured dependencies...")
                _UNSTRUCTURED_AVAILABLE = _ensure_unstructured_available()
                
                if not _UNSTRUCTURED_AVAILABLE:
                    # Try installing dependencies dynamically
                    logger.info("Attempting dynamic dependency installation...")
                    try:
                        import subprocess
                        import sys
                        result = subprocess.run([
                            sys.executable, "-m", "pip", "install", 
                            "unstructured[docx,pdf]==0.10.30", 
                            "pdfminer.six==20221105",
                            "--force-reinstall", "--no-cache-dir"
                        ], capture_output=True, text=True, timeout=300)
                        
                        if result.returncode == 0:
                            logger.info("Dynamic installation successful, reloading dependencies...")
                            _UNSTRUCTURED_AVAILABLE = _ensure_unstructured_available()
                        else:
                            logger.error(f"Dynamic installation failed: {result.stderr}")
                    except Exception as install_error:
                        logger.error(f"Failed to install dependencies dynamically: {install_error}")
            
            if not _UNSTRUCTURED_AVAILABLE:
                error_msg = "unstructured dependencies not available in this process"
                logger.error(error_msg)
                self._log_to_dead_letter_queue(url, error_msg, "dependency_check")
                return ""
            
            # Import partition function (should work now)
            from unstructured.partition.auto import partition
            
            # Validate document size (50MB limit)
            if len(content) > 50 * 1024 * 1024:
                logger.warning(f"Document too large ({len(content)} bytes): {url}")
                self._log_to_dead_letter_queue(url, "Document exceeds 50MB limit", "size_check")
                return ""
            
            # Use unstructured to parse document
            import tempfile
            
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(content)
                temp_file_path = temp_file.name
            
            try:
                logger.debug(f"Processing document with unstructured: {url}")
                
                # Partition document with error handling
                try:
                    elements = partition(temp_file_path)
                    logger.debug(f"Successfully partitioned document with {len(elements)} elements")
                except Exception as partition_error:
                    logger.error(f"Partition failed for {url}: {partition_error}")
                    
                    # Try specific partition methods based on file extension
                    if url.lower().endswith('.docx'):
                        try:
                            from unstructured.partition.docx import partition_docx
                            elements = partition_docx(temp_file_path)
                            logger.debug("Successfully used docx-specific partition")
                        except Exception as docx_error:
                            logger.error(f"DOCX partition failed: {docx_error}")
                            raise partition_error
                    elif url.lower().endswith('.pdf'):
                        try:
                            from unstructured.partition.pdf import partition_pdf
                            elements = partition_pdf(temp_file_path)
                            logger.debug("Successfully used pdf-specific partition")
                        except Exception as pdf_error:
                            logger.error(f"PDF partition failed: {pdf_error}")
                            raise partition_error
                    else:
                        raise partition_error
                
                # Extract text from elements
                extracted_text = ""
                for element in elements:
                    if hasattr(element, 'text') and element.text:
                        extracted_text += element.text + "\n"
                
                if not extracted_text.strip():
                    logger.warning(f"No text extracted from document: {url}")
                    return ""
                
                cleaned_text = clean_text(extracted_text)
                logger.info(f"Successfully extracted {len(cleaned_text)} characters from document {url}")
                
                return cleaned_text
                
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_file_path)
                except:
                    pass
                    
        except Exception as e:
            error_msg = f"Document parsing error for {url}: {e}"
            logger.error(error_msg)
            self._log_to_dead_letter_queue(url, str(e), "document_parsing")
            return ""
    
    def _log_to_dead_letter_queue(self, url: str, error: str, step: str):
        """
        Log failed processing attempts to dead letter queue.
        
        Args:
            url: Failed URL
            error: Error message
            step: Processing step that failed
        """
        try:
            import json
            from datetime import datetime
            
            dead_letter_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "url": url,
                "error": error,
                "step": step
            }
            
            # Append to dead letter log file
            with open("dead_letter_queue.json", "a") as f:
                f.write(json.dumps(dead_letter_entry) + "\n")
                
        except Exception as e:
            logger.error(f"Failed to write to dead letter queue: {e}")

# Convenience function for processing attachments
async def process_attachments(attachment_urls: List[str]) -> List[str]:
    """
    Process attachment URLs and return extracted text.
    
    Args:
        attachment_urls: List of attachment URLs
        
    Returns:
        List of extracted text content
    """
    async with FileProcessor() as processor:
        return await processor.process_attachments(attachment_urls) 