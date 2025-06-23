import os
import asyncio
import aiohttp
import magic
import pytesseract
from PIL import Image
from io import BytesIO
from typing import List, Dict, Any
from unstructured.partition.auto import partition
from dotenv import load_dotenv
from .logger import get_logger
from .utils import clean_text

# Load environment variables
load_dotenv()

logger = get_logger(__name__)

class FileProcessor:
    """Handles downloading and processing of file attachments."""
    
    def __init__(self):
        self.tesseract_languages = os.getenv("TESSERACT_LANGUAGES", "eng")
        self.session = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=60)
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
            # Open image with PIL
            image = Image.open(BytesIO(content))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Extract text using OCR
            extracted_text = pytesseract.image_to_string(
                image, 
                lang=self.tesseract_languages
            )
            
            cleaned_text = clean_text(extracted_text)
            logger.debug(f"Extracted {len(cleaned_text)} characters from image {url}")
            
            return cleaned_text
            
        except Exception as e:
            logger.error(f"OCR error for {url}: {e}")
            self._log_to_dead_letter_queue(url, str(e), "ocr")
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
            # Use unstructured to parse document
            # Save to temporary file for processing
            import tempfile
            
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(content)
                temp_file_path = temp_file.name
            
            try:
                # Partition document
                elements = partition(temp_file_path)
                
                # Extract text from elements
                extracted_text = ""
                for element in elements:
                    if hasattr(element, 'text') and element.text:
                        extracted_text += element.text + "\n"
                
                cleaned_text = clean_text(extracted_text)
                logger.debug(f"Extracted {len(cleaned_text)} characters from document {url}")
                
                return cleaned_text
                
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_file_path)
                except:
                    pass
                    
        except Exception as e:
            logger.error(f"Document parsing error for {url}: {e}")
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