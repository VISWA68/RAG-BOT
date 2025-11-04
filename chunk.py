"""
RAPTEE T30 RAG System - Document Chunking & ChromaDB Storage
Intelligent semantic chunking with metadata enrichment
"""

import re
from typing import List, Dict, Tuple
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

class RapteeDocumentChunker:
    """
    Intelligent chunking strategy for technical motorcycle documentation
    """
    
    def __init__(self, chunk_size: int = 400, overlap: int = 80):
        """
        Args:
            chunk_size: Target characters per chunk (not strict)
            overlap: Overlap characters between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        
        # Section definitions for semantic chunking
        self.sections = {
            "technology": ["HV-TEC", "High Voltage", "battery", "cooling", "lightweight"],
            "performance": ["Power", "Torque", "Speed", "Acceleration", "Riding Modes"],
            "range_charging": ["Range", "Charging", "CCS2", "Battery Capacity", "kWh"],
            "powertrain": ["Motor", "IPMSM", "Throttle", "Frame", "Suspension", "Kerb Weight"],
            "brakes_wheels": ["Braking", "ABS", "Disc", "Transmission", "Belt drive", "Tyres"],
            "dashboard_features": ["RAPTEE OS", "Screen", "Bluetooth", "OTA", "LED", "ParkZ"],
            "app_services": ["Raptee App", "Navigation", "Maps", "Insights", "Diagnosis", "Service"],
            "warranty": ["Warranty", "Colors", "Arctic White", "Eclipse Black"]
        }
    
    def extract_semantic_chunks(self, document: str) -> List[Dict[str, any]]:
        """
        Extract chunks based on semantic sections with metadata
        """
        chunks = []
        
        # Split by major sections (using your document structure)
        sections_text = {
            "Core Technology: HV-TEC Powertrain": self._extract_section(document, "Core Technology", "Performance and Riding"),
            "Performance and Riding Specifications": self._extract_section(document, "Performance and Riding", "Range and Charging"),
            "Range and Charging": self._extract_section(document, "Range and Charging", "Powertrain and Chassis"),
            "Powertrain and Chassis Details": self._extract_section(document, "Powertrain and Chassis", "Brakes, Wheels"),
            "Brakes, Wheels, and Transmission": self._extract_section(document, "Brakes, Wheels", "Dashboard and Peripheral"),
            "Dashboard and Peripheral Features": self._extract_section(document, "Dashboard and Peripheral", "App and Services"),
            "App and Services": self._extract_section(document, "App and Services", "Warranty and Colors"),
            "Warranty and Colors": self._extract_section(document, "Warranty and Colors", None)
        }
        
        chunk_id = 0
        for section_name, section_text in sections_text.items():
            if not section_text or len(section_text.strip()) < 50:
                continue
            
            # Determine category
            category = self._categorize_section(section_name)
            
            # Extract key specs from section
            specs = self._extract_specs(section_text)
            
            # Create sub-chunks if section is too long
            if len(section_text) > self.chunk_size:
                sub_chunks = self._split_with_overlap(section_text)
                for idx, sub_chunk in enumerate(sub_chunks):
                    chunks.append({
                        "id": f"chunk_{chunk_id}",
                        "text": f"{section_name}\n\n{sub_chunk}",
                        "metadata": {
                            "section": section_name,
                            "category": category,
                            "sub_chunk": idx + 1,
                            "total_sub_chunks": len(sub_chunks),
                            "specs": specs,  # Now a string instead of list
                            "char_count": len(sub_chunk)
                        }
                    })
                    chunk_id += 1
            else:
                chunks.append({
                    "id": f"chunk_{chunk_id}",
                    "text": f"{section_name}\n\n{section_text}",
                    "metadata": {
                        "section": section_name,
                        "category": category,
                        "specs": specs,  # Now a string instead of list
                        "char_count": len(section_text)
                    }
                })
                chunk_id += 1
        
        return chunks
    
    def _extract_section(self, document: str, start_marker: str, end_marker: str) -> str:
        """Extract text between two section markers"""
        start_idx = document.find(start_marker)
        if start_idx == -1:
            return ""
        
        if end_marker:
            end_idx = document.find(end_marker, start_idx)
            if end_idx == -1:
                return document[start_idx:]
            return document[start_idx:end_idx]
        else:
            return document[start_idx:]
    
    def _split_with_overlap(self, text: str) -> List[str]:
        """Split text with overlapping windows"""
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            
            # Try to break at sentence boundary
            if end < text_len:
                last_period = text.rfind('.', start, end)
                last_newline = text.rfind('\n', start, end)
                break_point = max(last_period, last_newline)
                
                if break_point > start + self.chunk_size // 2:
                    end = break_point + 1
            
            chunks.append(text[start:end].strip())
            start = end - self.overlap if end < text_len else text_len
        
        return chunks
    
    def _categorize_section(self, section_name: str) -> str:
        """Categorize section based on keywords"""
        section_lower = section_name.lower()
        
        for category, keywords in self.sections.items():
            if any(keyword.lower() in section_lower for keyword in keywords):
                return category
        
        return "general"
    
    def _extract_specs(self, text: str) -> str:
        """Extract key specifications and return as string instead of list"""
        specs = []
        
        # Extract patterns like "Power: 22 kW", "Range: 200 Km", etc.
        spec_pattern = r'([A-Z][A-Za-z\s]+):\s*([0-9.,]+\s*[A-Za-z/%]+)'
        matches = re.findall(spec_pattern, text)
        
        for match in matches:
            specs.append(f"{match[0].strip()}: {match[1].strip()}")
        
        # Join specs into a single string
        return "; ".join(specs[:5])  # Top 5 specs as a single string


class ChromaDBManager:
    """
    Manages ChromaDB collection for RAPTEE documentation
    """
    
    def __init__(self, collection_name: str = "raptee_t30_docs", persist_directory: str = "./chroma_db"):
        """Initialize ChromaDB client and collection"""
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Use sentence transformer for embeddings
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Create or get collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            print(f"✅ Loaded existing collection: {collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "RAPTEE T30 motorcycle documentation"}
            )
            print(f"✅ Created new collection: {collection_name}")
    
    def add_chunks(self, chunks: List[Dict[str, any]]):
        """Add document chunks to ChromaDB"""
        ids = [chunk["id"] for chunk in chunks]
        documents = [chunk["text"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(documents).tolist()
        
        # Add to collection
        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )
        
        print(f"✅ Added {len(chunks)} chunks to ChromaDB")
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection"""
        count = self.collection.count()
        return {
            "total_chunks": count,
            "collection_name": self.collection.name
        }


# Main execution function
def process_raptee_document(document_path: str, persist_directory: str = "./chroma_db"):
    """
    Complete pipeline: Load document -> Chunk -> Store in ChromaDB
    """
    
    # Read document
    with open(document_path, 'r', encoding='utf-8') as f:
        document = f.read()
    
    print(f"📄 Loaded document: {len(document)} characters")
    
    # Initialize chunker
    chunker = RapteeDocumentChunker(chunk_size=400, overlap=80)
    
    # Extract chunks
    chunks = chunker.extract_semantic_chunks(document)
    print(f"✂️  Created {len(chunks)} semantic chunks")
    
    # Print sample chunk for verification
    if chunks:
        print("\n📋 Sample chunk:")
        
        print(f"ID: {chunks[0]['id']}")
        print(f"Category: {chunks[0]['metadata']['category']}")
        print(f"Text preview: {chunks[0]['text'][:200]}...")
    
    # Initialize ChromaDB
    db_manager = ChromaDBManager(persist_directory=persist_directory)
    
    # Add chunks to database
    db_manager.add_chunks(chunks)
    
    # Print stats
    stats = db_manager.get_collection_stats()
    print(f"\n📊 Database Stats:")
    print(f"   Total chunks: {stats['total_chunks']}")
    print(f"   Collection: {stats['collection_name']}")
    
    return db_manager, chunks


# Example usage
if __name__ == "__main__":
    # Process the RAPTEE document
    db_manager, chunks = process_raptee_document(
        document_path="knowledge_base.txt",
        persist_directory="./chroma_db"
    )
    
    print("\n✅ Document processing complete!")
    print("🔍 Ready for retrieval queries")