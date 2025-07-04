from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from app.schemas.task import TaskRequest
from app.services.docsgen.util import extract_repo_info
from app.services.docsgen.prepare_db import DBPreparation
from app.core.rag import RAG
import logging
from fastapi import HTTPException
from app.core.logging import setup_logging
from app.services.docsgen.structure_generator import StructureGenerator
from app.services.docsgen.content_generator import ContentGenerator
import asyncio
import re


setup_logging()
logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/generate/docs")
async def create_docs(request: TaskRequest):
    """Generate documentation for the given repository URL."""
    try:
        repo_url = str(request.repo_url)
        access_token = request.access_token
        repo_info = extract_repo_info(str(request.repo_url))
        owner = repo_info.get("owner")
        repo_name = repo_info.get("repo_name")
        
        
        transformed_docs = DBPreparation().prepare_db(repo_url=repo_url, access_token=access_token)
        
        try:
            rag = RAG()
            rag.prepare_retriever(transformed_docs)
        except Exception as e:
            logger.error(f"Error preparing RAG retriever: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error preparing RAG retriever: {str(e)}")
        
        logger.info("Starting documentation generation ...")

        async def generate_full_docs():
            buffer = []
            structure_gen = StructureGenerator(rag=rag, repo_url=repo_url, access_token=access_token, owner=owner, repo_name=repo_name)
            async for chunk in structure_gen():
                buffer.append(chunk)
                yield chunk
                
            xml_string = "".join(buffer)

            # Extract structure from XML string
            xml_match = re.search(r'<documentation_structure>.*?</documentation_structure>', xml_string, re.DOTALL)
            if not xml_match:
                yield "\nError: No documentation structure found in the XML response. Try again later."
                return
            
            content_gen = ContentGenerator(rag=rag, xml_string=xml_match.group(0), repo_url=repo_url, access_token=access_token, owner=owner, repo_name=repo_name)
            async for chunk in content_gen():
                yield chunk
        return StreamingResponse(generate_full_docs(), media_type="text/event-stream")
    except Exception:
        raise
    
