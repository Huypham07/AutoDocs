from app.db.database import get_database
from app.models.structure import Structure, Section
from datetime import datetime
import logging
from typing import Optional, List
from pydantic import Field

logger = logging.getLogger(__name__)

class DocumentationService:
    def __init__(self):
        self.db = get_database()
        self.collection = self.db.documentations
    
    async def save_documentation(
        self,
        title: str,
        description: str,
        root_sections: List[Section],
        repo_url: str, 
        owner: str, 
        repo_name: str,
        created_at: datetime = Field(default_factory=datetime.now),
        updated_at: datetime = Field(default_factory=datetime.now),
    ) -> Optional[str]:
        try:
            doc_data = Structure(
                title=title,
                description=description,
                root_sections=root_sections,
                repo_url=repo_url,
                owner=owner,
                repo_name=repo_name,
                created_at=created_at,
                updated_at=updated_at,
                status="completed"  # Default status
            )
            
            # Convert to dict for MongoDB insertion
            doc_dict = doc_data.dict(by_alias=True, exclude_unset=True)
            
            result = await self.collection.insert_one(doc_dict)
            
            logger.info(f"Documentation saved successfully with ID: {result.inserted_id}")
            return str(result.inserted_id)
            
        except Exception as e:
            logger.error(f"Error saving documentation to MongoDB: {str(e)}")
            return None
    
    async def get_documentation_by_repo(self, owner: str, repo_name: str) -> Optional[dict]:
        try:
            result = await self.collection.find_one({"owner": owner, "repo_name": repo_name})
            if result:
                result["_id"] = str(result["_id"])
            return result
        except Exception as e:
            logger.error(f"Error getting documentation from MongoDB: {str(e)}")
            return None
    
    async def get_documentation_by_id(self, doc_id: str) -> Optional[dict]:
        try:
            from bson import ObjectId
            result = await self.collection.find_one({"_id": ObjectId(doc_id)})
            if result:
                result["_id"] = str(result["_id"])
            return result
        except Exception as e:
            logger.error(f"Error getting documentation by ID from MongoDB: {str(e)}")
            return None
    
    async def update_documentation(self, structure: Structure) -> bool:
        try:
            from bson import ObjectId
            update_data = structure.dict(by_alias=True, exclude_unset=True)
            
            result = await self.collection.update_one(
                {"_id": ObjectId(structure.id)},
                {"$set": update_data}
            )
            
            if result.modified_count > 0:
                logger.info(f"Documentation updated successfully: {structure.id}")
                return True
            else:
                logger.warning(f"No document found to update: {structure.id}")
                return False
                
        except Exception as e:
            logger.error(f"Error updating documentation: {str(e)}")
            return False
    
    async def delete_documentation(self, doc_id: str) -> bool:
        try:
            from bson import ObjectId
            result = await self.collection.delete_one({"_id": ObjectId(doc_id)})
            
            if result.deleted_count > 0:
                logger.info(f"Documentation deleted successfully: {doc_id}")
                return True
            else:
                logger.warning(f"No document found to delete: {doc_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting documentation: {str(e)}")
            return False
