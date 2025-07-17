import uuid
from typing import List, Optional, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field
from bson import ObjectId
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import core_schema
from pydantic import GetCoreSchemaHandler, GetJsonSchemaHandler
class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)

    @classmethod
    def __get_pydantic_core_schema__(cls, _source_type: Any, _handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        return core_schema.no_info_plain_validator_function(cls.validate)

    @classmethod
    def __get_pydantic_json_schema__(
        cls, core_schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        schema = handler(core_schema)
        schema.update(type="string")
        return schema
class Page(BaseModel):
    page_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    page_title: str
    content: str
    file_paths: List[str] = Field(default_factory=list)

class Section(BaseModel):
    section_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    section_title: str
    pages: List[Page]
    subsections: List["Section"] = Field(default_factory=list)

class Structure(BaseModel):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    title: str
    description : str
    root_sections: List[Section]
    repo_url: str
    owner: str
    repo_name: str
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    status: str = "completed"

    model_config = {
        "populate_by_name": True,
        "arbitrary_types_allowed": True,
        "json_encoders": {ObjectId: str},
    }

        
def parse_structure_from_xml(xml_string: str) -> Structure:
    import xml.etree.ElementTree as ET
    
    root = ET.fromstring(xml_string)
    title_el = root.find("title")
    description_el = root.find("description")

    title = title_el.text.strip() if title_el is not None and title_el.text else ""
    description = description_el.text.strip() if description_el is not None and description_el.text else ""
    
    # Parse pages
    page_map = {}
    page_count = 0
    page_elements = root.findall(".//page")
    
    for page_el in page_elements:
        page_id = page_el.get("id") or f"page-{page_count + 1}"
        page_title = page_el.find("title").text.strip() if page_el.find("title") is not None else ""
        file_path_els = page_el.findall("relavant_files/file_path")
        file_paths = [fp.text.strip() for fp in file_path_els if fp.text]
        
        page_count += 1
        
        page = Page(
            page_title=page_title,
            content="",  # to be filled/generated later
            file_paths=file_paths,
            page_id=page_id
        )
        
        page_map[page_id] = page
        
    # Parse sections
    section_map = {}
    section_count = 0
    referenced_section_ids = set()
    section_elements = root.findall(".//section")
    
    for section_el in section_elements:
        section_id = section_el.get("id") or f"section-{section_count + 1}"
        section_title = section_el.find("title").text.strip() if section_el.find("title") is not None else ""
        
        # Pages in section
        page_ids = [el.text.strip() for el in section_el.findall("pages/page_ref") if el.text]
        pages = [page_map[pid] for pid in page_ids if pid in page_map]
        
        # Subsections (only store IDs, will resolve later)
        subsection_ids = [el.text.strip() for el in section_el.findall("subsections/section_ref") if el.text]
        for sid in subsection_ids:
            referenced_section_ids.add(sid)
            
        section = Section(
            section_title=section_title,
            pages=pages,
            subsections=[],
            section_id=section_id
        )

        section_map[section_id] = {
            "section": section,
            "subsection_ids": subsection_ids
        }
        
    # Resolve subsections
    for section_id, info in section_map.items():
        section = info["section"]
        subsection_ids = info["subsection_ids"]
        section.subsections = [section_map[sid]["section"] for sid in subsection_ids if sid in section_map]
            
    # Root sections
    all_section_ids = set(section_map.keys())
    root_section_ids = all_section_ids - referenced_section_ids
    root_sections = [section_map[sid]["section"] for sid in root_section_ids]
    
    return Structure(
        title=title,
        description=description,
        root_sections=root_sections,
        repo_url="",  # Placeholder, can be set later
        owner="",  # Placeholder, can be set later
        repo_name="",  # Placeholder, can be set later
        created_at=datetime.now(),
        updated_at=datetime.now(),
        status="completed"  # Default status
    )
    
def get_pages_from_structure(structure: Structure) -> List[Page]:
    """
    Recursively collects all pages from the structure.
    """
    pages = []
    
    def collect_pages(section: Section):
        pages.extend(section.pages)
        for subsection in section.subsections:
            collect_pages(subsection)
    
    for root_section in structure.root_sections:
        collect_pages(root_section)
    
    return pages