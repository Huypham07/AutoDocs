import uuid
from typing import List, Optional

class Page:
    def __init__(self, page_title: str, content: str, file_paths: List[str], page_id: str = None):
        self.page_id = page_id or str(uuid.uuid4())
        self.page_title = page_title
        self.content = content
        self.file_paths = file_paths

    def to_dict(self):
        return {
            "id": self.id,
            "title": self.title,
            "content": self.content,
            "file_paths": self.file_paths
        }


class Section:
    def __init__(self, section_title: str, pages: List[Page], subsections: Optional[List["Section"]] = None, section_id: str = None):
        self.id = section_id or str(uuid.uuid4())
        self.section_title = section_title
        self.pages = pages
        self.subsections = subsections or []

    def to_dict(self):
        return {
            "title": self.title,
            "pages": [page.to_dict() for page in self.pages],
            "subsections": [section.to_dict() for section in self.subsections]
        }


class Structure:
    def __init__(self, title: str, description: str, root_sections: List[Section], structure_id: str = None):
        self.id = structure_id or str(uuid.uuid4())
        self.title = title
        self.description = description
        self.root_sections = root_sections

    def to_dict(self):
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "root_sections": [section.to_dict() for section in self.root_sections]
        }

    def validate(self):
        if not self.title or not isinstance(self.title, str):
            raise ValueError("Title must be a non-empty string")
        if not isinstance(self.sections, list):
            raise ValueError("Sections must be a list")
        # Validate each section recursively
        for section in self.root_sections:
            if not isinstance(section, Section):
                raise ValueError("Invalid section type")

        
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
        root_sections=root_sections
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