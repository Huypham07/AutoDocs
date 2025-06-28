from adalflow.core.types import Document, List
from app.db.repo_db_manager import DBManager

class DBPreparation:
    def __init__(self):
        self.db_manager = None

    def prepare_db(self, repo_url: str, access_token: str = None)-> List[Document]:
        self.db_manager = DBManager()
        return self.db_manager.prepare_db(repo_url=repo_url, access_token=access_token)
     