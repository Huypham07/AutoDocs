from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class GraphDatabaseConfig:
    """Configuration for graph database."""
    # Neo4j configuration
    neo4j_uri: str = 'neo4j://localhost:7687'
    neo4j_username: str = 'neo4j'
    neo4j_password: str = 'password'
    neo4j_database: str = 'neo4j'

    @classmethod
    def from_env(cls) -> GraphDatabaseConfig:
        """Create configuration from environment variables."""
        return cls(
            # Neo4j settings
            neo4j_uri=os.getenv('NEO4J_URI', 'neo4j://localhost:7687'),
            neo4j_username=os.getenv('NEO4J_USERNAME', 'neo4j'),
            neo4j_password=os.getenv('NEO4J_PASSWORD', 'password'),
            neo4j_database=os.getenv('NEO4J_DATABASE', 'neo4j'),
        )

    def get_neo4j_config(self) -> dict:
        """Get Neo4j configuration."""
        return {
            'uri': self.neo4j_uri,
            'username': self.neo4j_username,
            'password': self.neo4j_password,
            'database': self.neo4j_database,
        }


# Global configuration instance
graph_config = GraphDatabaseConfig.from_env()
