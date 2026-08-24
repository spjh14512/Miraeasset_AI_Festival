from agent_graph.tools import CypherQuery
from agent_graph.state import RetrievalResult
from neo4j import Driver, GraphDatabase

from dotenv import load_dotenv
import os


cypher_query = CypherQuery(

    cypher= """
    MATCH (n:Section) RETURN n LIMIT 25;
        """,

    parameters= {
         "corp_name": "삼성전자" 
         }
)

load_dotenv()
uri = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")

driver = GraphDatabase.driver(uri, auth=(username, password))

with driver.session() as session:
    result = session.run(
        cypher_query.cypher,
        parameters=cypher_query.parameters,
    )
    records = [record.data() for record in result]

print(records)
driver.close()
