import os
from neo4j import GraphDatabase


def get_graph_data(cypher="MATCH (s)-[r:!MENTIONS]->(t) RETURN s,r,t LIMIT 50"):
    driver = GraphDatabase.driver(
        uri=os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"])
    )
    with driver.session() as session:
        result = session.run(cypher)
        nodes = set()
        edges = []
        for record in result:
            for node in [record["s"], record["t"]]:
                nodes.add(f'"{node["id"]}" [label="{node["id"]}"]')
            edges.append(f'"{record["s"]["id"]}" -> "{record["t"]["id"]}" [label="{record["r"].type}"]')

    dot_data = "digraph G {\n" + "\n".join(nodes) + "\n" + "\n".join(edges) + "\n}"
    return dot_data