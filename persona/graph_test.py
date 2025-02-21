from src.manager.manager_graph import Manager_Graph

if __name__ == "__main__":
    # Get the singleton instance.
    graph_manager = Manager_Graph()
    
    # Run a sample query.
    sample_query = "MATCH (n) RETURN n LIMIT 5"
    result = graph_manager.run_query(sample_query)
    print("Query result:", result)
    
    # Close the connection when finished.
    graph_manager.close()

