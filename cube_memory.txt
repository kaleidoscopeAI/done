class CubeMemory:
    def __init__(self):
        self.memory_store = {}

    def store_insight(self, node_id, insight):
        """Store insights inside the Cube memory."""
        if node_id not in self.memory_store:
            self.memory_store[node_id] =
        self.memory_store[node_id].append(insight)

    def retrieve_insights(self, node_id):
        """Retrieve stored insights for a given node."""
        return self.memory_store.get(node_id,)
