import ray
from typing import List, Dict
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from pathlib import Path
import numpy as np
from dataclasses import dataclass
import asyncio
import zmq
import zmq.asyncio

@dataclass
class ClusterConfig:
    num_workers: int
    memory_per_worker: int
    input_dim: int
    hidden_dim: int
    batch_size: int

@ray.remote
class DistributedNode:
    def __init__(self, rank: int, world_size: int, config: ClusterConfig):
        self.rank = rank
        self.world_size = world_size
        self.config = config
        self._initialize_distributed()
        self.processor = self._create_processor()
        
    def _initialize_distributed(self):
        dist.init_process_group(
            backend='nccl' if torch.cuda.is_available() else 'gloo',
            rank=self.rank,
            world_size=self.world_size
        )
        
    def _create_processor(self) -> torch.nn.Module:
        device = torch.device(f'cuda:{self.rank}' if torch.cuda.is_available() else 'cpu')
        model = KaleidoscopeEngine(self.config.input_dim, self.config.hidden_dim).to(device)
        return DistributedDataParallel(model)
        
    async def process_data(self, data: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.processor(data)
            
    def cleanup(self):
        dist.destroy_process_group()

class DistributedCluster:
    def __init__(self, config: ClusterConfig):
        self.config = config
        ray.init()
        self.nodes: List[DistributedNode] = []
        self.initialize_cluster()
        
    def initialize_cluster(self):
        for i in range(self.config.num_workers):
            node = DistributedNode.remote(i, self.config.num_workers, self.config)
            self.nodes.append(node)
            
    async def process_batch(self, batch: torch.Tensor) -> List[torch.Tensor]:
        futures = [node.process_data.remote(batch) for node in self.nodes]
        return await asyncio.gather(*[ray.get(future) for future in futures])
        
    def shutdown(self):
        for node in self.nodes:
            ray.get(node.cleanup.remote())
        ray.shutdown()

class KaleidoscopeRunner:
    def __init__(self, data_path: Path, config: ClusterConfig):
        self.data_path = data_path
        self.config = config
        self.cluster = DistributedCluster(config)
        self.context = zmq.asyncio.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind("tcp://*:5555")
        
    def load_data(self) -> torch.utils.data.DataLoader:
        dataset = torch.load(self.data_path)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4
        )
        
    async def run_processing(self):
        data_loader = self.load_data()
        kaleidoscope = create_kaleidoscope_ai()
        
        for batch in data_loader:
            # Distribute batch processing
            processed_batches = await self.cluster.process_batch(batch)
            
            # Aggregate results
            combined_batch = torch.stack(processed_batches).mean(0)
            
            # Feed into KaleidoscopeAI
            supernode = kaleidoscope.run(combined_batch)
            
            # Handle chat interface
            while True:
                message = await self.socket.recv_string()
                if message == "EXIT":
                    break
                    
                response = kaleidoscope.chat_interface(message)
                await self.socket.send_string(response)
                
        self.cluster.shutdown()
        
    @staticmethod
    async def create_and_run(data_path: str, num_workers: int = 4):
        config = ClusterConfig(
            num_workers=num_workers,
            memory_per_worker=8192,
            input_dim=512,
            hidden_dim=1024,
            batch_size=32
        )
        
        runner = KaleidoscopeRunner(Path(data_path), config)
        await runner.run_processing()

if __name__ == "__main__":
    import sys
    data_path = sys.argv[1]
    num_workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    
    asyncio.run(KaleidoscopeRunner.create_and_run(data_path, num_workers))
