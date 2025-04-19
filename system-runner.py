import asyncio
import torch
import ray
from typing import Dict, Any
import logging
from pathlib import Path
import yaml
import argparse

class SystemRunner:
    def __init__(self, config_path: str):
        self.config = self.load_config(config_path)
        self.system = self.initialize_system()
        self.scheduler = DistributedScheduler(self.config['world_size'])
        self.logger = self.setup_logging()
        
    def load_config(self, config_path: str) -> Dict[str, Any]:
        with open(config_path) as f:
            return yaml.safe_load(f)
            
    def initialize_system(self) -> FaultTolerantProcessor:
        return create_distributed_system(
            world_size=self.config['world_size'],
            hdim=self.config['hdim']
        )
        
    def setup_logging(self) -> logging.Logger:
        logger = logging.getLogger('SystemRunner')
        logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler('system.log')
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(handler)
        
        return logger
        
    async def run(self, data_path: str):
        try:
            # Load data
            data = torch.load(data_path)
            self.logger.info(f"Loaded data: {data.shape}")
            
            # Create processing schedule
            tasks = self.create_task_list(data)
            schedule = self.scheduler.schedule_tasks(tasks)
            self.logger.info(f"Created schedule with {len(tasks)} tasks")
            
            # Execute processing
            results = await self.scheduler.execute_schedule(
                schedule,
                self.system.processor
            )
            self.logger.info("Processing completed successfully")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error during processing: {str(e)}", exc_info=True)
            raise
            
    def create_task_list(self, data: torch.Tensor) -> List[Tuple[str, Any]]:
        n_chunks = self.config['world_size']
        chunks = torch.chunk(data, n_chunks)
        
        tasks = []
        for i in range(n_chunks):
            # Add topology processing task
            tasks.append(('topology', []))
            # Add quantum processing task
            tasks.append(('quantum', []))
            # Add combination task depending on previous tasks
            tasks.append(('combine', [i*2, i*2+1]))
            
        return tasks
        
    async def shutdown(self):
        try:
            await self.system.processor.shutdown()
            ray.shutdown()
            self.logger.info("System shutdown completed")
        except Exception as e:
            self.logger.error(f"Error during shutdown: {str(e)}", exc_info=True)

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--data', type=str, required=True)
    args = parser.parse_args()
    
    runner = SystemRunner(args.config)
    try:
        results = await runner.run(args.data)
        print("Processing results:", results)
    finally:
        await runner.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
