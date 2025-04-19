class AnalysisMonitor:
    def __init__(self):
        self.start_time = datetime.now()
        self.current_phase = None
        self.progress = {}
        
    async def monitor_analysis(self, analysis_suite):
        print("Starting Comprehensive Cell Biology Analysis")
        print("-" * 50)
        
        try:
            # Monitor Mitochondrial Analysis
            print("\nPhase 1: Mitochondrial Energy Production")
            mitochondrial_results = await self._monitor_phase(
                analysis_suite, 
                "mitochondrial_analysis.yaml"
            )
            
            # Monitor Nuclear Analysis
            print("\nPhase 2: Nuclear Organization")
            nuclear_results = await self._monitor_phase(
                analysis_suite,
                "nuclear_regulation.yaml"
            )
            
            # Monitor Protein Trafficking
            print("\nPhase 3: Protein Trafficking")
            trafficking_results = await self._monitor_phase(
                analysis_suite,
                "protein_trafficking.yaml"
            )
            
            # Monitor Cytoskeletal Analysis
            print("\nPhase 4: Cytoskeletal Dynamics")
            cytoskeletal_results = await self._monitor_phase(
                analysis_suite,
                "cytoskeletal_dynamics.yaml"
            )
            
            # Monitor Cell Cycle Analysis
            print("\nPhase 5: Cell Cycle Analysis")
            cell_cycle_results = await self._monitor_phase(
                analysis_suite,
                "cell_cycle.yaml"
            )
            
            # Generate Comprehensive Report
            print("\nGenerating Final Analysis Report")
            self._compile_results([
                mitochondrial_results,
                nuclear_results,
                trafficking_results,
                cytoskeletal_results,
                cell_cycle_results
            ])
            
        except Exception as e:
            print(f"\nError during analysis: {str(e)}")
            raise
        
    async def _monitor_phase(self, analysis_suite, config_file):
        phase_start = datetime.now()
        results = await analysis_suite._run_single_analysis(config_file)
        duration = datetime.now() - phase_start
        
        print(f"Time Elapsed: {duration}")
        print(f"Clusters Generated: {results['total_clusters']}")
        print(f"Knowledge Nodes: {results['total_nodes']}")
        
        return results
        
    def _compile_results(self, phase_results):
        total_duration = datetime.now() - self.start_time
        print("\nAnalysis Complete!")
        print(f"Total Analysis Time: {total_duration}")
        print("\nResults Summary:")
        print("-" * 50)
        print(f"Total Knowledge Clusters: {sum(r['total_clusters'] for r in phase_results)}")
        print(f"Total Knowledge Nodes: {sum(r['total_nodes'] for r in phase_results)}")

# Execute the monitored analysis
async def run_monitored_analysis():
    monitor = AnalysisMonitor()
    suite = BiologicalAnalysisSuite()
    await monitor.monitor_analysis(suite)

if __name__ == "__main__":
    asyncio.run(run_monitored_analysis())
