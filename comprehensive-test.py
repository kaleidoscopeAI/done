import asyncio
import yaml
import json
from datetime import datetime
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any

class BiologicalAnalysisSuite:
    """
    Comprehensive test suite for running multiple biological analyses.
    """
    def __init__(self):
        self.configurations = [
            "mitochondrial_analysis.yaml",
            "nuclear_regulation.yaml",
            "protein_trafficking.yaml",
            "cytoskeletal_dynamics.yaml",
            "cell_cycle.yaml"
        ]
        self.results_dir = Path("analysis_results")
        self.results_dir.mkdir(exist_ok=True)
        self.analysis_results = {}
        
    async def run_comprehensive_analysis(self):
        """
        Execute analysis for each biological configuration and compile results.
        """
        start_time = datetime.now()
        print(f"Starting Comprehensive Biological Analysis at {start_time}")
        
        for config_file in self.configurations:
            await self._run_single_analysis(config_file)
            
        self._compile_comparative_results()
        self._generate_analysis_report()
        
        end_time = datetime.now()
        duration = end_time - start_time
        print(f"\nComplete Analysis Duration: {duration}")
        
    async def _run_single_analysis(self, config_file: str):
        """
        Execute analysis for a single biological configuration.
        """
        print(f"\nProcessing Configuration: {config_file}")
        print("-" * 50)
        
        # Load configuration
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Initialize pipeline with configuration
        pipeline = ConfigurableDataPipeline()
        pipeline.load_config(config_file)
        
        try:
            # Execute pipeline and collect results
            start_time = datetime.now()
            kaleidoscope_system = await pipeline.run()
            duration = datetime.now() - start_time
            
            # Collect analysis metrics
            cluster_summary = kaleidoscope_system.cluster_manager.get_cluster_summary()
            
            # Store results
            self.analysis_results[config['name']] = {
                'execution_time': duration.total_seconds(),
                'total_clusters': cluster_summary['total_clusters'],
                'total_nodes': cluster_summary['total_nodes'],
                'clusters': cluster_summary['clusters'],
                'configuration': config,
                'system_state': kaleidoscope_system.get_status()
            }
            
            # Generate and save visualization
            fig = kaleidoscope_system.visualize_cellular_network()
            fig.savefig(self.results_dir / f"{config['name']}_network.png")
            plt.close(fig)
            
        except Exception as e:
            print(f"Error processing {config_file}: {str(e)}")
            self.analysis_results[config['name']] = {'error': str(e)}
    
    def _compile_comparative_results(self):
        """
        Compile comparative metrics across all analyses.
        """
        comparative_data = []
        
        for analysis_name, results in self.analysis_results.items():
            if 'error' not in results:
                comparative_data.append({
                    'Analysis': analysis_name,
                    'Execution Time (s)': results['execution_time'],
                    'Total Clusters': results['total_clusters'],
                    'Total Nodes': results['total_nodes'],
                    'Average Cluster Size': results['total_nodes'] / results['total_clusters'],
                    'Cross-Domain Connections': self._count_cross_domain_connections(results['clusters'])
                })
        
        self.comparative_results = pd.DataFrame(comparative_data)
        self.comparative_results.to_csv(self.results_dir / 'comparative_results.csv', index=False)
        
    def _count_cross_domain_connections(self, clusters: Dict) -> int:
        """
        Count connections between different domains in the cluster data.
        """
        cross_connections = 0
        processed_pairs = set()
        
        for cluster_id, info in clusters.items():
            domain = info['domain']
            for node in info['nodes']:
                for connection in node['connections']:
                    # Get domain of connected node
                    connected_domain = None
                    for other_cluster_id, other_info in clusters.items():
                        if connection in [n['id'] for n in other_info['nodes']]:
                            connected_domain = other_info['domain']
                            break
                    
                    if connected_domain and connected_domain != domain:
                        pair = tuple(sorted([domain, connected_domain]))
                        if pair not in processed_pairs:
                            cross_connections += 1
                            processed_pairs.add(pair)
        
        return cross_connections
    
    def _generate_analysis_report(self):
        """
        Generate comprehensive analysis report with visualizations.
        """
        report_path = self.results_dir / 'analysis_report.html'
        
        # Create comparative visualizations
        self._create_comparative_plots()
        
        # Generate HTML report
        with open(report_path, 'w') as f:
            f.write(self._generate_html_report())
            
        print(f"\nAnalysis report generated: {report_path}")
    
    def _create_comparative_plots(self):
        """
        Create comparative visualizations of analysis results.
        """
        # Execution time comparison
        plt.figure(figsize=(10, 6))
        self.comparative_results.plot(
            kind='bar',
            x='Analysis',
            y='Execution Time (s)',
            title='Analysis Execution Time Comparison'
        )
        plt.tight_layout()
        plt.savefig(self.results_dir / 'execution_time_comparison.png')
        plt.close()
        
        # Cluster metrics comparison
        plt.figure(figsize=(12, 6))
        self.comparative_results[['Analysis', 'Total Clusters', 'Average Cluster Size']].plot(
            kind='bar',
            x='Analysis',
            title='Cluster Metrics Comparison'
        )
        plt.tight_layout()
        plt.savefig(self.results_dir / 'cluster_metrics_comparison.png')
        plt.close()
        
        # Cross-domain connections
        plt.figure(figsize=(10, 6))
        self.comparative_results.plot(
            kind='bar',
            x='Analysis',
            y='Cross-Domain Connections',
            title='Cross-Domain Connections by Analysis'
        )
        plt.tight_layout()
        plt.savefig(self.results_dir / 'cross_domain_connections.png')
        plt.close()
    
    def _generate_html_report(self) -> str:
        """
        Generate HTML report content.
        """
        html_content = f"""
        <html>
        <head>
            <title>Biological Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .section {{ margin-bottom: 30px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                img {{ max-width: 100%; height: auto; }}
            </style>
        </head>
        <body>
            <h1>Comprehensive Biological Analysis Report</h1>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="section">
                <h2>Comparative Results</h2>
                {self.comparative_results.to_html()}
            </div>
            
            <div class="section">
                <h2>Execution Time Comparison</h2>
                <img src="execution_time_comparison.png" alt="Execution Time Comparison">
            </div>
            
            <div class="section">
                <h2>Cluster Metrics Comparison</h2>
                <img src="cluster_metrics_comparison.png" alt="Cluster Metrics Comparison">
            </div>
            
            <div class="section">
                <h2>Cross-Domain Connections</h2>
                <img src="cross_domain_connections.png" alt="Cross-Domain Connections">
            </div>
            
            <div class="section">
                <h2>Knowledge Networks</h2>
                {self._generate_network_gallery()}
            </div>
        </body>
        </html>
        """
        return html_content
    
    def _generate_network_gallery(self) -> str:
        """
        Generate HTML for knowledge network visualizations.
        """
        gallery_html = "<div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px;'>"
        
        for analysis_name in self.analysis_results.keys():
            network_path = f"{analysis_name}_network.png"
            if (self.results_dir / network_path).exists():
                gallery_html += f"""
                <div>
                    <h3>{analysis_name}</h3>
                    <img src="{network_path}" alt="{analysis_name} Network">
                </div>
                """
        
        gallery_html += "</div>"
        return gallery_html

async def run_analysis_suite():
    """
    Execute the complete analysis suite.
    """
    suite = BiologicalAnalysisSuite()
    await suite.run_comprehensive_analysis()

if __name__ == "__main__":
    asyncio.run(run_analysis_suite())
