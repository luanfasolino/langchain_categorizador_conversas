"""
Example usage of the Scalability Framework.

This example demonstrates how to use the complete scalability framework
to process large datasets efficiently with automatic scaling and optimization.
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from scalability_framework import ScalabilityFramework, ScalabilityConfiguration
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_ticket_processor(chunk: pd.DataFrame, chunk_index: int) -> list:
    """
    Example ticket processing function.
    
    This simulates the actual ticket categorization/summarization work.
    In practice, this would call your LangChain chains.
    """
    logger.info(f"Processing chunk {chunk_index} with {len(chunk)} tickets")
    
    # Simulate processing each ticket
    results = []
    for _, row in chunk.iterrows():
        # Simulate categorization result
        result = {
            "ticket_id": row.get("ticket_id", "unknown"),
            "categoria": "Categoria Exemplo",
            "resumo": f"Resumo do ticket {row.get('ticket_id', 'unknown')}",
            "processed_by_chunk": chunk_index
        }
        results.append(result)
    
    return results


def demo_basic_analysis():
    """Demonstrate basic dataset analysis."""
    print("\n=== Demo: Basic Dataset Analysis ===")
    
    # Configuration for analysis
    config = ScalabilityConfiguration(
        max_workers=8,
        target_mode="balanced",
        enable_auto_scaling=True,
        streaming_chunk_size=500,
        budget_limit_usd=50.0
    )
    
    # Initialize framework
    framework = ScalabilityFramework(config)
    
    # Create a dummy dataset file for demonstration
    dummy_file = Path("database/example_dataset.csv")
    dummy_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Create sample data if it doesn't exist
    if not dummy_file.exists():
        sample_data = {
            "ticket_id": [f"TICKET_{i:06d}" for i in range(1000)],
            "category": ["TEXT"] * 1000,
            "text": [f"Example ticket text content for ticket {i}" * 10 for i in range(1000)],
            "sender": ["USER" if i % 2 == 0 else "AGENT" for i in range(1000)]
        }
        pd.DataFrame(sample_data).to_csv(dummy_file, index=False, sep=";")
        print(f"Created sample dataset: {dummy_file}")
    
    # Analyze dataset requirements
    try:
        analysis = framework.analyze_dataset_requirements(dummy_file)
        
        print(f"\n📊 Analysis Results:")
        print(f"Recommended workers: {analysis['recommended_configuration']['workers']}")
        print(f"Estimated duration: {analysis['recommended_configuration']['estimated_duration_minutes']:.1f} minutes")
        print(f"Estimated cost: ${analysis['recommended_configuration']['estimated_cost']:.2f}")
        print(f"Processing mode: {analysis['recommended_configuration']['processing_mode']}")
        
        print(f"\n✅ Validation: {'PASSED' if analysis['validation']['is_valid'] else 'FAILED'}")
        if analysis['validation']['issues']:
            print("Issues found:")
            for issue in analysis['validation']['issues']:
                print(f"  - {issue}")
        
        print(f"\n💡 Framework Recommendations:")
        for rec in analysis['framework_recommendations']:
            print(f"  - {rec}")
        
        return analysis
        
    except Exception as e:
        logger.error(f"Error in analysis: {str(e)}")
        return None
    finally:
        framework.cleanup()


def demo_scalable_processing():
    """Demonstrate scalable dataset processing."""
    print("\n=== Demo: Scalable Dataset Processing ===")
    
    # Configuration for processing
    config = ScalabilityConfiguration(
        max_workers=4,
        worker_type="thread",
        target_mode="speed_optimized",
        enable_auto_scaling=True,
        streaming_chunk_size=100,  # Smaller chunks for demo
        streaming_memory_limit_mb=256
    )
    
    # Initialize framework
    framework = ScalabilityFramework(config)
    
    # Input and output files
    input_file = Path("database/example_dataset.csv")
    output_file = Path("database/processed_results.csv")
    
    if not input_file.exists():
        print("⚠️ Sample dataset not found. Run demo_basic_analysis() first.")
        return
    
    try:
        # Process dataset with scalability framework
        print("🚀 Starting scalable processing...")
        
        results = framework.process_dataset_scalable(
            input_file=input_file,
            processor_function=example_ticket_processor,
            output_file=output_file,
            nrows=500,  # Process only first 500 rows for demo
            enable_monitoring=True
        )
        
        print(f"\n📈 Processing Results:")
        print(f"Session ID: {results['session_id']}")
        print(f"Processing duration: {results['processing_duration']}")
        print(f"Items processed: {results['total_items_processed']:,}")
        print(f"Chunks processed: {results['chunks_processed']}")
        
        print(f"\n💾 Streaming Statistics:")
        streaming_stats = results['streaming_stats']
        print(f"Memory usage: {streaming_stats['memory_usage']['current_mb']:.1f}MB")
        print(f"Peak memory: {streaming_stats['memory_usage']['peak_mb']:.1f}MB")
        print(f"Temp files: {streaming_stats['temp_files']['count']}")
        
        if results.get('auto_scaling_stats'):
            print(f"\n🔄 Auto-scaling Statistics:")
            auto_stats = results['auto_scaling_stats']
            print(f"Final worker count: {auto_stats['current_status']['current_workers']}")
            print(f"Scaling events: {auto_stats['scaling_history']['total_events']}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in processing: {str(e)}")
        return None
    finally:
        framework.cleanup()


def demo_constraint_optimization():
    """Demonstrate optimization for specific constraints."""
    print("\n=== Demo: Constraint Optimization ===")
    
    config = ScalabilityConfiguration()
    framework = ScalabilityFramework(config)
    
    try:
        # Define constraints
        constraints = {
            "budget_limit": 25.0,  # $25 budget limit
            "time_limit_hours": 2.0,  # 2 hour time limit
            "memory_limit_gb": 4.0  # 4GB memory limit
        }
        
        # Optimize for dataset of 10,000 tickets
        dataset_size = 10000
        
        print(f"🎯 Optimizing configuration for {dataset_size:,} tickets")
        print(f"Constraints: {constraints}")
        
        optimization = framework.optimize_for_constraints(dataset_size, constraints)
        
        print(f"\n💰 Budget Optimization:")
        if "budget_optimization" in optimization["individual_optimizations"]:
            budget_opt = optimization["individual_optimizations"]["budget_optimization"]
            print(f"  Recommended workers: {budget_opt.get('recommended_changes', ['N/A'])[0]}")
            print(f"  Estimated cost: ${budget_opt.get('optimized_cost', 0):.2f}")
            print(f"  Savings: ${budget_opt.get('savings_amount', 0):.2f}")
        
        print(f"\n⏱️ Time Optimization:")
        if "time_optimization" in optimization["individual_optimizations"]:
            time_opt = optimization["individual_optimizations"]["time_optimization"]
            print(f"  Changes: {time_opt.get('recommended_changes', ['N/A'])}")
        
        print(f"\n🔧 Combined Recommendation:")
        combined = optimization["combined_recommendation"]
        print(f"  Workers: {combined['workers']}")
        print(f"  Memory per worker: {combined['memory_per_worker_mb']}MB")
        print(f"  Optimization priority: {combined['optimization_priority']}")
        
        print(f"\n⚖️ Trade-offs:")
        for trade_off in optimization["trade_offs"]:
            print(f"  - {trade_off}")
        
        return optimization
        
    except Exception as e:
        logger.error(f"Error in optimization: {str(e)}")
        return None
    finally:
        framework.cleanup()


def demo_real_time_dashboard():
    """Demonstrate real-time dashboard functionality."""
    print("\n=== Demo: Real-time Dashboard ===")
    
    config = ScalabilityConfiguration(enable_auto_scaling=True)
    framework = ScalabilityFramework(config)
    
    try:
        # Get dashboard data
        dashboard = framework.get_real_time_dashboard()
        
        print(f"📊 Framework Status:")
        status = dashboard["framework_status"]
        print(f"  Current session: {status['current_session'] or 'None'}")
        print(f"  Auto-scaling enabled: {status['auto_scaling_enabled']}")
        print(f"  Target mode: {status['framework_config']['target_mode']}")
        
        print(f"\n💻 Resource Utilization:")
        resources = dashboard["resource_utilization"]
        if resources.get("status") != "unavailable":
            print(f"  CPU usage: {resources.get('cpu_percent', 0):.1f}%")
            print(f"  Memory usage: {resources.get('memory_percent', 0):.1f}%")
            print(f"  Available memory: {resources.get('available_memory_gb', 0):.1f}GB")
        else:
            print("  Resource monitoring unavailable")
        
        print(f"\n💰 Cost Tracking:")
        cost_info = dashboard["cost_tracking"]
        print(f"  Session: {cost_info['session_id'] or 'No active session'}")
        print(f"  Estimated cost: ${cost_info['estimated_session_cost']:.2f}")
        
        return dashboard
        
    except Exception as e:
        logger.error(f"Error getting dashboard: {str(e)}")
        return None
    finally:
        framework.cleanup()


def demo_framework_summary():
    """Show framework capabilities summary."""
    print("\n=== Demo: Framework Summary ===")
    
    config = ScalabilityConfiguration()
    framework = ScalabilityFramework(config)
    
    try:
        summary = framework.export_framework_summary()
        
        print(f"🔧 Framework Information:")
        info = summary["framework_info"]
        print(f"  Version: {info['version']}")
        print(f"  Components: {len(info['components'])}")
        for component in info["components"]:
            print(f"    - {component}")
        
        print(f"\n🚀 Capabilities:")
        for capability in info["capabilities"]:
            print(f"  - {capability}")
        
        print(f"\n📁 Storage Locations:")
        storage = summary["storage_locations"]
        for location, path in storage.items():
            print(f"  {location}: {path}")
        
        print(f"\n📖 Usage Examples:")
        examples = summary["usage_examples"]
        for method, example in examples.items():
            print(f"  {method}: {example}")
        
        return summary
        
    except Exception as e:
        logger.error(f"Error getting summary: {str(e)}")
        return None
    finally:
        framework.cleanup()


if __name__ == "__main__":
    print("🎯 Scalability Framework Demo")
    print("=" * 50)
    
    # Run all demos
    try:
        # Basic analysis
        analysis = demo_basic_analysis()
        
        # Scalable processing
        if analysis:
            processing_results = demo_scalable_processing()
        
        # Constraint optimization
        optimization = demo_constraint_optimization()
        
        # Real-time dashboard
        dashboard = demo_real_time_dashboard()
        
        # Framework summary
        summary = demo_framework_summary()
        
        print("\n✅ All demos completed successfully!")
        print("\nNext steps:")
        print("1. Integrate with your existing BaseProcessor")
        print("2. Configure for your specific dataset sizes")
        print("3. Set up monitoring and alerting")
        print("4. Test with real data and adjust thresholds")
        
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        print(f"\n❌ Demo failed: {str(e)}")
        print("Check logs for detailed error information.")