# Scalability Framework Documentation

## Overview

The Scalability Framework is a comprehensive solution designed to handle conversation categorization and summarization at scale, from 19K to 500K+ tickets efficiently. It implements horizontal scaling, memory-efficient streaming, dynamic resource allocation, cost modeling, and auto-scaling capabilities.

## Architecture

### Core Components

1. **ScalabilityManager** - Central orchestrator for horizontal scaling operations
2. **StreamingProcessor** - Memory-efficient data streaming system  
3. **ResourceAllocator** - Intelligent resource allocation algorithms
4. **CostModels** - Cost modeling and analytics engine
5. **AutoScaler** - Real-time auto-scaling system
6. **ScalabilityFramework** - Unified integration layer

## Quick Start

### Basic Usage

```python
from scalability_framework import ScalabilityFramework, ScalabilityConfiguration

# Configure framework
config = ScalabilityConfiguration(
    max_workers=16,
    worker_type="thread",
    target_mode="balanced",
    enable_auto_scaling=True
)

# Initialize framework
framework = ScalabilityFramework(config)

# Analyze dataset requirements
analysis = framework.analyze_dataset_requirements(Path("data.csv"))

# Process dataset with scaling
def process_chunk(chunk, index):
    # Your processing logic here
    return processed_results

results = framework.process_dataset_scalable(
    input_file=Path("data.csv"),
    processor_function=process_chunk
)
```

### Advanced Configuration

```python
# Constraint-based optimization
optimization = framework.optimize_for_constraints(
    dataset_size=100000,
    constraints={
        "budget_limit": 500.0,  # $500 budget
        "time_limit_hours": 4.0,  # 4 hour limit
        "memory_limit_gb": 8.0   # 8GB memory limit
    }
)

# Real-time monitoring
dashboard = framework.get_real_time_dashboard()
```

## Resource Profiles

The framework includes predefined resource profiles optimized for different dataset sizes:

| Dataset Size | Workers | Memory/Worker | Cost/Ticket | Estimated Duration |
|--------------|---------|---------------|-------------|------------------|
| 1-1K | 2 | 512MB | $0.048 | ~7 minutes |
| 1K-10K | 4 | 512MB | $0.045 | ~28 minutes |
| 10K-50K | 8 | 768MB | $0.042 | ~1.8 hours |
| 50K-100K | 16 | 1024MB | $0.040 | ~2.4 hours |
| 100K-500K | 32 | 1536MB | $0.038 | ~4.2 hours |
| 500K+ | 64 | 2048MB | $0.036 | ~5.8 hours |

## Integration Guide

### With Existing Categorizer

```python
from categorizer import Categorizer
from scalability_framework import ScalabilityFramework

class ScalableCategorizer(Categorizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.framework = ScalabilityFramework()
    
    def categorize_tickets_scalable(self, input_file, nrows=None):
        # Analyze dataset
        analysis = self.framework.analyze_dataset_requirements(input_file)
        
        # Configure processing
        def process_batch(batch, index):
            return self.process_batch_internal(batch)
        
        # Process with scaling
        return self.framework.process_dataset_scalable(
            input_file=input_file,
            processor_function=process_batch,
            nrows=nrows
        )
```

### With Existing Summarizer

```python
from summarizer import Summarizer
from scalability_framework import ScalabilityFramework

class ScalableSummarizer(Summarizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.framework = ScalabilityFramework()
    
    def summarize_tickets_scalable(self, input_file, nrows=None):
        # Use streaming for memory efficiency
        config = ScalabilityConfiguration(
            target_mode="memory_optimized",
            streaming_chunk_size=500,
            streaming_memory_limit_mb=1024
        )
        
        framework = ScalabilityFramework(config)
        
        def process_batch(batch, index):
            return self.process_batch_internal(batch)
        
        return framework.process_dataset_scalable(
            input_file=input_file,
            processor_function=process_batch,
            nrows=nrows
        )
```

## Configuration Options

### ScalabilityConfiguration

```python
@dataclass
class ScalabilityConfiguration:
    # General settings
    max_workers: int = 16
    worker_type: str = "thread"  # "thread" or "process"
    target_mode: str = "balanced"  # "speed_optimized", "memory_optimized", "cost_optimized", "balanced"
    
    # Streaming settings
    streaming_chunk_size: int = 1000
    streaming_buffer_mb: int = 64
    streaming_memory_limit_mb: int = 512
    enable_compression: bool = True
    
    # Auto-scaling settings
    enable_auto_scaling: bool = True
    min_workers: int = 2
    max_auto_scale_workers: int = 32
    
    # Cost constraints
    budget_limit_usd: Optional[float] = None
    time_limit_hours: Optional[float] = None
```

## Performance Optimization

### Memory Management

- Use `target_mode="memory_optimized"` for large datasets
- Configure `streaming_memory_limit_mb` based on available RAM
- Enable compression with `enable_compression=True`

### Speed Optimization

- Use `target_mode="speed_optimized"` for time-critical processing
- Increase `max_workers` up to 2x CPU cores
- Use `worker_type="process"` for CPU-intensive tasks

### Cost Optimization

- Use `target_mode="cost_optimized"` for budget constraints
- Set `budget_limit_usd` for automatic optimization
- Monitor costs with real-time dashboard

## Monitoring and Analytics

### Real-time Dashboard

```python
dashboard = framework.get_real_time_dashboard()
print(f"Current workers: {dashboard['current_status']['current_workers']}")
print(f"Throughput: {dashboard['current_status']['current_metrics']['throughput_tasks_per_minute']}")
```

### Cost Analysis

```python
cost_analysis = framework.cost_analytics.generate_comprehensive_analysis(
    dataset_size=100000,
    budget_limit=500.0
)
print(f"Estimated cost: ${cost_analysis['summary']['cost_range']['optimal_cost_per_ticket']:.4f} per ticket")
```

### Performance Reports

```python
# Export detailed scaling report
report_path = framework.scalability_manager.export_scaling_report()
print(f"Report saved to: {report_path}")
```

## Error Handling and Recovery

### Automatic Recovery

- Failed batches are automatically retried
- Auto-scaling adjusts to system load
- Memory pressure triggers automatic optimization

### Error Monitoring

```python
# Monitor for errors
metrics = framework.get_real_time_dashboard()
error_rate = metrics['current_status']['current_metrics']['error_rate_percent']

if error_rate > 5.0:
    print("High error rate detected - scaling down")
    framework.auto_scaler.manual_scale(target_workers=4, reason="High error rate")
```

## Best Practices

### Dataset Size Guidelines

1. **Small datasets (< 10K)**: Use default configuration
2. **Medium datasets (10K-100K)**: Enable auto-scaling
3. **Large datasets (100K+)**: Use memory-optimized mode with streaming

### Resource Allocation

1. Start with balanced mode for unknown workloads
2. Monitor memory usage and adjust streaming limits
3. Use cost optimization for budget-constrained scenarios

### Performance Tuning

1. Profile your processing function before scaling
2. Monitor worker efficiency and adjust batch sizes
3. Use the resource allocator's historical learning

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce `streaming_chunk_size` and `streaming_buffer_mb`
2. **Low Throughput**: Increase `max_workers` or check processing function efficiency
3. **High Costs**: Enable cost optimization or reduce worker count

### Debugging

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.INFO)

# Check system resources
from resource_allocator import ResourceAllocator
allocator = ResourceAllocator()
system_resources = allocator.system_profiler.get_current_resources()
print(f"Available memory: {system_resources.memory_available_gb:.1f}GB")
```

## API Reference

### Main Classes

- `ScalabilityFramework`: Main orchestrator
- `ScalabilityConfiguration`: Configuration object
- `ScalabilityManager`: Horizontal scaling manager
- `StreamingDataProcessor`: Memory-efficient streaming
- `ResourceAllocator`: Intelligent resource allocation
- `CostAnalyticsEngine`: Cost modeling and optimization
- `AutoScaler`: Real-time auto-scaling

### Key Methods

- `analyze_dataset_requirements()`: Analyze and recommend configuration
- `process_dataset_scalable()`: Process with automatic scaling
- `optimize_for_constraints()`: Constraint-based optimization
- `get_real_time_dashboard()`: Monitor current status

## Examples

See the `examples/scalability_example.py` file for comprehensive usage examples including:

- Basic dataset analysis
- Scalable processing with different modes
- Constraint optimization
- Real-time monitoring
- Cost analysis and reporting

## Performance Benchmarks

The framework has been designed and tested to handle:

- Linear scaling from 19K to 500K+ tickets
- Constant memory footprint regardless of dataset size
- Cost efficiency of $0.036-$0.048 per ticket
- Processing rates of 100+ tickets per minute per worker

## Future Enhancements

Planned improvements include:

1. GPU acceleration support
2. Distributed processing across multiple machines
3. Advanced ML-based resource prediction
4. Integration with cloud auto-scaling services
5. Enhanced cost optimization algorithms

---

For support or questions, please refer to the implementation code or create an issue in the project repository.