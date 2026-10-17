"""
Complete end-to-end example of the multi-agent analytics platform.

This example demonstrates:
1. Data collection from a CSV file
2. Quality assurance checks
3. Data processing and cleaning
4. Machine learning analysis
5. Insight generation
6. Visualization creation
"""
import asyncio
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from agents.orchestrator_agent import OrchestratorAgent
from agents.data_collector_agent import DataCollectorAgent
from agents.quality_assurance_agent import QualityAssuranceAgent
from agents.data_processor_agent import DataProcessorAgent
from agents.analyzer_agent import AnalyzerAgent
from agents.insight_generator_agent import InsightGeneratorAgent
from agents.visualization_agent import VisualizationAgent
from utils.config import config
from utils.logger import get_logger

logger = get_logger("example")


def create_sample_data():
    """Create sample customer data for demonstration."""
    np.random.seed(42)

    # Generate sample e-commerce customer data
    n_samples = 1000

    data = {
        'customer_id': range(1, n_samples + 1),
        'age': np.random.randint(18, 70, n_samples),
        'purchase_amount': np.random.exponential(100, n_samples),
        'num_purchases': np.random.poisson(5, n_samples),
        'days_since_last_purchase': np.random.randint(1, 365, n_samples),
        'customer_segment': np.random.choice(['Bronze', 'Silver', 'Gold', 'Platinum'], n_samples),
        'satisfaction_score': np.random.randint(1, 6, n_samples),
        'churn_risk': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])  # Target variable
    }

    df = pd.DataFrame(data)

    # Add some missing values
    missing_mask = np.random.random(n_samples) < 0.05
    df.loc[missing_mask, 'satisfaction_score'] = np.nan

    # Save to CSV
    data_path = Path(__file__).parent.parent / 'data' / 'sample_customers.csv'
    data_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(data_path, index=False)

    logger.info(f"Sample data created: {data_path}")
    return data_path


async def run_complete_pipeline():
    """Run complete multi-agent pipeline."""

    print("=" * 80)
    print("MULTI-AGENT BIG DATA ANALYTICS PLATFORM - COMPLETE EXAMPLE")
    print("=" * 80)
    print()

    # Step 0: Create sample data
    print("Step 0: Creating sample customer data...")
    data_path = create_sample_data()
    print(f"✓ Sample data created: {data_path}\n")

    # Step 1: Initialize Orchestrator
    print("Step 1: Initializing Orchestrator Agent...")
    orchestrator = OrchestratorAgent()
    print("✓ Orchestrator initialized\n")

    # Step 2: Create workflow
    print("Step 2: Creating workflow...")
    data_info = {
        'type': 'structured',
        'format': 'csv',
        'size_bytes': data_path.stat().st_size,
        'source': 'file',
        'is_streaming': False,
        'file_path': str(data_path)
    }

    workflow_result = orchestrator.create_workflow(data_info)

    if workflow_result['status'] != 'success':
        print(f"✗ Failed to create workflow: {workflow_result.get('message')}")
        return

    workflow = workflow_result['output']['workflow']
    workflow_id = workflow['workflow_id']

    print(f"✓ Workflow created: {workflow_id}")
    print(f"  Pipeline: {workflow['pipeline']['name']}")
    print(f"  Estimated time: {workflow['pipeline']['estimated_time_seconds']}s")
    print(f"  Agents: {', '.join(workflow['pipeline']['agents'])}\n")

    # Step 3: Execute pipeline manually (since agents are implemented)
    print("Step 3: Executing pipeline...\n")

    # 3.1: Data Collection
    print("  → Data Collector Agent")
    collector = DataCollectorAgent()
    collection_task = {
        'task_id': 'task_collect',
        'input_data': {
            'source': 'file',
            'file_path': str(data_path)
        }
    }

    collection_result = collector.process_task(collection_task)
    if collection_result['status'] != 'success':
        print(f"    ✗ Data collection failed: {collection_result.get('message')}")
        return

    collected_data = collection_result['output']['data']
    print(f"    ✓ Collected {len(collected_data)} records")
    print(f"    ✓ Columns: {list(collected_data[0].keys())}\n")

    # 3.2: Quality Assurance
    print("  → Quality Assurance Agent")
    qa_agent = QualityAssuranceAgent()
    qa_task = {
        'task_id': 'task_qa',
        'input_data': collection_result['output']
    }

    qa_result = qa_agent.process_task(qa_task)
    if qa_result['status'] != 'success':
        print(f"    ✗ Quality check failed: {qa_result.get('message')}")
        return

    quality_report = qa_result['output']['quality_report']
    print(f"    ✓ Quality score: {quality_report['overall_score']:.2%}")
    print(f"    ✓ Action: {qa_result['output']['action']}")
    print(f"    ✓ Issues found: {len(quality_report['issues'])}\n")

    # 3.3: Data Processing
    print("  → Data Processor Agent")
    processor = DataProcessorAgent()
    process_task = {
        'task_id': 'task_process',
        'input_data': qa_result['output']
    }

    process_result = processor.process_task(process_task)
    if process_result['status'] != 'success':
        print(f"    ✗ Data processing failed: {process_result.get('message')}")
        return

    processed_data = process_result['output']['data']
    processing_info = process_result['output']['processing_info']
    print(f"    ✓ Processed {len(processed_data)} records")
    print(f"    ✓ Processing steps: {len(processing_info['processing_steps'])}\n")

    # 3.4: Analysis
    print("  → Analyzer Agent")
    analyzer = AnalyzerAgent()
    analysis_task = {
        'task_id': 'task_analyze',
        'input_data': process_result['output'],
        'analysis_type': 'classification'
    }

    analysis_result = analyzer.process_task(analysis_task)
    if analysis_result['status'] != 'success':
        print(f"    ✗ Analysis failed: {analysis_result.get('message')}")
        return

    analysis_output = analysis_result['output']
    print(f"    ✓ Analysis type: {analysis_output['analysis_type']}")
    print(f"    ✓ Model: {analysis_output.get('model', 'N/A')}")

    if 'results' in analysis_output:
        results = analysis_output['results']
        if 'accuracy' in results:
            print(f"    ✓ Accuracy: {results['accuracy']:.2%}")
    print()

    # 3.5: Insight Generation
    print("  → Insight Generator Agent")
    insight_gen = InsightGeneratorAgent()
    insight_task = {
        'task_id': 'task_insights',
        'input_data': analysis_result
    }

    insight_result = insight_gen.process_task(insight_task)
    if insight_result['status'] != 'success':
        print(f"    ✗ Insight generation failed: {insight_result.get('message')}")
        return

    insights = insight_result['output']['insights']
    exec_summary = insight_result['output']['executive_summary']

    print(f"    ✓ Generated {len(insights)} insights")
    print(f"    ✓ High-impact findings: {exec_summary['high_impact_count']}")
    print(f"    ✓ Actionable recommendations: {exec_summary['actionable_count']}\n")

    # 3.6: Visualization
    print("  → Visualization Agent")
    viz_agent = VisualizationAgent()
    viz_task = {
        'task_id': 'task_viz',
        'input_data': insight_result
    }

    viz_result = viz_agent.process_task(viz_task)
    if viz_result['status'] != 'success':
        print(f"    ✗ Visualization failed: {viz_result.get('message')}")
        return

    visualizations = viz_result['output']['visualizations']
    dashboard_url = viz_result['output'].get('dashboard_url')

    print(f"    ✓ Created {len(visualizations)} visualizations")
    if dashboard_url:
        print(f"    ✓ Dashboard: {dashboard_url}\n")

    # Step 4: Display Results
    print("=" * 80)
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print()

    print("📊 EXECUTIVE SUMMARY")
    print("-" * 80)
    print(exec_summary['summary_text'])
    print()

    print("🔍 TOP INSIGHTS")
    print("-" * 80)
    for idx, insight in enumerate(insights[:3], 1):
        print(f"\n{idx}. {insight['title']}")
        print(f"   {insight['description']}")
        print(f"   Impact: {insight['impact']} | Urgency: {insight['urgency']}")
        if insight.get('recommendation'):
            print(f"   💡 {insight['recommendation']}")

    print()
    print("=" * 80)
    print("✓ All visualizations saved to: ./output/")
    print("=" * 80)


async def run_simple_example():
    """Run a simple example with just one agent."""

    print("=" * 80)
    print("SIMPLE EXAMPLE: Data Collection Only")
    print("=" * 80)
    print()

    # Create sample data
    data_path = create_sample_data()

    # Initialize and run data collector
    collector = DataCollectorAgent()

    task = {
        'task_id': 'simple_task',
        'input_data': {
            'source': 'file',
            'file_path': str(data_path)
        }
    }

    result = collector.process_task(task)

    if result['status'] == 'success':
        print(f"✓ Successfully collected {result['output']['records_collected']} records")
        print(f"✓ Source: {result['output']['metadata']['source']}")
        print(f"✓ Execution time: {result['execution_time']:.2f}s")
    else:
        print(f"✗ Failed: {result.get('message')}")


if __name__ == "__main__":
    import sys

    print("\nMulti-Agent Big Data Analytics Platform")
    print("Select example to run:")
    print("1. Complete pipeline (all agents)")
    print("2. Simple example (data collection only)")
    print()

    try:
        choice = input("Enter choice (1 or 2): ").strip()

        if choice == "1":
            asyncio.run(run_complete_pipeline())
        elif choice == "2":
            asyncio.run(run_simple_example())
        else:
            print("Invalid choice. Running complete pipeline by default.")
            asyncio.run(run_complete_pipeline())

    except KeyboardInterrupt:
        print("\n\nExecution interrupted by user.")
    except Exception as e:
        print(f"\n\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
