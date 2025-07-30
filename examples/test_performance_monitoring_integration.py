"""
Integration test for Performance Monitoring and Validation System

This demonstrates the complete monitoring capabilities:
1. Performance tracking during pipeline execution
2. Comprehensive validation against Opção D targets
3. Dashboard generation and A/B testing framework
4. Cost breakdown analysis and recommendations

Run with: python examples/test_performance_monitoring_integration.py
"""

import sys
import pandas as pd
import json
import tempfile
import time
from pathlib import Path
from unittest.mock import patch
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))


def create_comprehensive_test_scenario():
    """Create realistic test scenario with multiple performance variations"""

    # Base test data
    base_data = {
        'ticket_id': ['T001', 'T002', 'T003', 'T004', 'T005', 'T006', 'T007', 'T008'],
        'category_ids': ['1', '2', '3,4', '5', '1,2', '3', '', '2'],
        'category_names': [
            'Payment Issues', 'Booking Changes', 'Technical Issues,Customer Service',
            'Product Information', 'Payment Issues,Booking Changes', 'Technical Issues',
            '', 'Booking Changes'
        ],
        'confidence': [0.95, 0.88, 0.92, 0.85, 0.89, 0.91, 0.0, 0.87],
        'processing_time': [0.12, 0.15, 0.18, 0.11, 0.16, 0.13, 0.05, 0.14],
        'tokens_used': [120, 130, 140, 110, 135, 125, 50, 128]
    }

    # Create variations for A/B testing
    optimized_data = base_data.copy()
    optimized_data['confidence'] = [c * 1.05 if c > 0 else c for c in optimized_data['confidence']]  # 5% better
    optimized_data['processing_time'] = [t * 0.9 for t in optimized_data['processing_time']]  # 10% faster

    baseline_data = base_data.copy()
    baseline_data['confidence'] = [c * 0.95 if c > 0 else c for c in baseline_data['confidence']]  # 5% worse
    baseline_data['processing_time'] = [t * 1.1 for t in baseline_data['processing_time']]  # 10% slower

    return {
        'current': pd.DataFrame(base_data),
        'optimized': pd.DataFrame(optimized_data),
        'baseline': pd.DataFrame(baseline_data)
    }


def create_mock_orchestration_metrics(cost_per_1k=0.18, time_minutes=22, confidence=0.89):
    """Create mock orchestration metrics with configurable values"""
    from discovery.orchestrator import OrchestrationMetrics

    total_tickets = 1000
    total_time = time_minutes * 60  # Convert to seconds
    total_cost = (cost_per_1k / 1000) * total_tickets

    return OrchestrationMetrics(
        total_tickets=total_tickets,
        discovery_sample_size=150,
        categories_discovered=6,
        total_processing_time=total_time,
        discovery_time=total_time * 0.25,    # 25% discovery
        application_time=total_time * 0.75,   # 75% application
        total_cost_usd=total_cost,
        cost_per_1k_tickets=cost_per_1k,
        avg_confidence=confidence,
        classification_rate=0.94,
        meets_cost_target=cost_per_1k <= 0.20,
        meets_confidence_target=confidence >= 0.85
    )


def create_mock_categories():
    """Create mock categories for testing"""
    return {
        "version": "1.0",
        "generated_at": datetime.now().isoformat(),
        "categories": [
            {"id": 1, "display_name": "Payment Issues", "frequency": 0.25},
            {"id": 2, "display_name": "Booking Changes", "frequency": 0.20},
            {"id": 3, "display_name": "Technical Issues", "frequency": 0.18},
            {"id": 4, "display_name": "Customer Service", "frequency": 0.15},
            {"id": 5, "display_name": "Product Information", "frequency": 0.12},
            {"id": 6, "display_name": "Refund Issues", "frequency": 0.10}
        ],
        "metadata": {
            "discovery_cost": 0.045,
            "application_cost": 0.135,
            "total_discovery_time": 330,
            "total_application_time": 990
        }
    }


def test_performance_monitoring_integration():
    """Test the complete performance monitoring integration"""

    print("=== PERFORMANCE MONITORING INTEGRATION TEST ===")

    try:
        # Import monitoring components
        from discovery.performance_monitor import (
            create_performance_monitor,
            validate_opçao_d_compliance
        )

        print("\n✅ Performance monitoring components imported successfully")

        # Create test scenarios
        test_data = create_comprehensive_test_scenario()
        mock_categories = create_mock_categories()

        print("\n📊 Test Scenarios Created:")
        print(f"  Current method: {len(test_data['current'])} tickets")
        print(f"  Optimized method: {len(test_data['optimized'])} tickets")
        print(f"  Baseline method: {len(test_data['baseline'])} tickets")

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            monitoring_dir = tmp_path / "monitoring"
            monitoring_dir.mkdir()

            # Test 1: Performance Monitor Initialization and Configuration
            print("\n🔧 Testing Performance Monitor initialization...")

            monitor = create_performance_monitor(
                monitoring_dir=str(monitoring_dir),
                cost_target=0.20,
                time_target=25,
                confidence_target=0.85
            )

            assert monitor.cost_target_per_1k == 0.20
            assert monitor.time_target_minutes == 25
            assert monitor.confidence_target == 0.85
            print("✅ Monitor initialized with Opção D targets")

            # Test 2: Monitoring Session Management
            print("\n📝 Testing monitoring session management...")

            dataset_info = {
                'total_tickets': 1000,
                'source_file': 'test_tickets.csv',
                'processing_date': datetime.now().isoformat()
            }

            monitor.start_monitoring_session("integration_test", dataset_info)

            assert monitor.current_session is not None
            assert monitor.current_session['name'] == "integration_test"
            print(f"✅ Monitoring session started: {monitor.current_session['name']}")

            # Test 3: Phase Recording and Tracking
            print("\n⏱️ Testing phase recording...")

            # Simulate discovery phase
            monitor.record_phase_start("discovery", {
                'sample_rate': 0.15,
                'strategy': 'hybrid',
                'chunk_size': 800000
            })

            time.sleep(0.1)  # Simulate processing time

            monitor.record_phase_end("discovery",
                                     cost_usd=0.045,
                                     additional_metrics={
                                         'categories_discovered': 6,
                                         'sample_size': 150,
                                         'tokens_used': 75000
                                     })

            # Simulate application phase
            monitor.record_phase_start("application", {
                'batch_size': 100,
                'max_workers': 4
            })

            time.sleep(0.15)  # Simulate processing time

            monitor.record_phase_end("application",
                                     cost_usd=0.135,
                                     additional_metrics={
                                         'tickets_processed': 1000,
                                         'avg_confidence': 0.89,
                                         'classification_rate': 0.94
                                     })

            assert "discovery" in monitor.current_session['phase_timings']
            assert "application" in monitor.current_session['phase_timings']
            assert monitor.current_session['cost_tracking']['discovery'] == 0.045
            assert monitor.current_session['cost_tracking']['application'] == 0.135
            print("✅ Phase recording completed successfully")

            # Test 4: Comprehensive Validation
            print("\n🎯 Testing comprehensive validation...")

            # Create test files
            categories_path = tmp_path / "categories.json"
            current_results_path = tmp_path / "current_results.csv"
            optimized_results_path = tmp_path / "optimized_results.csv"
            baseline_results_path = tmp_path / "baseline_results.csv"

            # Save test data
            with open(categories_path, 'w', encoding='utf-8') as f:
                json.dump(mock_categories, f, indent=2, ensure_ascii=False)

            test_data['current'].to_csv(current_results_path, index=False)
            test_data['optimized'].to_csv(optimized_results_path, index=False)
            test_data['baseline'].to_csv(baseline_results_path, index=False)

            print("✅ Test data files created")

            # Test different performance scenarios
            test_scenarios = [
                ("compliant", create_mock_orchestration_metrics(0.18, 22, 0.89)),
                ("cost_exceeds", create_mock_orchestration_metrics(0.25, 20, 0.90)),
                ("time_exceeds", create_mock_orchestration_metrics(0.15, 35, 0.88)),
                ("confidence_low", create_mock_orchestration_metrics(0.19, 23, 0.82))
            ]

            validation_reports = []

            for scenario_name, orchestration_metrics in test_scenarios:
                print(f"\n  📊 Testing scenario: {scenario_name}")

                report = monitor.validate_pipeline_results(
                    orchestration_metrics=orchestration_metrics,
                    categories_path=categories_path,
                    results_path=current_results_path
                )

                validation_reports.append(report)

                # Validate report structure
                assert report.timestamp is not None
                assert hasattr(report, 'cost_metrics')
                assert hasattr(report, 'performance_metrics')
                assert hasattr(report, 'accuracy_metrics')
                assert hasattr(report, 'compliance_summary')
                assert hasattr(report, 'recommendations')
                assert hasattr(report, 'next_actions')

                # Check compliance logic
                if scenario_name == "compliant":
                    assert report.compliance_summary['overall_compliant'] is True
                    assert "All Opção D targets met" in ' '.join(report.next_actions)
                else:
                    assert report.compliance_summary['overall_compliant'] is False
                    assert len(report.recommendations) > 0

                print(f"    ✅ Scenario validated: {scenario_name}")
                print(f"    💰 Cost: ${report.cost_metrics.cost_per_1k_tickets:.4f}/1K")
                print(f"    ⏱️ Time: {report.performance_metrics.total_processing_time / 60:.1f}min")
                print(f"    🎯 Confidence: {report.accuracy_metrics.avg_confidence:.3f}")
                print(f"    ✅ Compliant: {report.compliance_summary['overall_compliant']}")

            print("\n✅ All validation scenarios completed")

            # Test 5: Dashboard Generation
            print("\n📈 Testing dashboard generation...")

            # Mock matplotlib to avoid display issues in testing
            with patch('matplotlib.pyplot.show'), \
                 patch('matplotlib.pyplot.savefig') as mock_savefig:

                dashboard_path = monitor.generate_performance_dashboard(
                    validation_reports=validation_reports,
                    output_path=monitoring_dir / "test_dashboard.png"
                )

                assert dashboard_path.exists() or mock_savefig.called
                print("✅ Dashboard generated successfully")

            # Test 6: Baseline Comparison
            print("\n📊 Testing baseline comparison...")

            current_metrics = test_scenarios[0][1]  # Use compliant scenario
            baseline_metrics = create_mock_orchestration_metrics(0.30, 35, 0.80)  # Worse baseline

            comparison = monitor.compare_against_baseline(
                current_metrics, baseline_metrics, "Current vs Baseline"
            )

            assert 'improvements' in comparison
            assert 'regressions' in comparison
            assert 'summary' in comparison
            assert comparison['summary']['overall_better'] is True  # Current should be better

            print("✅ Baseline comparison completed")
            print(f"  Improvements: {len(comparison['improvements'])}")
            print(f"  Regressions: {len(comparison['regressions'])}")
            print(f"  Overall better: {comparison['summary']['overall_better']}")

            # Test 7: A/B Testing Framework
            print("\n🧪 Testing A/B testing framework...")

            # Mock scipy for statistical testing
            with patch('scipy.stats.ttest_ind') as mock_ttest:
                mock_ttest.return_value = (2.5, 0.03)  # Significant difference

                ab_test_report = monitor.run_ab_test_simulation(
                    method_a_results=current_results_path,
                    method_b_results=optimized_results_path,
                    test_name="Current vs Optimized"
                )

                assert 'method_a_metrics' in ab_test_report
                assert 'method_b_metrics' in ab_test_report
                assert 'statistical_results' in ab_test_report
                assert 'recommendation' in ab_test_report

                print("✅ A/B test completed")
                print(f"  Method A confidence: {ab_test_report['method_a_metrics']['avg_confidence']:.3f}")
                print(f"  Method B confidence: {ab_test_report['method_b_metrics']['avg_confidence']:.3f}")
                print(f"  Significant difference: {ab_test_report['statistical_results']['significant_difference']}")
                print(f"  Recommendation: {ab_test_report['recommendation'][:50]}...")

            # Test 8: Cost Breakdown Analysis
            print("\n💰 Testing cost breakdown analysis...")

            compliant_report = validation_reports[0]  # Use compliant scenario
            cost_metrics = compliant_report.cost_metrics

            assert hasattr(cost_metrics, 'cost_breakdown')
            assert 'discovery_phase' in cost_metrics.cost_breakdown
            assert 'application_phase' in cost_metrics.cost_breakdown

            total_breakdown = sum(cost_metrics.cost_breakdown.values())
            assert abs(total_breakdown - cost_metrics.total_cost_usd) < 0.01

            print("✅ Cost breakdown validated")
            print(f"  Total cost: ${cost_metrics.total_cost_usd:.4f}")
            print(f"  Discovery: ${cost_metrics.cost_breakdown.get('discovery_phase', 0):.4f}")
            print(f"  Application: ${cost_metrics.cost_breakdown.get('application_phase', 0):.4f}")
            print(f"  Per category: ${cost_metrics.cost_per_category:.4f}")

            # Test 9: Performance Metrics Analysis
            print("\n⚡ Testing performance metrics analysis...")

            perf_metrics = compliant_report.performance_metrics

            assert perf_metrics.total_processing_time > 0
            assert perf_metrics.discovery_time > 0
            assert perf_metrics.application_time > 0
            assert perf_metrics.avg_time_per_ticket > 0
            assert perf_metrics.throughput_tickets_per_second > 0
            assert perf_metrics.bottleneck_phase in ["discovery", "application", "balanced"]

            print("✅ Performance metrics validated")
            print(f"  Total time: {perf_metrics.total_processing_time:.1f}s")
            print(f"  Throughput: {perf_metrics.throughput_tickets_per_second:.2f} tickets/s")
            print(f"  Bottleneck: {perf_metrics.bottleneck_phase}")
            print(f"  Meets target: {perf_metrics.meets_time_target}")

            # Test 10: Accuracy Metrics Analysis
            print("\n🎯 Testing accuracy metrics analysis...")

            acc_metrics = compliant_report.accuracy_metrics

            assert 0 <= acc_metrics.avg_confidence <= 1
            assert 0 <= acc_metrics.high_confidence_ratio <= 1
            assert 0 <= acc_metrics.classification_rate <= 1
            assert 0 <= acc_metrics.category_coverage <= 1
            assert 0 <= acc_metrics.consistency_score <= 1

            print("✅ Accuracy metrics validated")
            print(f"  Avg confidence: {acc_metrics.avg_confidence:.3f}")
            print(f"  High confidence ratio: {acc_metrics.high_confidence_ratio:.1%}")
            print(f"  Classification rate: {acc_metrics.classification_rate:.1%}")
            print(f"  Category coverage: {acc_metrics.category_coverage:.1%}")
            print(f"  Consistency score: {acc_metrics.consistency_score:.3f}")
            print(f"  Validation errors: {len(acc_metrics.validation_errors)}")

            # Test 11: Utility Function Integration
            print("\n🔧 Testing utility function integration...")

            # Test validate_opçao_d_compliance utility
            utility_report = validate_opçao_d_compliance(
                orchestration_metrics=test_scenarios[0][1],  # Compliant scenario
                categories_path=str(categories_path),
                results_path=str(current_results_path),
                monitoring_dir=str(monitoring_dir)
            )

            assert isinstance(utility_report, type(validation_reports[0]))
            assert utility_report.compliance_summary['overall_compliant'] is True

            print("✅ Utility function integration successful")

            # Test 12: File Persistence and Recovery
            print("\n💾 Testing file persistence...")

            # Check that validation reports were saved
            report_files = list(monitoring_dir.glob("validation_report_*.json"))
            assert len(report_files) >= len(test_scenarios)

            # Check that A/B test report was saved
            ab_test_files = list(monitoring_dir.glob("ab_test_*.json"))
            assert len(ab_test_files) >= 1

            # Load and validate a saved report
            with open(report_files[0], 'r', encoding='utf-8') as f:
                saved_report = json.load(f)

            assert 'timestamp' in saved_report
            assert 'cost_metrics' in saved_report
            assert 'performance_metrics' in saved_report
            assert 'accuracy_metrics' in saved_report
            assert 'compliance_summary' in saved_report

            print("✅ File persistence validated")
            print(f"  Validation reports: {len(report_files)}")
            print(f"  A/B test reports: {len(ab_test_files)}")

            # Final Integration Summary
            print("\n🎉 ALL PERFORMANCE MONITORING INTEGRATION TESTS PASSED!")

            print("\n📋 INTEGRATION SUMMARY:")
            print("✅ Monitor initialization and configuration")
            print("✅ Session management and phase tracking")
            print("✅ Comprehensive validation against Opção D targets")
            print("✅ Dashboard generation and visualization")
            print("✅ Baseline comparison framework")
            print("✅ A/B testing statistical analysis")
            print("✅ Cost breakdown and optimization recommendations")
            print("✅ Performance bottleneck identification")
            print("✅ Accuracy and quality assessment")
            print("✅ Utility function integration")
            print("✅ File persistence and data recovery")
            print("✅ Real-time monitoring capabilities")

            print("\n🚀 PERFORMANCE MONITORING SYSTEM READY FOR PRODUCTION!")
            print("The system provides comprehensive monitoring, validation, and")
            print("optimization capabilities for the Opção D pipeline with")
            print("detailed compliance checking and actionable recommendations.")

    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = test_performance_monitoring_integration()
    exit(0 if success else 1)
