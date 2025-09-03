#!/usr/bin/env python3
"""
Comprehensive NeMo-Guardrails Analysis Runner

This script runs all analysis modules and generates a complete report
covering use cases, performance, security, and recommendations.
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Dict, Any, List

# Import our analysis modules
from nemo_guardrails_analysis import NeMoGuardrailsAnalyzer
from prompt_injection_tests import PromptInjectionTester
from performance_benchmarks import PerformanceBenchmarker


class ComprehensiveAnalyzer:
    """Main class to run comprehensive analysis"""
    
    def __init__(self):
        self.start_time = time.time()
        self.reports = {}
    
    async def run_all_analyses(self) -> Dict[str, Any]:
        """Run all analysis modules"""
        print("🚀 Starting Comprehensive NeMo-Guardrails Analysis")
        print("=" * 80)
        print(f"Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        # 1. Basic Analysis
        print("\n📋 Phase 1: Basic Analysis and Use Cases")
        print("-" * 50)
        basic_analyzer = NeMoGuardrailsAnalyzer()
        basic_report = await basic_analyzer.run_full_analysis()
        self.reports['basic_analysis'] = basic_report
        
        # 2. Prompt Injection Security Tests
        print("\n🔒 Phase 2: Prompt Injection Security Tests")
        print("-" * 50)
        security_tester = PromptInjectionTester()
        security_report = security_tester.generate_security_report()
        # Run the actual tests
        await security_tester.run_all_injection_tests()
        security_report = security_tester.generate_security_report()
        self.reports['security_analysis'] = security_report
        
        # 3. Performance Benchmarks
        print("\n⚡ Phase 3: Performance Benchmarks")
        print("-" * 50)
        performance_benchmarker = PerformanceBenchmarker()
        performance_report = await performance_benchmarker.run_full_benchmark_suite()
        self.reports['performance_analysis'] = performance_report
        
        # 4. Generate comprehensive report
        print("\n📊 Phase 4: Generating Comprehensive Report")
        print("-" * 50)
        comprehensive_report = self.generate_comprehensive_report()
        
        # 5. Save all reports
        self.save_all_reports(comprehensive_report)
        
        # 6. Print final summary
        self.print_final_summary(comprehensive_report)
        
        return comprehensive_report
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive report combining all analyses"""
        end_time = time.time()
        total_analysis_time = end_time - self.start_time
        
        # Extract key metrics from all reports
        basic_metrics = self.reports['basic_analysis'].get('performance_metrics', {})
        security_metrics = self.reports['security_analysis'].get('test_summary', {})
        performance_metrics = self.reports['performance_analysis'].get('benchmark_metrics', {})
        
        comprehensive_report = {
            "analysis_metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_analysis_time_seconds": total_analysis_time,
                "nemoguardrails_version": "0.15.0",
                "analysis_scope": "comprehensive"
            },
            "executive_summary": self.generate_executive_summary(),
            "detailed_analyses": {
                "basic_analysis": self.reports['basic_analysis'],
                "security_analysis": self.reports['security_analysis'],
                "performance_analysis": self.reports['performance_analysis']
            },
            "key_metrics": {
                "overall_performance": {
                    "avg_response_time": performance_metrics.get('avg_response_time', 0),
                    "throughput": performance_metrics.get('throughput', 0),
                    "success_rate": (performance_metrics.get('successful_tests', 0) / 
                                   max(performance_metrics.get('total_tests', 1), 1)) * 100
                },
                "security_effectiveness": {
                    "prompt_injection_block_rate": security_metrics.get('overall_block_rate', 0),
                    "critical_attacks_blocked": security_metrics.get('blocked_tests', 0),
                    "security_level": self.reports['security_analysis'].get('security_assessment', {}).get('security_level', 'Unknown')
                },
                "use_case_coverage": {
                    "total_use_cases_analyzed": len(self.reports['basic_analysis'].get('use_cases', {})),
                    "security_measures_identified": len(self.reports['basic_analysis'].get('security_measures', {})),
                    "drawbacks_identified": len(self.reports['basic_analysis'].get('drawbacks', {}))
                }
            },
            "recommendations": self.generate_comprehensive_recommendations(),
            "risk_assessment": self.generate_risk_assessment(),
            "deployment_guidelines": self.generate_deployment_guidelines()
        }
        
        return comprehensive_report
    
    def generate_executive_summary(self) -> Dict[str, Any]:
        """Generate executive summary of the analysis"""
        security_level = self.reports['security_analysis'].get('security_assessment', {}).get('security_level', 'Unknown')
        performance_level = self.reports['performance_analysis'].get('performance_assessment', {}).get('performance_level', 'Unknown')
        
        # Overall assessment
        if security_level in ['Excellent', 'Good'] and performance_level in ['Excellent', 'Good']:
            overall_assessment = "NeMo-Guardrails is suitable for production deployment with proper configuration."
        elif security_level in ['Excellent', 'Good'] or performance_level in ['Excellent', 'Good']:
            overall_assessment = "NeMo-Guardrails shows strengths in some areas but needs improvement in others."
        else:
            overall_assessment = "NeMo-Guardrails requires significant optimization before production deployment."
        
        return {
            "overall_assessment": overall_assessment,
            "security_rating": security_level,
            "performance_rating": performance_level,
            "production_readiness": "Ready" if security_level in ['Excellent', 'Good'] and performance_level in ['Excellent', 'Good'] else "Needs Improvement",
            "key_strengths": [
                "Comprehensive prompt injection protection",
                "Flexible configuration system",
                "Multiple LLM provider support",
                "Custom action capabilities"
            ],
            "key_concerns": [
                "Performance overhead in some scenarios",
                "Complexity of configuration",
                "Potential for false positives",
                "Resource usage requirements"
            ]
        }
    
    def generate_comprehensive_recommendations(self) -> Dict[str, List[str]]:
        """Generate comprehensive recommendations based on all analyses"""
        recommendations = {
            "immediate_actions": [
                "Implement NeMo-Guardrails for production AI applications requiring safety",
                "Configure appropriate security guardrails based on use case",
                "Set up monitoring and alerting for security events",
                "Establish regular security testing procedures"
            ],
            "performance_optimization": [
                "Optimize guardrails configuration for your specific use case",
                "Implement caching for frequently used safety checks",
                "Monitor and optimize resource usage",
                "Consider horizontal scaling for high-traffic applications"
            ],
            "security_enhancement": [
                "Regularly update guardrails against new attack vectors",
                "Implement custom actions for domain-specific requirements",
                "Conduct regular penetration testing",
                "Monitor for new prompt injection techniques"
            ],
            "operational_considerations": [
                "Train team on NeMo-Guardrails configuration and maintenance",
                "Establish incident response procedures for security events",
                "Implement proper logging and audit trails",
                "Plan for regular updates and maintenance"
            ],
            "long_term_strategy": [
                "Consider multiple AI safety frameworks for defense in depth",
                "Invest in continuous security research and development",
                "Build internal expertise in AI safety and security",
                "Participate in AI safety community and research"
            ]
        }
        
        return recommendations
    
    def generate_risk_assessment(self) -> Dict[str, Any]:
        """Generate risk assessment based on analysis results"""
        security_metrics = self.reports['security_analysis'].get('test_summary', {})
        performance_metrics = self.reports['performance_analysis'].get('benchmark_metrics', {})
        
        # Calculate risk scores
        security_risk = 100 - security_metrics.get('overall_block_rate', 0)
        performance_risk = min(100, performance_metrics.get('avg_response_time', 0) * 50)  # Scale response time to risk
        
        overall_risk = (security_risk + performance_risk) / 2
        
        risk_level = "Low" if overall_risk < 25 else "Medium" if overall_risk < 50 else "High"
        
        return {
            "overall_risk_level": risk_level,
            "overall_risk_score": overall_risk,
            "security_risk_score": security_risk,
            "performance_risk_score": performance_risk,
            "risk_factors": [
                "Prompt injection vulnerabilities" if security_risk > 25 else None,
                "Performance bottlenecks" if performance_risk > 25 else None,
                "Configuration complexity" if overall_risk > 50 else None,
                "Resource requirements" if performance_metrics.get('avg_memory_usage', 0) > 100 else None
            ],
            "mitigation_strategies": [
                "Implement comprehensive testing procedures",
                "Use multiple validation layers",
                "Monitor system performance continuously",
                "Regular security audits and updates"
            ]
        }
    
    def generate_deployment_guidelines(self) -> Dict[str, Any]:
        """Generate deployment guidelines based on analysis"""
        return {
            "pre_deployment": [
                "Conduct thorough testing with realistic workloads",
                "Configure appropriate security guardrails",
                "Set up monitoring and alerting systems",
                "Train operational team on NeMo-Guardrails",
                "Establish incident response procedures"
            ],
            "deployment": [
                "Deploy in staging environment first",
                "Gradual rollout with monitoring",
                "Implement proper logging and audit trails",
                "Set up performance monitoring",
                "Configure automated backups"
            ],
            "post_deployment": [
                "Monitor security events and performance metrics",
                "Regular security testing and updates",
                "Performance optimization based on real usage",
                "Regular review of configuration and rules",
                "Continuous improvement based on feedback"
            ],
            "maintenance": [
                "Regular updates to guardrails rules",
                "Performance monitoring and optimization",
                "Security testing and vulnerability assessment",
                "Documentation updates and team training",
                "Backup and disaster recovery procedures"
            ]
        }
    
    def save_all_reports(self, comprehensive_report: Dict[str, Any]):
        """Save all reports to files"""
        # Save comprehensive report
        with open("comprehensive_nemo_guardrails_report.json", "w") as f:
            json.dump(comprehensive_report, f, indent=2)
        
        # Save individual reports
        with open("basic_analysis_report.json", "w") as f:
            json.dump(self.reports['basic_analysis'], f, indent=2)
        
        with open("security_analysis_report.json", "w") as f:
            json.dump(self.reports['security_analysis'], f, indent=2)
        
        with open("performance_analysis_report.json", "w") as f:
            json.dump(self.reports['performance_analysis'], f, indent=2)
        
        print("📄 All reports saved to JSON files")
    
    def print_final_summary(self, report: Dict[str, Any]):
        """Print final comprehensive summary"""
        print("\n" + "=" * 80)
        print("📊 COMPREHENSIVE NEMO-GUARDRAILS ANALYSIS SUMMARY")
        print("=" * 80)
        
        metadata = report['analysis_metadata']
        summary = report['executive_summary']
        metrics = report['key_metrics']
        risk = report['risk_assessment']
        
        print(f"Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total analysis time: {metadata['total_analysis_time_seconds']:.1f} seconds")
        print(f"NeMo-Guardrails version: {metadata['nemoguardrails_version']}")
        
        print(f"\n🎯 EXECUTIVE SUMMARY")
        print(f"Overall Assessment: {summary['overall_assessment']}")
        print(f"Security Rating: {summary['security_rating']}")
        print(f"Performance Rating: {summary['performance_rating']}")
        print(f"Production Readiness: {summary['production_readiness']}")
        
        print(f"\n📊 KEY METRICS")
        perf_metrics = metrics['overall_performance']
        sec_metrics = metrics['security_effectiveness']
        use_metrics = metrics['use_case_coverage']
        
        print(f"Performance:")
        print(f"  Average Response Time: {perf_metrics['avg_response_time']:.3f}s")
        print(f"  Throughput: {perf_metrics['throughput']:.2f} req/s")
        print(f"  Success Rate: {perf_metrics['success_rate']:.1f}%")
        
        print(f"Security:")
        print(f"  Prompt Injection Block Rate: {sec_metrics['prompt_injection_block_rate']:.1f}%")
        print(f"  Security Level: {sec_metrics['security_level']}")
        print(f"  Critical Attacks Blocked: {sec_metrics['critical_attacks_blocked']}")
        
        print(f"Coverage:")
        print(f"  Use Cases Analyzed: {use_metrics['total_use_cases_analyzed']}")
        print(f"  Security Measures: {use_metrics['security_measures_identified']}")
        print(f"  Drawbacks Identified: {use_metrics['drawbacks_identified']}")
        
        print(f"\n⚠️  RISK ASSESSMENT")
        print(f"Overall Risk Level: {risk['overall_risk_level']}")
        print(f"Overall Risk Score: {risk['overall_risk_score']:.1f}/100")
        print(f"Security Risk Score: {risk['security_risk_score']:.1f}/100")
        print(f"Performance Risk Score: {risk['performance_risk_score']:.1f}/100")
        
        print(f"\n💡 KEY RECOMMENDATIONS")
        recommendations = report['recommendations']
        for category, recs in recommendations.items():
            if recs:  # Only show non-empty categories
                print(f"\n{category.replace('_', ' ').title()}:")
                for i, rec in enumerate(recs[:3], 1):  # Show top 3
                    print(f"  {i}. {rec}")
        
        print(f"\n📄 REPORTS GENERATED")
        print(f"  - comprehensive_nemo_guardrails_report.json (Main report)")
        print(f"  - basic_analysis_report.json (Use cases and features)")
        print(f"  - security_analysis_report.json (Security testing)")
        print(f"  - performance_analysis_report.json (Performance benchmarks)")
        
        print(f"\n🎉 Analysis Complete!")
        print("=" * 80)


async def main():
    """Main function to run comprehensive analysis"""
    analyzer = ComprehensiveAnalyzer()
    report = await analyzer.run_all_analyses()
    return report


if __name__ == "__main__":
    asyncio.run(main())
