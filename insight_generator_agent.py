"""
Insight Generator Agent - Generates natural language insights and recommendations.

Converts analytical results into actionable business insights.
"""
import time
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from agents.base_agent import BaseAgent
from utils.config import config
from utils.logger import get_logger


class InsightGeneratorAgent(BaseAgent):
    """Generates insights and recommendations from analysis results."""

    def __init__(self, agent_id: str = "insight_generator"):
        """Initialize insight generator agent."""
        super().__init__(agent_id)
        self.logger = get_logger(__name__)

        agent_config = config.get_agent_config('insight_generator')
        self.generation_method = agent_config.get('generation_method', 'template')
        self.max_insights = agent_config.get('max_insights', 10)
        self.min_confidence = agent_config.get('min_confidence', 0.7)

        self.logger.info("Insight Generator Agent initialized")

    def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process insight generation task.

        Args:
            task: Task containing analysis results

        Returns:
            Generated insights
        """
        task_id = task.get('task_id', 'unknown')
        self._log_task_start(task_id)
        start_time = time.time()

        try:
            input_data = task.get('input_data', {})

            # Extract analysis results from previous agent's output
            if 'output' in input_data and 'results' in input_data['output']:
                analysis_results = input_data['output']
            elif 'results' in input_data:
                analysis_results = input_data
            else:
                analysis_results = input_data

            # Generate insights
            insights = self.generate_insights(analysis_results)

            # Create executive summary
            executive_summary = self.create_executive_summary(insights)

            # Prioritize recommendations
            recommendations = self.prioritize_recommendations(insights)

            execution_time = time.time() - start_time
            self._log_task_complete(task_id, True)

            return self._create_success_response(
                output={
                    'insights': insights,
                    'executive_summary': executive_summary,
                    'recommendations': recommendations,
                    'analysis_results': analysis_results
                },
                message="Insights generated successfully",
                execution_time=execution_time
            )

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Insight generation failed: {e}")
            self._log_task_complete(task_id, False)
            return self._create_error_response(str(e), execution_time=execution_time)

    def generate_insights(self, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate insights from analysis results.

        Args:
            analysis_results: Results from analyzer agent

        Returns:
            List of insights
        """
        insights = []

        # Determine analysis type
        analysis_type = analysis_results.get('analysis_type', 'unknown')

        # Generate insights based on type
        if analysis_type == 'classification':
            insights.extend(self._generate_classification_insights(analysis_results))
        elif analysis_type == 'regression':
            insights.extend(self._generate_regression_insights(analysis_results))
        elif analysis_type == 'clustering':
            insights.extend(self._generate_clustering_insights(analysis_results))
        elif analysis_type == 'sentiment':
            insights.extend(self._generate_sentiment_insights(analysis_results))
        elif analysis_type == 'descriptive':
            insights.extend(self._generate_descriptive_insights(analysis_results))
        else:
            insights.extend(self._generate_general_insights(analysis_results))

        # Limit insights
        insights = insights[:self.max_insights]

        # Add metadata
        for idx, insight in enumerate(insights):
            insight['insight_id'] = f"insight_{idx + 1}"
            insight['generated_at'] = datetime.now().isoformat()

        self.logger.info(f"Generated {len(insights)} insights")

        return insights

    def _generate_classification_insights(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate insights from classification results."""
        insights = []
        data = results.get('results', {})

        accuracy = data.get('accuracy', 0)
        confidence = results.get('confidence', 'MEDIUM')
        feature_importance = data.get('feature_importance', {})

        # Insight 1: Model performance
        insights.append({
            'type': 'predictive',
            'category': 'model_performance',
            'title': 'Classification Model Performance',
            'description': f"The classification model achieved {accuracy:.1%} accuracy with {confidence} confidence.",
            'impact': 'HIGH' if accuracy > 0.85 else 'MEDIUM',
            'urgency': 'MEDIUM',
            'actionable': True,
            'recommendation': 'Model is ready for production use' if accuracy > 0.85 else 'Consider improving model with more features or data'
        })

        # Insight 2: Key predictors
        if feature_importance:
            top_feature = list(feature_importance.keys())[0]
            top_importance = list(feature_importance.values())[0]

            insights.append({
                'type': 'diagnostic',
                'category': 'feature_importance',
                'title': 'Key Predictive Features Identified',
                'description': f"'{top_feature}' is the most important predictor, accounting for {top_importance:.1%} of model decisions.",
                'impact': 'HIGH',
                'urgency': 'LOW',
                'actionable': True,
                'recommendation': f"Focus data collection efforts on maintaining high quality '{top_feature}' data"
            })

        # Insight 3: Data quality requirement
        insights.append({
            'type': 'prescriptive',
            'category': 'data_requirements',
            'title': 'Maintain Data Quality for Predictions',
            'description': f"Model requires {data.get('num_features', 0)} features to make accurate predictions.",
            'impact': 'MEDIUM',
            'urgency': 'LOW',
            'actionable': True,
            'recommendation': 'Ensure all required features are collected and validated before making predictions'
        })

        return insights

    def _generate_regression_insights(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate insights from regression results."""
        insights = []
        data = results.get('results', {})

        r2_score = data.get('r2_score', 0)
        rmse = data.get('rmse', 0)
        feature_importance = data.get('feature_importance', {})

        # Insight 1: Predictive power
        variance_explained = r2_score * 100

        insights.append({
            'type': 'predictive',
            'category': 'model_performance',
            'title': 'Predictive Model Explains Variance',
            'description': f"The model explains {variance_explained:.1f}% of variance in the target variable with RMSE of {rmse:.2f}.",
            'impact': 'HIGH' if r2_score > 0.8 else 'MEDIUM',
            'urgency': 'MEDIUM',
            'actionable': True,
            'recommendation': 'Use model for forecasting and decision support' if r2_score > 0.8 else 'Improve model before production use'
        })

        # Insight 2: Key drivers
        if feature_importance:
            top_feature = list(feature_importance.keys())[0]

            insights.append({
                'type': 'diagnostic',
                'category': 'key_drivers',
                'title': 'Primary Driver Identified',
                'description': f"'{top_feature}' is the primary driver of outcomes, showing the strongest predictive relationship.",
                'impact': 'HIGH',
                'urgency': 'MEDIUM',
                'actionable': True,
                'recommendation': f"Monitor and optimize '{top_feature}' to improve target outcomes"
            })

        return insights

    def _generate_clustering_insights(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate insights from clustering results."""
        insights = []
        data = results.get('results', {})

        num_clusters = data.get('num_clusters', 0)
        cluster_dist = data.get('cluster_distribution', {})

        # Insight 1: Segment discovery
        insights.append({
            'type': 'descriptive',
            'category': 'segmentation',
            'title': f'{num_clusters} Distinct Segments Identified',
            'description': f"Analysis revealed {num_clusters} distinct segments in the data with varying characteristics.",
            'impact': 'HIGH',
            'urgency': 'MEDIUM',
            'actionable': True,
            'recommendation': 'Develop targeted strategies for each segment'
        })

        # Insight 2: Largest segment
        if cluster_dist:
            largest_cluster = max(cluster_dist, key=cluster_dist.get)
            largest_size = cluster_dist[largest_cluster]

            insights.append({
                'type': 'descriptive',
                'category': 'segment_distribution',
                'title': 'Dominant Segment Identified',
                'description': f"Segment {largest_cluster} is the largest, containing {largest_size} records.",
                'impact': 'MEDIUM',
                'urgency': 'LOW',
                'actionable': True,
                'recommendation': f"Prioritize resources for the largest segment to maximize impact"
            })

        return insights

    def _generate_sentiment_insights(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate insights from sentiment analysis."""
        insights = []
        data = results.get('results', {})

        overall_sentiment = data.get('overall_sentiment', 'NEUTRAL')
        positive_pct = data.get('positive_percentage', 0)
        negative_count = data.get('negative_count', 0)
        total_analyzed = data.get('total_analyzed', 0)

        # Insight 1: Overall sentiment
        sentiment_desc = {
            'POSITIVE': 'favorable',
            'NEGATIVE': 'concerning',
            'NEUTRAL': 'mixed'
        }

        insights.append({
            'type': 'descriptive',
            'category': 'sentiment',
            'title': f'{overall_sentiment.title()} Sentiment Detected',
            'description': f"Overall sentiment is {sentiment_desc.get(overall_sentiment, 'unknown')} with {positive_pct:.1f}% positive responses out of {total_analyzed} analyzed.",
            'impact': 'HIGH' if overall_sentiment == 'NEGATIVE' else 'MEDIUM',
            'urgency': 'HIGH' if overall_sentiment == 'NEGATIVE' else 'LOW',
            'actionable': True,
            'recommendation': 'Address negative feedback immediately' if overall_sentiment == 'NEGATIVE' else 'Maintain current approach'
        })

        # Insight 2: Negative feedback action
        if negative_count > 0:
            insights.append({
                'type': 'prescriptive',
                'category': 'customer_satisfaction',
                'title': 'Negative Feedback Requires Attention',
                'description': f"{negative_count} negative responses detected that may indicate customer dissatisfaction.",
                'impact': 'HIGH',
                'urgency': 'HIGH',
                'actionable': True,
                'recommendation': 'Review negative feedback, identify root causes, and implement corrective actions'
            })

        return insights

    def _generate_descriptive_insights(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate insights from descriptive statistics."""
        insights = []
        data = results.get('results', {})

        num_records = data.get('num_records', 0)
        num_columns = data.get('num_columns', 0)

        insights.append({
            'type': 'descriptive',
            'category': 'data_overview',
            'title': 'Dataset Overview',
            'description': f"Dataset contains {num_records:,} records across {num_columns} dimensions.",
            'impact': 'MEDIUM',
            'urgency': 'LOW',
            'actionable': False,
            'recommendation': 'Proceed with deeper analysis to uncover patterns'
        })

        return insights

    def _generate_general_insights(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate general insights when type is unknown."""
        insights = []

        # Extract any insights already in results
        if 'insights' in results:
            for idx, insight_text in enumerate(results['insights']):
                insights.append({
                    'type': 'general',
                    'category': 'analysis',
                    'title': f'Finding {idx + 1}',
                    'description': insight_text,
                    'impact': 'MEDIUM',
                    'urgency': 'LOW',
                    'actionable': True,
                    'recommendation': 'Review finding and determine appropriate action'
                })

        return insights

    def create_executive_summary(self, insights: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create executive summary from insights.

        Args:
            insights: List of insights

        Returns:
            Executive summary
        """
        # Count insights by type
        insight_types = {}
        for insight in insights:
            itype = insight.get('type', 'general')
            insight_types[itype] = insight_types.get(itype, 0) + 1

        # Count by impact
        high_impact = len([i for i in insights if i.get('impact') == 'HIGH'])
        medium_impact = len([i for i in insights if i.get('impact') == 'MEDIUM'])

        # Count actionable insights
        actionable = len([i for i in insights if i.get('actionable', False)])

        # Generate summary text
        summary_text = f"""
Analysis completed successfully with {len(insights)} key findings identified.

Key Highlights:
- {high_impact} high-impact findings requiring attention
- {actionable} actionable recommendations generated
- Analysis spans {len(insight_types)} analytical dimensions

Top Priority: {insights[0]['title'] if insights else 'No insights generated'}

Recommendation: {'Immediate action required on high-priority items' if high_impact > 0 else 'Continue monitoring and optimization'}
        """.strip()

        return {
            'summary_text': summary_text,
            'total_insights': len(insights),
            'high_impact_count': high_impact,
            'medium_impact_count': medium_impact,
            'actionable_count': actionable,
            'insight_types': insight_types,
            'generated_at': datetime.now().isoformat()
        }

    def prioritize_recommendations(self, insights: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Prioritize recommendations by impact and urgency.

        Args:
            insights: List of insights

        Returns:
            Prioritized recommendations
        """
        # Filter to actionable insights only
        actionable_insights = [i for i in insights if i.get('actionable', False)]

        # Calculate priority score
        impact_scores = {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}
        urgency_scores = {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}

        for insight in actionable_insights:
            impact = impact_scores.get(insight.get('impact', 'LOW'), 1)
            urgency = urgency_scores.get(insight.get('urgency', 'LOW'), 1)

            # Priority score (impact weighted more heavily)
            insight['priority_score'] = (impact * 0.6) + (urgency * 0.4)

        # Sort by priority score
        prioritized = sorted(actionable_insights, key=lambda x: x['priority_score'], reverse=True)

        # Add priority labels
        for idx, insight in enumerate(prioritized):
            if idx < len(prioritized) * 0.3:
                insight['priority_label'] = 'CRITICAL'
            elif idx < len(prioritized) * 0.6:
                insight['priority_label'] = 'HIGH'
            else:
                insight['priority_label'] = 'MEDIUM'

        return prioritized

    def contextualize_insights(self, insights: List[Dict[str, Any]], domain: str) -> List[Dict[str, Any]]:
        """
        Add domain-specific context to insights.

        Args:
            insights: List of insights
            domain: Business domain (e.g., 'ecommerce', 'healthcare', 'finance')

        Returns:
            Contextualized insights
        """
        domain_contexts = {
            'ecommerce': {
                'customer_satisfaction': 'Customer lifetime value and retention',
                'sentiment': 'Product reviews and brand perception',
                'segmentation': 'Customer segments for targeted marketing'
            },
            'healthcare': {
                'model_performance': 'Patient outcome prediction accuracy',
                'key_drivers': 'Health risk factors',
                'data_requirements': 'Clinical data quality'
            },
            'finance': {
                'model_performance': 'Risk assessment accuracy',
                'key_drivers': 'Financial risk indicators',
                'segmentation': 'Customer risk profiles'
            }
        }

        context_map = domain_contexts.get(domain, {})

        for insight in insights:
            category = insight.get('category', '')
            if category in context_map:
                insight['domain_context'] = context_map[category]

        return insights

    def generate_insight_narrative(self, insights: List[Dict[str, Any]]) -> str:
        """
        Generate a narrative from insights.

        Args:
            insights: List of insights

        Returns:
            Narrative text
        """
        if not insights:
            return "No insights available."

        narrative_parts = [
            "# Analysis Insights Report\n",
            f"Generated on {datetime.now().strftime('%Y-%m-%d at %H:%M')}\n\n",
            "## Key Findings\n"
        ]

        for idx, insight in enumerate(insights[:5], 1):  # Top 5 insights
            narrative_parts.append(f"\n### {idx}. {insight['title']}\n")
            narrative_parts.append(f"{insight['description']}\n")
            narrative_parts.append(f"**Impact:** {insight['impact']} | **Urgency:** {insight['urgency']}\n")
            if insight.get('recommendation'):
                narrative_parts.append(f"**Recommendation:** {insight['recommendation']}\n")

        return "".join(narrative_parts)
