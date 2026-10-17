"""
Visualization Agent - Creates visualizations and dashboards.

Generates charts, dashboards, and reports from analysis results.
"""
import time
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
from agents.base_agent import BaseAgent
from utils.config import config
from utils.logger import get_logger


class VisualizationAgent(BaseAgent):
    """Creates visualizations and dashboards from insights."""

    def __init__(self, agent_id: str = "visualization"):
        """Initialize visualization agent."""
        super().__init__(agent_id)
        self.logger = get_logger(__name__)

        agent_config = config.get_agent_config('visualization')
        self.export_formats = agent_config.get('export_formats', ['html', 'png'])
        self.theme = agent_config.get('theme', 'plotly_white')
        self.output_path = Path(config.get('output_path', './output'))
        self.output_path.mkdir(parents=True, exist_ok=True)

        self.logger.info("Visualization Agent initialized")

    def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process visualization task.

        Args:
            task: Task containing data and insights to visualize

        Returns:
            Visualization results with file paths
        """
        task_id = task.get('task_id', 'unknown')
        self._log_task_start(task_id)
        start_time = time.time()

        try:
            input_data = task.get('input_data', {})

            # Extract data and insights
            if 'output' in input_data:
                data = input_data['output'].get('data', [])
                insights = input_data['output'].get('insights', [])
                analysis_results = input_data['output'].get('analysis_results', {})
            else:
                data = input_data.get('data', [])
                insights = input_data.get('insights', [])
                analysis_results = input_data.get('analysis_results', {})

            # Create visualizations
            visualizations = []

            # Create dashboard
            dashboard_path = self.create_dashboard(data, insights, analysis_results)
            if dashboard_path:
                visualizations.append({
                    'type': 'dashboard',
                    'path': str(dashboard_path),
                    'format': 'html'
                })

            # Create individual charts
            charts = self.generate_charts(data, analysis_results)
            visualizations.extend(charts)

            # Create report
            report_path = self.create_report(insights, visualizations)
            if report_path:
                visualizations.append({
                    'type': 'report',
                    'path': str(report_path),
                    'format': 'html'
                })

            execution_time = time.time() - start_time
            self._log_task_complete(task_id, True)

            self.logger.info(f"Created {len(visualizations)} visualizations")

            return self._create_success_response(
                output={
                    'visualizations': visualizations,
                    'dashboard_url': f"file://{dashboard_path}" if dashboard_path else None,
                    'total_visualizations': len(visualizations)
                },
                message=f"Created {len(visualizations)} visualizations",
                execution_time=execution_time
            )

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Visualization failed: {e}")
            self._log_task_complete(task_id, False)
            return self._create_error_response(str(e), execution_time=execution_time)

    def create_dashboard(self, data: Any, insights: List[Dict],
                        analysis_results: Dict[str, Any]) -> Optional[Path]:
        """
        Create interactive dashboard.

        Args:
            data: Raw data
            insights: List of insights
            analysis_results: Analysis results

        Returns:
            Path to dashboard HTML file
        """
        try:
            from plotly.subplots import make_subplots

            # Convert data to DataFrame if needed
            if isinstance(data, list) and len(data) > 0:
                df = pd.DataFrame(data)
            else:
                self.logger.warning("No data available for dashboard")
                return None

            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Data Overview', 'Key Metrics', 'Distribution', 'Trends'),
                specs=[[{'type': 'table'}, {'type': 'indicator'}],
                       [{'type': 'bar'}, {'type': 'scatter'}]]
            )

            # 1. Data overview table
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                summary = df[numeric_cols].describe().round(2)
                fig.add_trace(
                    go.Table(
                        header=dict(values=['Metric'] + list(summary.columns)),
                        cells=dict(values=[summary.index] + [summary[col] for col in summary.columns])
                    ),
                    row=1, col=1
                )

            # 2. Key metric indicator
            analysis_type = analysis_results.get('analysis_type', 'unknown')
            if analysis_type == 'classification':
                accuracy = analysis_results.get('results', {}).get('accuracy', 0)
                fig.add_trace(
                    go.Indicator(
                        mode="gauge+number+delta",
                        value=accuracy * 100,
                        title={'text': "Model Accuracy (%)"},
                        domain={'x': [0, 1], 'y': [0, 1]},
                        gauge={'axis': {'range': [None, 100]},
                               'bar': {'color': "darkblue"},
                               'threshold': {
                                   'line': {'color': "red", 'width': 4},
                                   'thickness': 0.75,
                                   'value': 80
                               }}
                    ),
                    row=1, col=2
                )

            # 3. Distribution chart
            if len(numeric_cols) > 0:
                col_to_plot = numeric_cols[0]
                fig.add_trace(
                    go.Histogram(x=df[col_to_plot], name=col_to_plot, nbinsx=30),
                    row=2, col=1
                )

            # 4. Scatter plot for relationships
            if len(numeric_cols) >= 2:
                fig.add_trace(
                    go.Scatter(
                        x=df[numeric_cols[0]],
                        y=df[numeric_cols[1]],
                        mode='markers',
                        name='Correlation'
                    ),
                    row=2, col=2
                )

            # Update layout
            fig.update_layout(
                title_text="Multi-Agent Analytics Dashboard",
                showlegend=True,
                height=800,
                template=self.theme
            )

            # Save dashboard
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            dashboard_path = self.output_path / f"dashboard_{timestamp}.html"
            fig.write_html(str(dashboard_path))

            self.logger.info(f"Dashboard created: {dashboard_path}")
            return dashboard_path

        except Exception as e:
            self.logger.error(f"Dashboard creation failed: {e}")
            return None

    def generate_charts(self, data: Any, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate individual charts based on data type.

        Args:
            data: Data to visualize
            analysis_results: Analysis results

        Returns:
            List of chart information
        """
        charts = []

        try:
            # Convert to DataFrame
            if isinstance(data, list) and len(data) > 0:
                df = pd.DataFrame(data)
            else:
                return charts

            analysis_type = analysis_results.get('analysis_type', 'unknown')

            # Chart 1: Feature importance (for ML models)
            if analysis_type in ['classification', 'regression']:
                feature_importance = analysis_results.get('results', {}).get('feature_importance', {})
                if feature_importance:
                    chart_path = self._create_feature_importance_chart(feature_importance)
                    if chart_path:
                        charts.append({
                            'type': 'feature_importance',
                            'path': str(chart_path),
                            'format': 'html'
                        })

            # Chart 2: Distribution plots for numeric columns
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                chart_path = self._create_distribution_chart(df, numeric_cols[:3])
                if chart_path:
                    charts.append({
                        'type': 'distribution',
                        'path': str(chart_path),
                        'format': 'html'
                    })

            # Chart 3: Correlation heatmap
            if len(numeric_cols) > 1:
                chart_path = self._create_correlation_heatmap(df[numeric_cols])
                if chart_path:
                    charts.append({
                        'type': 'correlation',
                        'path': str(chart_path),
                        'format': 'html'
                    })

        except Exception as e:
            self.logger.error(f"Chart generation failed: {e}")

        return charts

    def _create_feature_importance_chart(self, feature_importance: Dict[str, float]) -> Optional[Path]:
        """Create feature importance bar chart."""
        try:
            features = list(feature_importance.keys())
            importance = list(feature_importance.values())

            fig = go.Figure([go.Bar(x=importance, y=features, orientation='h')])
            fig.update_layout(
                title="Feature Importance",
                xaxis_title="Importance Score",
                yaxis_title="Features",
                template=self.theme,
                height=400
            )

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            chart_path = self.output_path / f"feature_importance_{timestamp}.html"
            fig.write_html(str(chart_path))

            return chart_path

        except Exception as e:
            self.logger.error(f"Feature importance chart failed: {e}")
            return None

    def _create_distribution_chart(self, df: pd.DataFrame, columns: List[str]) -> Optional[Path]:
        """Create distribution chart for numeric columns."""
        try:
            from plotly.subplots import make_subplots

            num_cols = len(columns)
            fig = make_subplots(rows=1, cols=num_cols, subplot_titles=columns)

            for idx, col in enumerate(columns, 1):
                fig.add_trace(
                    go.Histogram(x=df[col], name=col, nbinsx=30),
                    row=1, col=idx
                )

            fig.update_layout(
                title="Data Distributions",
                showlegend=False,
                template=self.theme,
                height=400
            )

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            chart_path = self.output_path / f"distributions_{timestamp}.html"
            fig.write_html(str(chart_path))

            return chart_path

        except Exception as e:
            self.logger.error(f"Distribution chart failed: {e}")
            return None

    def _create_correlation_heatmap(self, df: pd.DataFrame) -> Optional[Path]:
        """Create correlation heatmap."""
        try:
            corr_matrix = df.corr()

            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0
            ))

            fig.update_layout(
                title="Feature Correlation Heatmap",
                template=self.theme,
                height=500,
                width=600
            )

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            chart_path = self.output_path / f"correlation_{timestamp}.html"
            fig.write_html(str(chart_path))

            return chart_path

        except Exception as e:
            self.logger.error(f"Correlation heatmap failed: {e}")
            return None

    def create_report(self, insights: List[Dict], visualizations: List[Dict]) -> Optional[Path]:
        """
        Create HTML report with insights and visualizations.

        Args:
            insights: List of insights
            visualizations: List of visualization paths

        Returns:
            Path to report file
        """
        try:
            html_parts = [
                "<!DOCTYPE html>",
                "<html>",
                "<head>",
                "<title>Multi-Agent Analytics Report</title>",
                "<style>",
                "body { font-family: Arial, sans-serif; margin: 40px; }",
                "h1 { color: #2c3e50; }",
                "h2 { color: #34495e; margin-top: 30px; }",
                ".insight { background: #f8f9fa; padding: 15px; margin: 10px 0; border-left: 4px solid #3498db; }",
                ".insight-title { font-weight: bold; color: #2c3e50; }",
                ".impact-HIGH { border-left-color: #e74c3c; }",
                ".impact-MEDIUM { border-left-color: #f39c12; }",
                ".impact-LOW { border-left-color: #27ae60; }",
                ".timestamp { color: #7f8c8d; font-size: 0.9em; }",
                "</style>",
                "</head>",
                "<body>",
                f"<h1>Multi-Agent Analytics Report</h1>",
                f"<p class='timestamp'>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>",
                "<h2>Key Insights</h2>"
            ]

            # Add insights
            for insight in insights[:10]:  # Top 10
                impact = insight.get('impact', 'MEDIUM')
                html_parts.append(f"<div class='insight impact-{impact}'>")
                html_parts.append(f"<div class='insight-title'>{insight.get('title', 'Insight')}</div>")
                html_parts.append(f"<p>{insight.get('description', '')}</p>")
                if insight.get('recommendation'):
                    html_parts.append(f"<p><strong>Recommendation:</strong> {insight['recommendation']}</p>")
                html_parts.append(f"<p><em>Impact: {impact} | Urgency: {insight.get('urgency', 'N/A')}</em></p>")
                html_parts.append("</div>")

            # Add visualizations section
            html_parts.append("<h2>Visualizations</h2>")
            html_parts.append("<p>Interactive visualizations have been generated:</p>")
            html_parts.append("<ul>")
            for viz in visualizations:
                if viz['type'] != 'report':  # Don't link to self
                    html_parts.append(f"<li><a href='{Path(viz['path']).name}'>{viz['type'].replace('_', ' ').title()}</a></li>")
            html_parts.append("</ul>")

            html_parts.extend(["</body>", "</html>"])

            # Write report
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_path = self.output_path / f"report_{timestamp}.html"

            with open(report_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(html_parts))

            self.logger.info(f"Report created: {report_path}")
            return report_path

        except Exception as e:
            self.logger.error(f"Report creation failed: {e}")
            return None

    def export_visualization(self, fig: go.Figure, filename: str, format: str = 'html') -> Optional[Path]:
        """
        Export visualization to file.

        Args:
            fig: Plotly figure
            filename: Output filename
            format: Export format (html, png, pdf)

        Returns:
            Path to exported file
        """
        try:
            output_path = self.output_path / filename

            if format == 'html':
                fig.write_html(str(output_path))
            elif format == 'png':
                fig.write_image(str(output_path))
            elif format == 'pdf':
                fig.write_image(str(output_path), format='pdf')
            else:
                self.logger.error(f"Unsupported format: {format}")
                return None

            return output_path

        except Exception as e:
            self.logger.error(f"Export failed: {e}")
            return None
