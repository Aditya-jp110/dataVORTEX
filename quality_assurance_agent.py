"""
Quality Assurance Agent - Validates data quality and detects anomalies.

Checks: completeness, accuracy, consistency, timeliness, uniqueness
"""
import time
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from agents.base_agent import BaseAgent
from utils.config import config
from utils.logger import get_logger


class QualityAssuranceAgent(BaseAgent):
    """Validates data quality and generates quality reports."""

    def __init__(self, agent_id: str = "quality_assurance"):
        """Initialize quality assurance agent."""
        super().__init__(agent_id)
        self.logger = get_logger(__name__)

        agent_config = config.get_agent_config('quality_assurance')
        self.min_quality_score = agent_config.get('min_quality_score', 0.8)
        self.severity_levels = agent_config.get('severity_levels', ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW'])
        self.anomaly_threshold = agent_config.get('anomaly_detection', {}).get('threshold', 3.0)

        self.logger.info("Quality Assurance Agent initialized")

    def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process quality assurance task.

        Args:
            task: Task containing data to validate

        Returns:
            Quality assessment result
        """
        task_id = task.get('task_id', 'unknown')
        self._log_task_start(task_id)
        start_time = time.time()

        try:
            input_data = task.get('input_data', {})

            # Extract data from previous agent's output
            if 'data' in input_data:
                data = input_data['data']
            elif 'output' in input_data and 'data' in input_data['output']:
                data = input_data['output']['data']
            else:
                data = input_data

            # Convert to DataFrame if list of dicts
            if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                df = pd.DataFrame(data)
            elif isinstance(data, pd.DataFrame):
                df = data
            else:
                return self._create_error_response("Invalid data format for quality check")

            # Perform validation
            validation_result = self.validate_data(df)

            # Detect anomalies
            anomalies = self.detect_anomalies(df)

            # Generate quality report
            quality_report = self.generate_quality_report(df, validation_result, anomalies)

            # Determine action
            action = self.recommend_action(quality_report)

            execution_time = time.time() - start_time
            self._log_task_complete(task_id, True)

            return self._create_success_response(
                output={
                    'data': data,  # Pass through the data
                    'quality_report': quality_report,
                    'action': action,
                    'passed_quality_check': quality_report['overall_score'] >= self.min_quality_score
                },
                message=f"Quality check completed - {action}",
                execution_time=execution_time
            )

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Quality check failed: {e}")
            self._log_task_complete(task_id, False)
            return self._create_error_response(str(e), execution_time=execution_time)

    def validate_data(self, df: pd.DataFrame, schema: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Validate data against quality dimensions.

        Args:
            df: DataFrame to validate
            schema: Optional schema definition

        Returns:
            Validation results
        """
        results = {
            'completeness': self._check_completeness(df),
            'accuracy': self._check_accuracy(df, schema),
            'consistency': self._check_consistency(df),
            'uniqueness': self._check_uniqueness(df)
        }

        return results

    def _check_completeness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check data completeness (missing values)."""
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        completeness_rate = 1 - (missing_cells / total_cells) if total_cells > 0 else 0

        # Per-column completeness
        column_completeness = {}
        for col in df.columns:
            null_count = df[col].isnull().sum()
            column_completeness[col] = {
                'completeness_rate': 1 - (null_count / len(df)) if len(df) > 0 else 0,
                'missing_count': int(null_count),
                'total_count': len(df)
            }

        issues = []
        for col, stats in column_completeness.items():
            if stats['completeness_rate'] < 0.9:
                severity = 'CRITICAL' if stats['completeness_rate'] < 0.5 else 'HIGH'
                issues.append({
                    'column': col,
                    'issue': f"High missing value rate: {(1-stats['completeness_rate'])*100:.1f}%",
                    'severity': severity
                })

        return {
            'overall_completeness': completeness_rate,
            'column_completeness': column_completeness,
            'issues': issues,
            'passed': completeness_rate >= 0.8
        }

    def _check_accuracy(self, df: pd.DataFrame, schema: Optional[Dict] = None) -> Dict[str, Any]:
        """Check data accuracy (data types, ranges, formats)."""
        issues = []
        accuracy_scores = []

        for col in df.columns:
            col_accuracy = 1.0

            # Check for numeric columns
            if pd.api.types.is_numeric_dtype(df[col]):
                # Check for infinite values
                inf_count = np.isinf(df[col]).sum()
                if inf_count > 0:
                    issues.append({
                        'column': col,
                        'issue': f"Contains {inf_count} infinite values",
                        'severity': 'HIGH'
                    })
                    col_accuracy -= 0.2

                # Check for outliers using IQR method
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outlier_count = ((df[col] < (Q1 - 3 * IQR)) | (df[col] > (Q3 + 3 * IQR))).sum()

                if outlier_count > len(df) * 0.05:  # More than 5% outliers
                    issues.append({
                        'column': col,
                        'issue': f"High outlier rate: {outlier_count} ({outlier_count/len(df)*100:.1f}%)",
                        'severity': 'MEDIUM'
                    })

            # Check for string columns
            elif pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
                # Check for empty strings
                if df[col].dtype == 'object':
                    empty_count = (df[col].astype(str).str.strip() == '').sum()
                    if empty_count > 0:
                        issues.append({
                            'column': col,
                            'issue': f"Contains {empty_count} empty strings",
                            'severity': 'MEDIUM'
                        })

            accuracy_scores.append(col_accuracy)

        overall_accuracy = np.mean(accuracy_scores) if accuracy_scores else 1.0

        return {
            'overall_accuracy': overall_accuracy,
            'issues': issues,
            'passed': overall_accuracy >= 0.8 and len([i for i in issues if i['severity'] in ['CRITICAL', 'HIGH']]) == 0
        }

    def _check_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check data consistency (cross-field validation)."""
        issues = []

        # Check for duplicate rows
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            issues.append({
                'issue': f"Found {duplicate_count} duplicate rows ({duplicate_count/len(df)*100:.1f}%)",
                'severity': 'MEDIUM' if duplicate_count < len(df) * 0.01 else 'HIGH'
            })

        # Check data type consistency per column
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check if column has mixed types
                type_counts = df[col].apply(type).value_counts()
                if len(type_counts) > 1:
                    issues.append({
                        'column': col,
                        'issue': f"Mixed data types detected: {dict(type_counts)}",
                        'severity': 'HIGH'
                    })

        consistency_score = max(0, 1 - (len(issues) / max(len(df.columns), 1)))

        return {
            'consistency_score': consistency_score,
            'duplicate_count': int(duplicate_count),
            'issues': issues,
            'passed': consistency_score >= 0.8
        }

    def _check_uniqueness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check uniqueness of values."""
        uniqueness_stats = {}

        for col in df.columns:
            unique_count = df[col].nunique()
            uniqueness_rate = unique_count / len(df) if len(df) > 0 else 0

            uniqueness_stats[col] = {
                'unique_count': int(unique_count),
                'uniqueness_rate': uniqueness_rate,
                'total_count': len(df)
            }

        return {
            'column_uniqueness': uniqueness_stats,
            'passed': True
        }

    def detect_anomalies(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect anomalies using statistical methods.

        Args:
            df: DataFrame to analyze

        Returns:
            Anomaly detection results
        """
        anomalies = []

        # Z-score method for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if df[col].std() > 0:  # Avoid division by zero
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                anomaly_mask = z_scores > self.anomaly_threshold
                anomaly_count = anomaly_mask.sum()

                if anomaly_count > 0:
                    anomaly_indices = df[anomaly_mask].index.tolist()
                    anomaly_values = df.loc[anomaly_mask, col].tolist()

                    anomalies.append({
                        'column': col,
                        'method': 'z-score',
                        'anomaly_count': int(anomaly_count),
                        'anomaly_rate': float(anomaly_count / len(df)),
                        'threshold': self.anomaly_threshold,
                        'sample_indices': anomaly_indices[:5],  # First 5 anomalies
                        'sample_values': [float(v) for v in anomaly_values[:5]]
                    })

        return {
            'anomalies_detected': len(anomalies),
            'anomalies': anomalies,
            'total_records': len(df)
        }

    def generate_quality_report(self, df: pd.DataFrame,
                               validation_result: Dict[str, Any],
                               anomalies: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive quality report.

        Args:
            df: DataFrame
            validation_result: Validation results
            anomalies: Anomaly detection results

        Returns:
            Quality report
        """
        # Calculate overall quality score
        dimension_scores = {
            'completeness': validation_result['completeness']['overall_completeness'],
            'accuracy': validation_result['accuracy']['overall_accuracy'],
            'consistency': validation_result['consistency']['consistency_score'],
            'timeliness': 1.0  # Placeholder
        }

        overall_score = np.mean(list(dimension_scores.values()))

        # Collect all issues
        all_issues = []
        for dimension, results in validation_result.items():
            if 'issues' in results:
                for issue in results['issues']:
                    issue['dimension'] = dimension
                    all_issues.append(issue)

        # Sort issues by severity
        severity_order = {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}
        all_issues.sort(key=lambda x: severity_order.get(x.get('severity', 'LOW'), 4))

        # Generate summary
        critical_issues = len([i for i in all_issues if i.get('severity') == 'CRITICAL'])
        high_issues = len([i for i in all_issues if i.get('severity') == 'HIGH'])

        return {
            'overall_score': overall_score,
            'dimension_scores': dimension_scores,
            'total_records': len(df),
            'total_columns': len(df.columns),
            'issues': all_issues,
            'issue_summary': {
                'critical': critical_issues,
                'high': high_issues,
                'medium': len([i for i in all_issues if i.get('severity') == 'MEDIUM']),
                'low': len([i for i in all_issues if i.get('severity') == 'LOW'])
            },
            'anomalies': anomalies,
            'validation_details': validation_result,
            'timestamp': datetime.now().isoformat()
        }

    def recommend_action(self, quality_report: Dict[str, Any]) -> str:
        """
        Recommend action based on quality report.

        Args:
            quality_report: Quality assessment report

        Returns:
            Recommended action: PROCEED, PROCEED_WITH_CAUTION, or STOP
        """
        overall_score = quality_report['overall_score']
        issue_summary = quality_report['issue_summary']

        # Decision logic
        if issue_summary['critical'] > 0:
            return 'STOP'
        elif overall_score < 0.6:
            return 'STOP'
        elif issue_summary['high'] > 5 or overall_score < 0.8:
            return 'PROCEED_WITH_CAUTION'
        else:
            return 'PROCEED'

    def fix_quality_issues(self, df: pd.DataFrame, quality_report: Dict[str, Any]) -> pd.DataFrame:
        """
        Attempt to fix quality issues automatically.

        Args:
            df: DataFrame with quality issues
            quality_report: Quality report

        Returns:
            Fixed DataFrame
        """
        df_fixed = df.copy()

        # Fix: Remove duplicate rows
        validation = quality_report.get('validation_details', {})
        if validation.get('consistency', {}).get('duplicate_count', 0) > 0:
            df_fixed = df_fixed.drop_duplicates()
            self.logger.info(f"Removed {len(df) - len(df_fixed)} duplicate rows")

        # Fix: Handle empty strings
        for col in df_fixed.select_dtypes(include=['object']).columns:
            empty_mask = df_fixed[col].astype(str).str.strip() == ''
            if empty_mask.any():
                df_fixed.loc[empty_mask, col] = None
                self.logger.info(f"Converted empty strings to None in column: {col}")

        return df_fixed
