"""
Analyzer Agent - Performs ML analysis and statistical analysis.

Capabilities: classification, regression, clustering, sentiment analysis, time series forecasting
"""
import time
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from agents.base_agent import BaseAgent
from utils.config import config
from utils.logger import get_logger


class AnalyzerAgent(BaseAgent):
    """Performs machine learning and statistical analysis."""

    def __init__(self, agent_id: str = "analyzer"):
        """Initialize analyzer agent."""
        super().__init__(agent_id)
        self.logger = get_logger(__name__)

        agent_config = config.get_agent_config('analyzer')
        self.models = agent_config.get('models', {})
        self.use_gpu = config.get('use_gpu', False)
        self.test_size = agent_config.get('test_size', 0.2)

        self.logger.info("Analyzer Agent initialized")

    def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process analysis task.

        Args:
            task: Task containing data to analyze

        Returns:
            Analysis results
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

            # Determine analysis type
            analysis_type = task.get('analysis_type', self._determine_analysis_type(data))

            # Perform analysis
            result = self.analyze(data, analysis_type)

            execution_time = time.time() - start_time
            self._log_task_complete(task_id, result['status'] == 'success')

            if result['status'] == 'success':
                return self._create_success_response(
                    output=result['output'],
                    message=f"Analysis completed: {analysis_type}",
                    execution_time=execution_time
                )
            else:
                return result

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Analysis failed: {e}")
            self._log_task_complete(task_id, False)
            return self._create_error_response(str(e), execution_time=execution_time)

    def _determine_analysis_type(self, data: Any) -> str:
        """Automatically determine the type of analysis to perform."""
        if isinstance(data, list) and len(data) > 0:
            if isinstance(data[0], dict):
                # Check for text data
                if any('text' in k or 'cleaned_text' in k for k in data[0].keys()):
                    return 'sentiment'
                else:
                    return 'classification'

        return 'descriptive'

    def analyze(self, data: Any, analysis_type: str) -> Dict[str, Any]:
        """
        Main analysis method.

        Args:
            data: Data to analyze
            analysis_type: Type of analysis (classification, regression, sentiment, etc.)

        Returns:
            Analysis results
        """
        try:
            if analysis_type == 'classification':
                return self.perform_classification(data)
            elif analysis_type == 'regression':
                return self.perform_regression(data)
            elif analysis_type == 'clustering':
                return self.perform_clustering(data)
            elif analysis_type == 'sentiment':
                return self.sentiment_analysis(data)
            elif analysis_type == 'forecasting':
                return self.forecast_timeseries(data)
            elif analysis_type == 'descriptive':
                return self.descriptive_statistics(data)
            else:
                return {
                    'status': 'error',
                    'message': f'Unknown analysis type: {analysis_type}'
                }

        except Exception as e:
            self.logger.error(f"Analysis error: {e}")
            return {'status': 'error', 'message': str(e)}

    def perform_classification(self, data: Any) -> Dict[str, Any]:
        """
        Perform classification analysis.

        Args:
            data: Data with features and target

        Returns:
            Classification results
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

        try:
            # Convert to DataFrame
            if isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = data

            # Identify target column (assume last column or 'target')
            if 'target' in df.columns:
                target_col = 'target'
            elif 'label' in df.columns:
                target_col = 'label'
            else:
                # Use last column as target
                target_col = df.columns[-1]

            # Prepare features and target
            X = df.drop(columns=[target_col])
            y = df[target_col]

            # Select only numeric features
            X = X.select_dtypes(include=[np.number])

            if X.empty:
                return {
                    'status': 'error',
                    'message': 'No numeric features found for classification'
                }

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=42
            )

            # Train model
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)

            # Feature importance
            feature_importance = dict(zip(X.columns, model.feature_importances_))
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

            # Get top 5 features
            top_features = dict(sorted_features[:5])

            self.logger.info(f"Classification completed with accuracy: {accuracy:.2%}")

            return {
                'status': 'success',
                'output': {
                    'analysis_type': 'classification',
                    'model': 'RandomForestClassifier',
                    'results': {
                        'accuracy': float(accuracy),
                        'num_samples': len(X),
                        'num_features': len(X.columns),
                        'num_classes': int(y.nunique()),
                        'feature_importance': {k: float(v) for k, v in top_features.items()}
                    },
                    'insights': [
                        f"Model accuracy: {accuracy:.2%}",
                        f"Top feature: {sorted_features[0][0]} ({sorted_features[0][1]:.3f})",
                        f"Trained on {len(X_train)} samples, tested on {len(X_test)} samples"
                    ],
                    'confidence': 'HIGH' if accuracy > 0.85 else 'MEDIUM' if accuracy > 0.7 else 'LOW',
                    'timestamp': datetime.now().isoformat()
                }
            }

        except Exception as e:
            return {'status': 'error', 'message': f'Classification error: {str(e)}'}

    def perform_regression(self, data: Any) -> Dict[str, Any]:
        """
        Perform regression analysis.

        Args:
            data: Data with features and target

        Returns:
            Regression results
        """
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

        try:
            # Convert to DataFrame
            if isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = data

            # Identify target column
            if 'target' in df.columns:
                target_col = 'target'
            else:
                target_col = df.columns[-1]

            # Prepare features and target
            X = df.drop(columns=[target_col])
            y = df[target_col]

            # Select only numeric features
            X = X.select_dtypes(include=[np.number])

            if X.empty:
                return {'status': 'error', 'message': 'No numeric features found'}

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=42
            )

            # Train model
            model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)

            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # Feature importance
            feature_importance = dict(zip(X.columns, model.feature_importances_))
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            top_features = dict(sorted_features[:5])

            self.logger.info(f"Regression completed with R² score: {r2:.3f}")

            return {
                'status': 'success',
                'output': {
                    'analysis_type': 'regression',
                    'model': 'GradientBoostingRegressor',
                    'results': {
                        'r2_score': float(r2),
                        'rmse': float(rmse),
                        'mae': float(mae),
                        'mse': float(mse),
                        'feature_importance': {k: float(v) for k, v in top_features.items()}
                    },
                    'insights': [
                        f"R² score: {r2:.3f} (variance explained)",
                        f"RMSE: {rmse:.3f}",
                        f"Top predictor: {sorted_features[0][0]}"
                    ],
                    'confidence': 'HIGH' if r2 > 0.8 else 'MEDIUM' if r2 > 0.6 else 'LOW',
                    'timestamp': datetime.now().isoformat()
                }
            }

        except Exception as e:
            return {'status': 'error', 'message': f'Regression error: {str(e)}'}

    def perform_clustering(self, data: Any) -> Dict[str, Any]:
        """
        Perform clustering analysis.

        Args:
            data: Data to cluster

        Returns:
            Clustering results
        """
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        try:
            # Convert to DataFrame
            if isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = data

            # Select numeric columns
            X = df.select_dtypes(include=[np.number])

            if X.empty:
                return {'status': 'error', 'message': 'No numeric features found'}

            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Determine optimal number of clusters (simple heuristic)
            n_clusters = min(5, max(2, len(X) // 20))

            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(X_scaled)

            # Add clusters to original data
            df['cluster'] = clusters

            # Analyze cluster distribution
            cluster_counts = pd.Series(clusters).value_counts().sort_index()

            self.logger.info(f"Clustering completed with {n_clusters} clusters")

            return {
                'status': 'success',
                'output': {
                    'analysis_type': 'clustering',
                    'model': 'KMeans',
                    'results': {
                        'num_clusters': n_clusters,
                        'cluster_distribution': cluster_counts.to_dict(),
                        'inertia': float(kmeans.inertia_)
                    },
                    'insights': [
                        f"Identified {n_clusters} distinct clusters",
                        f"Largest cluster: {cluster_counts.idxmax()} with {cluster_counts.max()} samples"
                    ],
                    'confidence': 'MEDIUM',
                    'timestamp': datetime.now().isoformat()
                }
            }

        except Exception as e:
            return {'status': 'error', 'message': f'Clustering error: {str(e)}'}

    def sentiment_analysis(self, data: Any) -> Dict[str, Any]:
        """
        Perform sentiment analysis on text data.

        Args:
            data: Text data to analyze

        Returns:
            Sentiment analysis results
        """
        try:
            # Extract texts
            if isinstance(data, list):
                if len(data) > 0 and isinstance(data[0], dict):
                    texts = [d.get('cleaned_text', d.get('text', str(d))) for d in data]
                else:
                    texts = [str(t) for t in data]
            else:
                texts = [str(data)]

            # Simple sentiment analysis using TextBlob as fallback
            try:
                from textblob import TextBlob

                sentiments = []
                for text in texts:
                    blob = TextBlob(text)
                    polarity = blob.sentiment.polarity

                    if polarity > 0.1:
                        sentiment = 'POSITIVE'
                    elif polarity < -0.1:
                        sentiment = 'NEGATIVE'
                    else:
                        sentiment = 'NEUTRAL'

                    sentiments.append({
                        'text': text[:100],  # First 100 chars
                        'sentiment': sentiment,
                        'polarity': polarity,
                        'subjectivity': blob.sentiment.subjectivity
                    })

                # Aggregate results
                positive_count = sum(1 for s in sentiments if s['sentiment'] == 'POSITIVE')
                negative_count = sum(1 for s in sentiments if s['sentiment'] == 'NEGATIVE')
                neutral_count = sum(1 for s in sentiments if s['sentiment'] == 'NEUTRAL')

                overall_sentiment = 'POSITIVE' if positive_count > len(texts) / 2 else \
                                  'NEGATIVE' if negative_count > len(texts) / 2 else 'NEUTRAL'

                avg_polarity = np.mean([s['polarity'] for s in sentiments])

                self.logger.info(f"Sentiment analysis completed: {overall_sentiment}")

                return {
                    'status': 'success',
                    'output': {
                        'analysis_type': 'sentiment',
                        'model': 'TextBlob',
                        'results': {
                            'overall_sentiment': overall_sentiment,
                            'positive_count': positive_count,
                            'negative_count': negative_count,
                            'neutral_count': neutral_count,
                            'positive_percentage': float(positive_count / len(texts) * 100),
                            'average_polarity': float(avg_polarity),
                            'total_analyzed': len(texts)
                        },
                        'sample_sentiments': sentiments[:5],  # First 5 samples
                        'insights': [
                            f"Overall sentiment: {overall_sentiment}",
                            f"{positive_count} positive, {negative_count} negative, {neutral_count} neutral",
                            f"Average polarity: {avg_polarity:.3f}"
                        ],
                        'confidence': 'MEDIUM',
                        'timestamp': datetime.now().isoformat()
                    }
                }

            except ImportError:
                # Fallback: simple rule-based sentiment
                return self._simple_sentiment_analysis(texts)

        except Exception as e:
            return {'status': 'error', 'message': f'Sentiment analysis error: {str(e)}'}

    def _simple_sentiment_analysis(self, texts: List[str]) -> Dict[str, Any]:
        """Simple rule-based sentiment analysis fallback."""
        positive_words = ['good', 'great', 'excellent', 'amazing', 'love', 'best', 'happy', 'wonderful']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'worst', 'horrible', 'poor', 'disappointing']

        results = []
        for text in texts:
            text_lower = text.lower()
            pos_count = sum(text_lower.count(word) for word in positive_words)
            neg_count = sum(text_lower.count(word) for word in negative_words)

            if pos_count > neg_count:
                sentiment = 'POSITIVE'
            elif neg_count > pos_count:
                sentiment = 'NEGATIVE'
            else:
                sentiment = 'NEUTRAL'

            results.append(sentiment)

        positive_count = results.count('POSITIVE')
        negative_count = results.count('NEGATIVE')
        overall_sentiment = 'POSITIVE' if positive_count > len(texts) / 2 else \
                          'NEGATIVE' if negative_count > len(texts) / 2 else 'NEUTRAL'

        return {
            'status': 'success',
            'output': {
                'analysis_type': 'sentiment',
                'model': 'rule-based',
                'results': {
                    'overall_sentiment': overall_sentiment,
                    'positive_percentage': float(positive_count / len(texts) * 100)
                },
                'insights': [f"Overall sentiment: {overall_sentiment}"],
                'confidence': 'LOW',
                'timestamp': datetime.now().isoformat()
            }
        }

    def forecast_timeseries(self, data: Any) -> Dict[str, Any]:
        """
        Perform time series forecasting.

        Args:
            data: Time series data

        Returns:
            Forecast results
        """
        # Placeholder for time series forecasting
        return {
            'status': 'success',
            'output': {
                'analysis_type': 'forecasting',
                'model': 'placeholder',
                'results': {
                    'note': 'Time series forecasting not fully implemented'
                },
                'insights': ['Time series forecasting requires specialized implementation'],
                'confidence': 'LOW',
                'timestamp': datetime.now().isoformat()
            }
        }

    def descriptive_statistics(self, data: Any) -> Dict[str, Any]:
        """
        Compute descriptive statistics.

        Args:
            data: Data to analyze

        Returns:
            Descriptive statistics
        """
        try:
            # Convert to DataFrame
            if isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = data

            # Get numeric columns
            numeric_df = df.select_dtypes(include=[np.number])

            if numeric_df.empty:
                return {
                    'status': 'success',
                    'output': {
                        'analysis_type': 'descriptive',
                        'results': {'note': 'No numeric columns found'},
                        'insights': ['Dataset contains no numeric data for statistical analysis'],
                        'confidence': 'HIGH',
                        'timestamp': datetime.now().isoformat()
                    }
                }

            # Compute statistics
            stats = numeric_df.describe().to_dict()

            # Additional insights
            correlations = numeric_df.corr()

            insights = [
                f"Dataset contains {len(df)} records with {len(df.columns)} columns",
                f"{len(numeric_df.columns)} numeric columns analyzed"
            ]

            return {
                'status': 'success',
                'output': {
                    'analysis_type': 'descriptive',
                    'results': {
                        'statistics': stats,
                        'num_records': len(df),
                        'num_columns': len(df.columns),
                        'num_numeric_columns': len(numeric_df.columns)
                    },
                    'insights': insights,
                    'confidence': 'HIGH',
                    'timestamp': datetime.now().isoformat()
                }
            }

        except Exception as e:
            return {'status': 'error', 'message': f'Descriptive statistics error: {str(e)}'}

    def detect_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect patterns in data.

        Args:
            data: DataFrame to analyze

        Returns:
            Pattern detection results
        """
        patterns = []

        # Check for trends in numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            # Simple trend detection
            values = data[col].dropna()
            if len(values) > 1:
                trend = np.polyfit(range(len(values)), values, 1)[0]
                if abs(trend) > values.std() * 0.1:
                    direction = 'increasing' if trend > 0 else 'decreasing'
                    patterns.append({
                        'column': col,
                        'pattern': f'{direction} trend',
                        'strength': float(abs(trend))
                    })

        return {
            'patterns_detected': len(patterns),
            'patterns': patterns
        }
