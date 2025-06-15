#!/usr/bin/env python3
"""
PlantsWorld API Module
Provides REST API endpoints for data access and integration with other projects.
"""

import json
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
from config import config
import io
import zipfile

# Configure logging
logger = logging.getLogger(__name__)

class PlantsWorldAPI:
    """API class for PlantsWorld dashboard"""
    
    def __init__(self, app=None):
        self.app = app
        self.datasets = {}
        if app:
            self.init_app(app)
    
    def init_app(self, app):
        """Initialize the API with Flask app"""
        if config.INTEGRATION_CONFIG['cors_enabled']:
            CORS(app)
        
        # Load datasets
        self.load_datasets()
        
        # Register routes
        self.register_routes(app)
    
    def load_datasets(self):
        """Load all datasets for API access"""
        for dataset_key, dataset_config in config.DATASETS_CONFIG.items():
            try:
                latest_file = config.get_latest_dataset_file(dataset_key)
                if latest_file and latest_file.exists():
                    df = pd.read_csv(latest_file)
                    self.datasets[dataset_key] = df
                    logger.info(f"API: Loaded {dataset_config['display_name']}: {len(df)} records")
                else:
                    self.datasets[dataset_key] = pd.DataFrame()
            except Exception as e:
                logger.error(f"API: Error loading {dataset_config['display_name']}: {e}")
                self.datasets[dataset_key] = pd.DataFrame()
    
    def register_routes(self, app):
        """Register API routes"""
        api_prefix = config.INTEGRATION_CONFIG['api_prefix']
        
        @app.route(f'{api_prefix}/health')
        def health_check():
            """Health check endpoint"""
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'version': '1.0.0',
                'datasets_loaded': len([k for k, v in self.datasets.items() if not v.empty])
            })
        
        @app.route(f'{api_prefix}/datasets')
        def list_datasets():
            """List available datasets"""
            datasets_info = {}
            for key, df in self.datasets.items():
                if key in config.DATASETS_CONFIG:
                    datasets_info[key] = {
                        'name': config.DATASETS_CONFIG[key]['display_name'],
                        'description': config.DATASETS_CONFIG[key]['description'],
                        'records': len(df),
                        'columns': list(df.columns) if not df.empty else [],
                        'last_updated': datetime.now().isoformat()
                    }
            return jsonify(datasets_info)
        
        @app.route(f'{api_prefix}/datasets/<dataset_key>')
        def get_dataset(dataset_key):
            """Get specific dataset"""
            if dataset_key not in self.datasets:
                return jsonify({'error': 'Dataset not found'}), 404
            
            df = self.datasets[dataset_key]
            if df.empty:
                return jsonify({'error': 'Dataset is empty'}), 404
            
            # Handle pagination
            page = request.args.get('page', 1, type=int)
            per_page = min(request.args.get('per_page', 100, type=int), 1000)
            
            start_idx = (page - 1) * per_page
            end_idx = start_idx + per_page
            
            paginated_df = df.iloc[start_idx:end_idx]
            
            return jsonify({
                'data': paginated_df.to_dict('records'),
                'pagination': {
                    'page': page,
                    'per_page': per_page,
                    'total_records': len(df),
                    'total_pages': (len(df) + per_page - 1) // per_page
                },
                'metadata': {
                    'dataset': config.DATASETS_CONFIG.get(dataset_key, {}).get('display_name', dataset_key),
                    'columns': list(df.columns),
                    'description': config.DATASETS_CONFIG.get(dataset_key, {}).get('description', '')
                }
            })
        
        @app.route(f'{api_prefix}/datasets/<dataset_key>/export')
        def export_dataset(dataset_key):
            """Export dataset in various formats"""
            if dataset_key not in self.datasets:
                return jsonify({'error': 'Dataset not found'}), 404
            
            df = self.datasets[dataset_key]
            if df.empty:
                return jsonify({'error': 'Dataset is empty'}), 404
            
            format_type = request.args.get('format', 'csv').lower()
            
            if format_type not in config.INTEGRATION_CONFIG['export_formats']:
                return jsonify({'error': f'Format {format_type} not supported'}), 400
            
            # Create filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{dataset_key}_{timestamp}.{format_type}"
            
            if format_type == 'csv':
                output = io.StringIO()
                df.to_csv(output, index=False)
                output.seek(0)
                return send_file(
                    io.BytesIO(output.getvalue().encode()),
                    mimetype='text/csv',
                    as_attachment=True,
                    download_name=filename
                )
            
            elif format_type == 'json':
                output = io.StringIO()
                df.to_json(output, orient='records', indent=2)
                output.seek(0)
                return send_file(
                    io.BytesIO(output.getvalue().encode()),
                    mimetype='application/json',
                    as_attachment=True,
                    download_name=filename
                )
            
            elif format_type == 'xlsx':
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df.to_excel(writer, sheet_name=dataset_key, index=False)
                output.seek(0)
                return send_file(
                    output,
                    mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    as_attachment=True,
                    download_name=filename
                )
        
        @app.route(f'{api_prefix}/stats')
        def get_stats():
            """Get overall statistics"""
            stats = {
                'total_datasets': len(self.datasets),
                'total_records': sum(len(df) for df in self.datasets.values()),
                'datasets': {}
            }
            
            for key, df in self.datasets.items():
                if not df.empty:
                    stats['datasets'][key] = {
                        'records': len(df),
                        'columns': len(df.columns),
                        'memory_usage': df.memory_usage(deep=True).sum()
                    }
            
            return jsonify(stats)
        
        @app.route(f'{api_prefix}/search')
        def search_data():
            """Search across datasets"""
            query = request.args.get('q', '').strip()
            dataset_key = request.args.get('dataset')
            
            if not query:
                return jsonify({'error': 'Query parameter q is required'}), 400
            
            results = {}
            
            datasets_to_search = [dataset_key] if dataset_key else self.datasets.keys()
            
            for key in datasets_to_search:
                if key in self.datasets and not self.datasets[key].empty:
                    df = self.datasets[key]
                    
                    # Simple text search across all string columns
                    string_cols = df.select_dtypes(include=['object']).columns
                    matches = pd.DataFrame()
                    
                    for col in string_cols:
                        mask = df[col].astype(str).str.contains(query, case=False, na=False)
                        matches = pd.concat([matches, df[mask]])
                    
                    if not matches.empty:
                        matches = matches.drop_duplicates()
                        results[key] = {
                            'matches': len(matches),
                            'data': matches.head(50).to_dict('records')  # Limit to 50 results
                        }
            
            return jsonify({
                'query': query,
                'results': results,
                'total_matches': sum(r['matches'] for r in results.values())
            })

# Create API instance
api = PlantsWorldAPI()

def create_api_app():
    """Create standalone API Flask app"""
    app = Flask(__name__)
    api.init_app(app)
    return app

if __name__ == '__main__':
    # Run standalone API server
    app = create_api_app()
    app.run(debug=config.DEBUG, host=config.HOST, port=config.PORT + 1)