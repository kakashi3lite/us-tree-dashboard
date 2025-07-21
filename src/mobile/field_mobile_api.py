#!/usr/bin/env python3
"""
Field Team Mobile Integration for Conservation Operations.
Optimized for Eli's workflow - offline capabilities, mobile-first design, and real-time field data sync.
"""

import asyncio
import logging
import json
import sqlite3
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import qrcode
import io
import base64
from PIL import Image
import gpxpy
import geopy.distance

# Configure logging for field operations
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/field_mobile.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class FieldObservation:
    """Field observation data structure"""
    observation_id: str
    team_member: str
    location_lat: float
    location_lon: float
    timestamp: datetime
    species_name: Optional[str] = None
    health_status: str = "good"  # good, fair, poor, dead
    dbh_cm: Optional[float] = None
    height_m: Optional[float] = None
    notes: Optional[str] = None
    photos: List[str] = None
    threats: List[str] = None
    management_actions: List[str] = None
    weather_conditions: Optional[str] = None
    accuracy_m: Optional[float] = None
    synced: bool = False

@dataclass
class FieldRoute:
    """Field team route tracking"""
    route_id: str
    team_member: str
    start_time: datetime
    end_time: Optional[datetime] = None
    waypoints: List[Tuple[float, float, datetime]] = None
    total_distance_km: float = 0.0
    observations_count: int = 0
    status: str = "active"  # active, completed, paused

@dataclass
class FieldAssignment:
    """Field work assignment"""
    assignment_id: str
    team_member: str
    area_name: str
    priority: str  # high, medium, low
    task_type: str  # survey, monitoring, maintenance, emergency
    coordinates: List[Tuple[float, float]] = None
    instructions: Optional[str] = None
    due_date: Optional[datetime] = None
    status: str = "assigned"  # assigned, in_progress, completed
    estimated_hours: Optional[float] = None

class FieldMobileAPI:
    """
    Mobile API for field team operations.
    Provides offline-capable endpoints for data collection and synchronization.
    """
    
    def __init__(self, data_dir: str = "data/field"):
        self.app = Flask(__name__)
        CORS(self.app)  # Enable CORS for mobile apps
        
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize SQLite database for offline storage
        self.db_path = self.data_dir / "field_mobile.db"
        self._init_database()
        
        # Setup API routes
        self._setup_routes()
        
        logger.info("📱 Field Mobile API initialized for conservation operations")
    
    def _init_database(self):
        """Initialize SQLite database for field data"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Field observations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS field_observations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    observation_id TEXT UNIQUE NOT NULL,
                    team_member TEXT NOT NULL,
                    location_lat REAL NOT NULL,
                    location_lon REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    species_name TEXT,
                    health_status TEXT DEFAULT 'good',
                    dbh_cm REAL,
                    height_m REAL,
                    notes TEXT,
                    photos TEXT,
                    threats TEXT,
                    management_actions TEXT,
                    weather_conditions TEXT,
                    accuracy_m REAL,
                    synced BOOLEAN DEFAULT FALSE
                )
            """)
            
            # Field routes table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS field_routes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    route_id TEXT UNIQUE NOT NULL,
                    team_member TEXT NOT NULL,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    waypoints TEXT,
                    total_distance_km REAL DEFAULT 0.0,
                    observations_count INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'active'
                )
            """)
            
            # Field assignments table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS field_assignments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    assignment_id TEXT UNIQUE NOT NULL,
                    team_member TEXT NOT NULL,
                    area_name TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    task_type TEXT NOT NULL,
                    coordinates TEXT,
                    instructions TEXT,
                    due_date TEXT,
                    status TEXT DEFAULT 'assigned',
                    estimated_hours REAL,
                    created_date TEXT NOT NULL
                )
            """)
            
            # Offline sync queue
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sync_queue (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    operation TEXT NOT NULL,
                    table_name TEXT NOT NULL,
                    record_id TEXT NOT NULL,
                    data TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    synced BOOLEAN DEFAULT FALSE
                )
            """)
            
            # Create indexes for performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_observations_team ON field_observations(team_member)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_observations_timestamp ON field_observations(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_routes_team ON field_routes(team_member)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_assignments_team ON field_assignments(team_member)")
            
            conn.commit()
    
    def _setup_routes(self):
        """Setup Flask API routes"""
        
        @self.app.route('/api/mobile/health', methods=['GET'])
        def health_check():
            """API health check endpoint"""
            return jsonify({
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "service": "field_mobile_api"
            })
        
        @self.app.route('/api/mobile/observations', methods=['POST'])
        def create_observation():
            """Create new field observation"""
            try:
                data = request.get_json()
                
                observation = FieldObservation(
                    observation_id=str(uuid.uuid4()),
                    team_member=data.get('team_member', 'unknown'),
                    location_lat=float(data['location_lat']),
                    location_lon=float(data['location_lon']),
                    timestamp=datetime.now(),
                    species_name=data.get('species_name'),
                    health_status=data.get('health_status', 'good'),
                    dbh_cm=data.get('dbh_cm'),
                    height_m=data.get('height_m'),
                    notes=data.get('notes'),
                    photos=data.get('photos', []),
                    threats=data.get('threats', []),
                    management_actions=data.get('management_actions', []),
                    weather_conditions=data.get('weather_conditions'),
                    accuracy_m=data.get('accuracy_m')
                )
                
                self._store_observation(observation)
                
                logger.info(f"📝 New field observation created: {observation.observation_id}")
                
                return jsonify({
                    "success": True,
                    "observation_id": observation.observation_id,
                    "message": "Observation created successfully"
                }), 201
                
            except Exception as e:
                logger.error(f"Failed to create observation: {e}")
                return jsonify({"success": False, "error": str(e)}), 400
        
        @self.app.route('/api/mobile/observations', methods=['GET'])
        def get_observations():
            """Get field observations (with optional filtering)"""
            try:
                team_member = request.args.get('team_member')
                days = int(request.args.get('days', 7))
                
                observations = self._get_observations(team_member, days)
                
                return jsonify({
                    "success": True,
                    "observations": observations,
                    "count": len(observations)
                })
                
            except Exception as e:
                logger.error(f"Failed to get observations: {e}")
                return jsonify({"success": False, "error": str(e)}), 400
        
        @self.app.route('/api/mobile/routes/start', methods=['POST'])
        def start_route():
            """Start new field route tracking"""
            try:
                data = request.get_json()
                
                route = FieldRoute(
                    route_id=str(uuid.uuid4()),
                    team_member=data.get('team_member', 'unknown'),
                    start_time=datetime.now(),
                    waypoints=[]
                )
                
                self._store_route(route)
                
                logger.info(f"🗺️ New field route started: {route.route_id}")
                
                return jsonify({
                    "success": True,
                    "route_id": route.route_id,
                    "message": "Route tracking started"
                }), 201
                
            except Exception as e:
                logger.error(f"Failed to start route: {e}")
                return jsonify({"success": False, "error": str(e)}), 400
        
        @self.app.route('/api/mobile/routes/<route_id>/waypoint', methods=['POST'])
        def add_waypoint(route_id):
            """Add waypoint to active route"""
            try:
                data = request.get_json()
                
                lat = float(data['latitude'])
                lon = float(data['longitude'])
                timestamp = datetime.now()
                
                self._add_waypoint(route_id, lat, lon, timestamp)
                
                return jsonify({
                    "success": True,
                    "message": "Waypoint added"
                })
                
            except Exception as e:
                logger.error(f"Failed to add waypoint: {e}")
                return jsonify({"success": False, "error": str(e)}), 400
        
        @self.app.route('/api/mobile/routes/<route_id>/complete', methods=['POST'])
        def complete_route(route_id):
            """Complete field route tracking"""
            try:
                self._complete_route(route_id)
                
                logger.info(f"✅ Field route completed: {route_id}")
                
                return jsonify({
                    "success": True,
                    "message": "Route completed"
                })
                
            except Exception as e:
                logger.error(f"Failed to complete route: {e}")
                return jsonify({"success": False, "error": str(e)}), 400
        
        @self.app.route('/api/mobile/assignments', methods=['GET'])
        def get_assignments():
            """Get field assignments for team member"""
            try:
                team_member = request.args.get('team_member')
                status = request.args.get('status', 'assigned')
                
                assignments = self._get_assignments(team_member, status)
                
                return jsonify({
                    "success": True,
                    "assignments": assignments,
                    "count": len(assignments)
                })
                
            except Exception as e:
                logger.error(f"Failed to get assignments: {e}")
                return jsonify({"success": False, "error": str(e)}), 400
        
        @self.app.route('/api/mobile/assignments/<assignment_id>/status', methods=['PUT'])
        def update_assignment_status(assignment_id):
            """Update assignment status"""
            try:
                data = request.get_json()
                new_status = data.get('status')
                
                self._update_assignment_status(assignment_id, new_status)
                
                return jsonify({
                    "success": True,
                    "message": "Assignment status updated"
                })
                
            except Exception as e:
                logger.error(f"Failed to update assignment: {e}")
                return jsonify({"success": False, "error": str(e)}), 400
        
        @self.app.route('/api/mobile/qr-code', methods=['POST'])
        def generate_qr_code():
            """Generate QR code for quick data entry"""
            try:
                data = request.get_json()
                
                # Create QR code data
                qr_data = {
                    "type": "field_location",
                    "location": {
                        "lat": data.get('latitude'),
                        "lon": data.get('longitude')
                    },
                    "assignment_id": data.get('assignment_id'),
                    "timestamp": datetime.now().isoformat()
                }
                
                # Generate QR code
                qr = qrcode.QRCode(
                    version=1,
                    error_correction=qrcode.constants.ERROR_CORRECT_L,
                    box_size=10,
                    border=4,
                )
                qr.add_data(json.dumps(qr_data))
                qr.make(fit=True)
                
                # Create QR code image
                img = qr.make_image(fill_color="black", back_color="white")
                img_buffer = io.BytesIO()
                img.save(img_buffer, format='PNG')
                img_buffer.seek(0)
                
                # Convert to base64 for mobile apps
                img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
                
                return jsonify({
                    "success": True,
                    "qr_code": f"data:image/png;base64,{img_base64}",
                    "data": qr_data
                })
                
            except Exception as e:
                logger.error(f"Failed to generate QR code: {e}")
                return jsonify({"success": False, "error": str(e)}), 400
        
        @self.app.route('/api/mobile/sync', methods=['POST'])
        def sync_data():
            """Synchronize offline data with main database"""
            try:
                # Get all unsynced records
                unsynced_count = self._sync_with_main_database()
                
                return jsonify({
                    "success": True,
                    "synced_records": unsynced_count,
                    "message": f"Synchronized {unsynced_count} records"
                })
                
            except Exception as e:
                logger.error(f"Failed to sync data: {e}")
                return jsonify({"success": False, "error": str(e)}), 400
        
        @self.app.route('/api/mobile/export/gpx/<route_id>', methods=['GET'])
        def export_route_gpx(route_id):
            """Export route as GPX file"""
            try:
                gpx_content = self._export_route_as_gpx(route_id)
                
                # Create in-memory file
                gpx_buffer = io.BytesIO()
                gpx_buffer.write(gpx_content.encode('utf-8'))
                gpx_buffer.seek(0)
                
                return send_file(
                    gpx_buffer,
                    as_attachment=True,
                    download_name=f"field_route_{route_id}.gpx",
                    mimetype="application/gpx+xml"
                )
                
            except Exception as e:
                logger.error(f"Failed to export GPX: {e}")
                return jsonify({"success": False, "error": str(e)}), 400
        
        @self.app.route('/api/mobile/offline-data', methods=['GET'])
        def get_offline_data():
            """Get essential data for offline operation"""
            try:
                offline_data = {
                    "species_list": self._get_common_species(),
                    "health_statuses": ["good", "fair", "poor", "dead"],
                    "threat_types": [
                        "disease", "insects", "drought", "flooding", "fire_damage",
                        "human_damage", "storm_damage", "construction", "pollution"
                    ],
                    "management_actions": [
                        "watering", "pruning", "treatment", "removal", "protection",
                        "monitoring", "fertilization", "pest_control"
                    ],
                    "weather_options": [
                        "sunny", "cloudy", "rainy", "windy", "fog", "snow"
                    ]
                }
                
                return jsonify({
                    "success": True,
                    "offline_data": offline_data
                })
                
            except Exception as e:
                logger.error(f"Failed to get offline data: {e}")
                return jsonify({"success": False, "error": str(e)}), 400
    
    def _store_observation(self, observation: FieldObservation):
        """Store field observation in database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO field_observations 
                (observation_id, team_member, location_lat, location_lon, timestamp,
                 species_name, health_status, dbh_cm, height_m, notes, photos,
                 threats, management_actions, weather_conditions, accuracy_m, synced)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                observation.observation_id,
                observation.team_member,
                observation.location_lat,
                observation.location_lon,
                observation.timestamp.isoformat(),
                observation.species_name,
                observation.health_status,
                observation.dbh_cm,
                observation.height_m,
                observation.notes,
                json.dumps(observation.photos or []),
                json.dumps(observation.threats or []),
                json.dumps(observation.management_actions or []),
                observation.weather_conditions,
                observation.accuracy_m,
                observation.synced
            ))
            conn.commit()
    
    def _get_observations(self, team_member: Optional[str] = None, days: int = 7) -> List[Dict]:
        """Get field observations with optional filtering"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Build query with optional filtering
            query = """
                SELECT * FROM field_observations 
                WHERE timestamp > datetime('now', '-{} days')
            """.format(days)
            
            params = []
            if team_member:
                query += " AND team_member = ?"
                params.append(team_member)
            
            query += " ORDER BY timestamp DESC"
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            # Convert to list of dictionaries
            columns = [description[0] for description in cursor.description]
            observations = []
            
            for row in rows:
                obs_dict = dict(zip(columns, row))
                # Parse JSON fields
                obs_dict['photos'] = json.loads(obs_dict.get('photos', '[]'))
                obs_dict['threats'] = json.loads(obs_dict.get('threats', '[]'))
                obs_dict['management_actions'] = json.loads(obs_dict.get('management_actions', '[]'))
                observations.append(obs_dict)
            
            return observations
    
    def _store_route(self, route: FieldRoute):
        """Store field route in database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO field_routes 
                (route_id, team_member, start_time, end_time, waypoints,
                 total_distance_km, observations_count, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                route.route_id,
                route.team_member,
                route.start_time.isoformat(),
                route.end_time.isoformat() if route.end_time else None,
                json.dumps(route.waypoints or []),
                route.total_distance_km,
                route.observations_count,
                route.status
            ))
            conn.commit()
    
    def _add_waypoint(self, route_id: str, lat: float, lon: float, timestamp: datetime):
        """Add waypoint to route and update distance"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get current waypoints
            cursor.execute("SELECT waypoints, total_distance_km FROM field_routes WHERE route_id = ?", (route_id,))
            result = cursor.fetchone()
            
            if result:
                current_waypoints = json.loads(result[0] or '[]')
                current_distance = result[1] or 0.0
                
                # Add new waypoint
                new_waypoint = [lat, lon, timestamp.isoformat()]
                current_waypoints.append(new_waypoint)
                
                # Calculate distance if previous waypoint exists
                if len(current_waypoints) > 1:
                    prev_waypoint = current_waypoints[-2]
                    distance = geopy.distance.geodesic(
                        (prev_waypoint[0], prev_waypoint[1]),
                        (lat, lon)
                    ).kilometers
                    current_distance += distance
                
                # Update route
                cursor.execute("""
                    UPDATE field_routes 
                    SET waypoints = ?, total_distance_km = ?
                    WHERE route_id = ?
                """, (json.dumps(current_waypoints), current_distance, route_id))
                
                conn.commit()
    
    def _complete_route(self, route_id: str):
        """Complete field route tracking"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Count observations made during this route
            cursor.execute("""
                SELECT COUNT(*) FROM field_observations o
                JOIN field_routes r ON o.team_member = r.team_member
                WHERE r.route_id = ? AND o.timestamp >= r.start_time
            """, (route_id,))
            
            observations_count = cursor.fetchone()[0]
            
            # Update route status
            cursor.execute("""
                UPDATE field_routes 
                SET status = 'completed', end_time = ?, observations_count = ?
                WHERE route_id = ?
            """, (datetime.now().isoformat(), observations_count, route_id))
            
            conn.commit()
    
    def _get_assignments(self, team_member: Optional[str] = None, status: str = "assigned") -> List[Dict]:
        """Get field assignments"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            query = "SELECT * FROM field_assignments WHERE status = ?"
            params = [status]
            
            if team_member:
                query += " AND team_member = ?"
                params.append(team_member)
            
            query += " ORDER BY priority DESC, due_date ASC"
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            columns = [description[0] for description in cursor.description]
            assignments = []
            
            for row in rows:
                assignment = dict(zip(columns, row))
                # Parse coordinates JSON
                if assignment.get('coordinates'):
                    assignment['coordinates'] = json.loads(assignment['coordinates'])
                assignments.append(assignment)
            
            return assignments
    
    def _update_assignment_status(self, assignment_id: str, new_status: str):
        """Update assignment status"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE field_assignments 
                SET status = ?
                WHERE assignment_id = ?
            """, (new_status, assignment_id))
            conn.commit()
    
    def _sync_with_main_database(self) -> int:
        """Synchronize local data with main database"""
        # This would integrate with the main dashboard database
        # For now, just mark local records as synced
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get unsynced observations
            cursor.execute("SELECT COUNT(*) FROM field_observations WHERE synced = FALSE")
            unsynced_count = cursor.fetchone()[0]
            
            # Mark as synced (in real implementation, would push to main DB)
            cursor.execute("UPDATE field_observations SET synced = TRUE WHERE synced = FALSE")
            cursor.execute("UPDATE field_routes SET status = 'synced' WHERE status = 'completed'")
            
            conn.commit()
            
        logger.info(f"🔄 Synced {unsynced_count} field observations")
        return unsynced_count
    
    def _export_route_as_gpx(self, route_id: str) -> str:
        """Export route as GPX format"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM field_routes WHERE route_id = ?", (route_id,))
            route_data = cursor.fetchone()
            
            if not route_data:
                raise ValueError(f"Route {route_id} not found")
            
            waypoints = json.loads(route_data[5] or '[]')  # waypoints column
            
            # Create GPX
            gpx = gpxpy.gpx.GPX()
            
            # Create track
            gpx_track = gpxpy.gpx.GPXTrack()
            gpx_track.name = f"Field Route {route_id}"
            gpx.tracks.append(gpx_track)
            
            # Create segment
            gpx_segment = gpxpy.gpx.GPXTrackSegment()
            gpx_track.segments.append(gpx_segment)
            
            # Add waypoints
            for waypoint in waypoints:
                if len(waypoint) >= 3:
                    point = gpxpy.gpx.GPXTrackPoint(
                        latitude=waypoint[0],
                        longitude=waypoint[1],
                        time=datetime.fromisoformat(waypoint[2])
                    )
                    gpx_segment.points.append(point)
            
            return gpx.to_xml()
    
    def _get_common_species(self) -> List[str]:
        """Get list of common species for offline use"""
        # This would typically come from the main species database
        return [
            "Quercus alba", "Acer rubrum", "Pinus strobus", "Betula nigra",
            "Fraxinus americana", "Carya ovata", "Liquidambar styraciflua",
            "Liriodendron tulipifera", "Fagus grandifolia", "Tsuga canadensis"
        ]
    
    def create_field_assignment(self, team_member: str, area_name: str, 
                              task_type: str, priority: str = "medium",
                              coordinates: Optional[List[Tuple[float, float]]] = None,
                              instructions: Optional[str] = None,
                              due_date: Optional[datetime] = None,
                              estimated_hours: Optional[float] = None) -> str:
        """Create new field assignment"""
        assignment_id = str(uuid.uuid4())
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO field_assignments 
                (assignment_id, team_member, area_name, priority, task_type,
                 coordinates, instructions, due_date, estimated_hours, created_date)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                assignment_id,
                team_member,
                area_name,
                priority,
                task_type,
                json.dumps(coordinates) if coordinates else None,
                instructions,
                due_date.isoformat() if due_date else None,
                estimated_hours,
                datetime.now().isoformat()
            ))
            conn.commit()
        
        logger.info(f"📋 New field assignment created: {assignment_id} for {team_member}")
        return assignment_id
    
    def get_field_summary(self, team_member: Optional[str] = None) -> Dict[str, Any]:
        """Get field operations summary"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Build base queries
            obs_filter = "WHERE team_member = ?" if team_member else ""
            route_filter = "WHERE team_member = ?" if team_member else ""
            assign_filter = "WHERE team_member = ?" if team_member else ""
            params = [team_member] if team_member else []
            
            # Get observation counts
            cursor.execute(f"""
                SELECT COUNT(*) FROM field_observations 
                {obs_filter} AND timestamp > datetime('now', '-7 days')
            """, params)
            recent_observations = cursor.fetchone()[0]
            
            # Get route statistics
            cursor.execute(f"""
                SELECT COUNT(*), SUM(total_distance_km) 
                FROM field_routes 
                {route_filter} AND start_time > datetime('now', '-7 days')
            """, params)
            route_stats = cursor.fetchone()
            recent_routes = route_stats[0] or 0
            total_distance = route_stats[1] or 0.0
            
            # Get assignment counts
            cursor.execute(f"""
                SELECT 
                    SUM(CASE WHEN status = 'assigned' THEN 1 ELSE 0 END) as pending,
                    SUM(CASE WHEN status = 'in_progress' THEN 1 ELSE 0 END) as active,
                    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed
                FROM field_assignments {assign_filter}
            """, params)
            assignment_stats = cursor.fetchone()
            
        return {
            "team_member": team_member or "all",
            "summary_date": datetime.now().isoformat(),
            "recent_observations": recent_observations,
            "recent_routes": recent_routes,
            "total_distance_km": round(total_distance, 2),
            "assignments": {
                "pending": assignment_stats[0] or 0,
                "active": assignment_stats[1] or 0,
                "completed": assignment_stats[2] or 0
            }
        }
    
    def run(self, host: str = "0.0.0.0", port: int = 5001, debug: bool = False):
        """Run the mobile API server"""
        logger.info(f"📱 Starting Field Mobile API server on {host}:{port}")
        self.app.run(host=host, port=port, debug=debug)

def main():
    """Main function for running mobile API"""
    mobile_api = FieldMobileAPI()
    
    # Create sample assignments for testing
    mobile_api.create_field_assignment(
        team_member="ranger_smith",
        area_name="Oak Grove Section A",
        task_type="survey",
        priority="high",
        coordinates=[(40.7128, -74.0060), (40.7130, -74.0058)],
        instructions="Complete health assessment of mature oak trees",
        due_date=datetime.now() + timedelta(days=3),
        estimated_hours=4.0
    )
    
    mobile_api.create_field_assignment(
        team_member="botanist_jones",
        area_name="Wetland Preserve",
        task_type="monitoring",
        priority="medium",
        instructions="Monitor invasive species and water levels",
        due_date=datetime.now() + timedelta(days=7),
        estimated_hours=6.0
    )
    
    # Start API server
    try:
        mobile_api.run(host="0.0.0.0", port=5001, debug=True)
    except KeyboardInterrupt:
        logger.info("📱 Field Mobile API server stopped")

if __name__ == "__main__":
    main()
