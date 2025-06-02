"""
Gunicorn configuration for production deployment.
"""
import multiprocessing
from src.config import WORKERS, HOST, PORT

# Server socket
bind = f"{HOST}:{PORT}"
backlog = 2048

# Worker processes
workers = WORKERS or multiprocessing.cpu_count() * 2 + 1
worker_class = 'gthread'
threads = 4
worker_connections = 1000
timeout = 30
keepalive = 2

# Process naming
proc_name = 'us_tree_dashboard'
pythonpath = '.'

# Logging
accesslog = '-'
errorlog = '-'
loglevel = 'info'

# SSL (uncomment if using HTTPS)
# keyfile = 'ssl/key.pem'
# certfile = 'ssl/cert.pem'

# Server Mechanics
daemon = False
pidfile = None
umask = 0
user = None
group = None
tmp_upload_dir = None

# Deployment-specific settings
preload_app = True
reload = False
spew = False
check_config = False

def post_fork(server, worker):
    """Called after forking worker processes."""
    server.log.info("Worker spawned (pid: %s)", worker.pid)

def pre_fork(server, worker):
    """Called before forking worker processes."""
    pass

def pre_exec(server):
    """Called before reexecuting the process."""
    server.log.info("Forked child, reexecuting.")

def when_ready(server):
    """Called when server is ready to serve."""
    server.log.info("Server is ready. Spawning workers.")
