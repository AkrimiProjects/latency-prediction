import os
import sys
import time
import yaml
import logging
import subprocess
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from datetime import datetime
import requests
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler('ci_watcher.log'),
        logging.StreamHandler()
    ]
)

class CIEventHandler(FileSystemEventHandler):
    def __init__(self, config):
        self.config = config
        self.last_trigger = 0
        self.cooldown = 5  # Reduce cooldown to 5 seconds for testing
        
    def should_ignore(self, path):
        for pattern in self.config['ignore_paths']:
            if pattern in path:
                logging.info(f"Ignoring path: {path}")
                return True
        return False
        
    def should_watch(self, path):
        # Convert Windows backslashes to forward slashes for matching
        normalized_path = path.replace('\\', '/')
        
        # Remove leading './' if present
        if normalized_path.startswith('./'):
            normalized_path = normalized_path[2:]
            
        # Look for simple directory prefix matches
        for pattern in self.config['watch_paths']:
            # Remove wildcards for simple matching
            dir_pattern = pattern.split('*')[0].rstrip('/')
            if dir_pattern in normalized_path:
                logging.info(f"Watching path: {path} (matches {pattern})")
                return True
                
        logging.info(f"Path not watched: {path}")
        return False
        
    def on_modified(self, event):
        if event.is_directory:
            return
            
        logging.info(f"File modified: {event.src_path}")
        
        if not self.should_watch(event.src_path) or self.should_ignore(event.src_path):
            logging.info(f"File not in watched paths or in ignored paths: {event.src_path}")
            return
            
        current_time = time.time()
        if current_time - self.last_trigger < self.cooldown:
            logging.info(f"Cooldown period active, skipping trigger for: {event.src_path}")
            return
            
        self.last_trigger = current_time
        logging.info(f"Triggering CI pipeline for change in: {event.src_path}")
        self.trigger_ci_pipeline()

    def on_created(self, event):
        if event.is_directory:
            return
            
        logging.info(f"File created: {event.src_path}")
        
        if not self.should_watch(event.src_path) or self.should_ignore(event.src_path):
            return
            
        current_time = time.time()
        if current_time - self.last_trigger < self.cooldown:
            return
            
        self.last_trigger = current_time
        logging.info(f"Triggering CI pipeline for new file: {event.src_path}")
        self.trigger_ci_pipeline()
        
    def trigger_ci_pipeline(self):
        logging.info(f"Changes detected, triggering CI pipeline at {datetime.now()}")
        
        # Run security checks first
        if self.config['security']['scan_dependencies']:
            self.run_security_checks()
            
        # Run CI steps
        for step in self.config['ci_steps']:
            logging.info(f"Running {step['name']}")
            for tool in step['tools']:
                self.run_tool(tool)
                
        # Collect metrics
        if self.config['metrics']['collect_coverage']:
            self.collect_metrics()
            
        # Send notifications
        self.send_notifications()
        
    def run_security_checks(self):
        logging.info("Running security checks...")
        try:
            # Run Bandit for security analysis
            subprocess.run(['bandit', '-r', 'src/'], check=True)
            # Run Safety for dependency vulnerabilities
            subprocess.run(['safety', 'check'], check=True)
        except subprocess.CalledProcessError as e:
            logging.error(f"Security check failed: {e}")
            self.send_notifications("Security check failed!")
            
    def run_tool(self, tool):
        try:
            if tool == 'pylint':
                subprocess.run(['pylint', 'src/'], check=True)
            elif tool == 'black':
                subprocess.run(['black', 'src/'], check=True)
            elif tool == 'pytest':
                logging.info("Running unit tests with pytest...")
                subprocess.run(['pytest', 'tests/'], check=True)
            elif tool == 'coverage':
                subprocess.run(['coverage', 'run', '-m', 'pytest', 'tests/'], check=True)
            elif tool == 'isort':
                subprocess.run(['isort', 'src/'], check=True)
            elif tool == 'bandit':
                subprocess.run(['bandit', '-r', 'src/'], check=True)
            elif tool == 'safety':
                subprocess.run(['safety', 'check'], check=True)
            else:
                logging.warning(f"Unknown tool: {tool}")
        except subprocess.CalledProcessError as e:
            logging.error(f"Tool {tool} failed: {e}")
            
    def collect_metrics(self):
        try:
            # Generate coverage report from previously run tests
            subprocess.run(['coverage', 'report'], check=True)
            subprocess.run(['coverage', 'xml'], check=True)
            logging.info("Coverage metrics collected")
        except subprocess.CalledProcessError as e:
            logging.error(f"Metrics collection failed: {e}")
            
    def send_notifications(self, message=None):
        if not message:
            message = "CI pipeline completed"
            
        # Send Slack notification if configured
        if 'slack_webhook' in self.config['notifications']:
            webhook_url = os.getenv('SLACK_WEBHOOK_URL')
            if webhook_url:
                requests.post(webhook_url, json={'text': message})
                
        # Send email if configured
        if 'email' in self.config['notifications']:
            # Add email sending logic here
            pass

def main():
    config_path = 'config/ci_config.yaml'
    
    if not os.path.exists(config_path):
        logging.error(f"Config file not found: {config_path}")
        sys.exit(1)
        
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        event_handler = CIEventHandler(config)
        observer = Observer()
        observer.schedule(event_handler, path='.', recursive=True)
        observer.start()
        
        logging.info("CI watcher started. Monitoring for changes...")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logging.info("CI watcher stopped by user")
            observer.stop()
        observer.join()
    except Exception as e:
        logging.error(f"Error starting CI watcher: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 