"""
Auto-Drive Dashboard Server
Flask server sharing state with the main CV tracking loop via a thread-safe dict.
"""

from flask import Flask, jsonify, request, send_from_directory, Response
import threading
import os
import time
import logging

# Suppress Flask request logs in production
log = logging.getLogger('werkzeug')
log.setLevel(logging.WARNING)


class DashboardServer:
    """Serves the Auto-Drive dashboard UI and REST API on a background thread."""

    def __init__(self, shared_state, port=5000):
        """
        Args:
            shared_state: dict with a 'lock' key holding a threading.Lock().
                Main loop writes:
                    mode (str): 'idle', 'auto', or 'manual'
                    x_cm (float): robot x position in cm
                    y_cm (float): robot y position in cm
                    heading_rad (float): robot heading in radians
                    detected (bool): whether ArUco marker is currently detected
                    fps (float): camera frame rate
                    esp32_host (str): ESP32 address
                    mission_name (str): current mission name or ''
                    mission_progress (float): 0.0 to 1.0
                    waypoints (list[dict]): [{x, y, status}] status: 'pending'|'reached'|'current'
                    trail (list[tuple]): [(x, y), ...] recent position history
                    available_missions (list[str]): mission names
                Dashboard writes:
                    pending_command (dict|None): command from dashboard for main loop
            port: HTTP port to serve on.
        """
        self._state = shared_state
        self._port = port
        self._thread = None
        self._frame_callback = None  # set by main loop to get JPEG frames

        # Resolve dashboard static files directory
        self._dashboard_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dashboard')

        self._app = Flask(__name__, static_folder=None)
        self._register_routes()

    def _register_routes(self):
        app = self._app

        # --- Static file serving ---

        @app.route('/')
        def serve_index():
            return send_from_directory(self._dashboard_dir, 'index.html')

        @app.route('/css/<path:filename>')
        def serve_css(filename):
            return send_from_directory(os.path.join(self._dashboard_dir, 'css'), filename)

        @app.route('/js/<path:filename>')
        def serve_js(filename):
            return send_from_directory(os.path.join(self._dashboard_dir, 'js'), filename)

        # --- MJPEG video stream ---

        @app.route('/api/video_feed')
        def video_feed():
            """MJPEG stream of the camera with tracking overlay."""
            def generate():
                while True:
                    if self._frame_callback:
                        jpeg = self._frame_callback()
                        if jpeg is not None:
                            yield (b'--frame\r\n'
                                   b'Content-Type: image/jpeg\r\n\r\n'
                                   + jpeg + b'\r\n')
                    time.sleep(1.0 / 30)  # cap stream at 30fps
            return Response(
                generate(),
                mimetype='multipart/x-mixed-replace; boundary=frame',
            )

        # --- API endpoints ---

        @app.route('/api/status')
        def api_status():
            """Return current system state."""
            lock = self._state.get('lock')
            if lock:
                lock.acquire()
            try:
                data = {
                    'mode': self._state.get('mode', 'idle'),
                    'x_cm': round(self._state.get('x_cm', 0.0), 1),
                    'y_cm': round(self._state.get('y_cm', 0.0), 1),
                    'heading_rad': round(self._state.get('heading_rad', 0.0), 4),
                    'heading_deg': round(
                        self._state.get('heading_rad', 0.0) * 180.0 / 3.141592653589793, 1
                    ),
                    'detected': self._state.get('detected', False),
                    'fps': round(self._state.get('fps', 0.0), 1),
                    'esp32_host': self._state.get('esp32_host', ''),
                    'mission_name': self._state.get('mission_name', ''),
                    'mission_progress': round(self._state.get('mission_progress', 0.0), 3),
                    'waypoints': self._state.get('waypoints', []),
                    'trail': self._state.get('trail', []),
                    'measure_result': self._state.get('measure_result', None),
                    'calib_points': self._state.get('calib_points', 0),
                    'system_mode': self._state.get('system_mode', 'config'),
                    'battle_state': self._state.get('battle_state', None),
                    'match_remaining_s': self._state.get('match_remaining_s', None),
                    'pin_remaining_s': self._state.get('pin_remaining_s', None),
                    'urgency': self._state.get('urgency', None),
                    'timestamp': time.time(),
                }
            finally:
                if lock:
                    lock.release()
            return jsonify(data)

        @app.route('/api/missions')
        def api_missions():
            """Return list of available missions."""
            lock = self._state.get('lock')
            if lock:
                lock.acquire()
            try:
                missions = list(self._state.get('available_missions', []))
            finally:
                if lock:
                    lock.release()
            return jsonify({'missions': missions})

        @app.route('/api/mission', methods=['POST'])
        def api_start_mission():
            """Start a mission. Body: {name: str, params: dict}"""
            body = request.get_json(silent=True)
            if not body or 'name' not in body:
                return jsonify({'error': 'Missing mission name'}), 400

            name = body['name']
            params = body.get('params', {})

            lock = self._state.get('lock')
            if lock:
                lock.acquire()
            try:
                current_mode = self._state.get('mode', 'idle')
                if current_mode == 'auto':
                    return jsonify({'error': 'Mission already in progress'}), 409

                self._state['pending_command'] = {
                    'type': 'mission',
                    'name': name,
                    'params': params,
                    'timestamp': time.time(),
                }
            finally:
                if lock:
                    lock.release()

            return jsonify({'ok': True, 'mission': name, 'params': params})

        @app.route('/api/mode', methods=['POST'])
        def api_set_mode():
            """Set mode. Body: {mode: str}"""
            body = request.get_json(silent=True)
            if not body or 'mode' not in body:
                return jsonify({'error': 'Missing mode'}), 400

            mode = body['mode']
            if mode not in ('idle', 'manual'):
                return jsonify({'error': 'Invalid mode (use idle or manual)'}), 400

            lock = self._state.get('lock')
            if lock:
                lock.acquire()
            try:
                self._state['pending_command'] = {
                    'type': 'set_mode',
                    'mode': mode,
                    'timestamp': time.time(),
                }
            finally:
                if lock:
                    lock.release()

            return jsonify({'ok': True, 'mode': mode})

        @app.route('/api/stop', methods=['POST'])
        def api_emergency_stop():
            """Emergency stop — immediately sets pending command to halt."""
            lock = self._state.get('lock')
            if lock:
                lock.acquire()
            try:
                self._state['pending_command'] = {
                    'type': 'emergency_stop',
                    'timestamp': time.time(),
                }
            finally:
                if lock:
                    lock.release()

            return jsonify({'ok': True, 'action': 'emergency_stop'})

        @app.route('/api/mix', methods=['GET', 'POST'])
        def api_mix():
            """Get or set autonomy throttle/steering mix."""
            lock = self._state.get('lock')
            if request.method == 'POST':
                body = request.get_json(silent=True) or {}
                if lock:
                    lock.acquire()
                try:
                    if 'throttle_mix' in body:
                        self._state['throttle_mix'] = max(0.0, min(1.0, float(body['throttle_mix'])))
                    if 'steering_mix' in body:
                        self._state['steering_mix'] = max(0.0, min(1.0, float(body['steering_mix'])))
                finally:
                    if lock:
                        lock.release()

            if lock:
                lock.acquire()
            try:
                data = {
                    'throttle_mix': self._state.get('throttle_mix', 0.4),
                    'steering_mix': self._state.get('steering_mix', 0.6),
                }
            finally:
                if lock:
                    lock.release()
            return jsonify(data)

        @app.route('/api/charuco_board')
        def api_charuco_board():
            """Serve the ChArUco board PNG for download/printing."""
            board_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), 'charuco_board.png'
            )
            if os.path.exists(board_path):
                return send_from_directory(
                    os.path.dirname(board_path),
                    'charuco_board.png',
                    mimetype='image/png',
                    as_attachment=True,
                    download_name='charuco_board.png',
                )
            return jsonify({'error': 'Board image not generated yet'}), 404

        @app.route('/api/calibrate', methods=['POST'])
        def api_calibrate():
            """Calibration commands.

            Body options:
              {action: "capture", x_cm: float, y_cm: float}
                — capture current marker position as a calibration point
              {action: "compute"} — compute homography from collected points
              {action: "clear"} — clear all calibration points
              {action: "save"} — save homography to disk
              {action: "load"} — load homography from disk
            """
            body = request.get_json(silent=True)
            if not body or 'action' not in body:
                return jsonify({'error': 'Missing action'}), 400

            lock = self._state.get('lock')
            if lock:
                lock.acquire()
            try:
                self._state['pending_command'] = {
                    'type': 'calibrate',
                    'action': body['action'],
                    'x_cm': body.get('x_cm', 0.0),
                    'y_cm': body.get('y_cm', 0.0),
                    'timestamp': time.time(),
                }
            finally:
                if lock:
                    lock.release()

            return jsonify({'ok': True, 'action': body['action']})

        @app.route('/api/click_goto', methods=['POST'])
        def api_click_goto():
            """Navigate to a clicked point on the live camera feed.

            Body: {x_frac: 0-1, y_frac: 0-1} — fractional position in the frame.
            Pixel→world conversion happens in the main loop (uses homography if available).
            """
            body = request.get_json(silent=True)
            if not body or 'x_frac' not in body or 'y_frac' not in body:
                return jsonify({'error': 'Missing x_frac/y_frac'}), 400

            lock = self._state.get('lock')
            if lock:
                lock.acquire()
            try:
                self._state['pending_command'] = {
                    'type': 'click_goto',
                    'x_frac': float(body['x_frac']),
                    'y_frac': float(body['y_frac']),
                    'timestamp': time.time(),
                }
            finally:
                if lock:
                    lock.release()

            return jsonify({'ok': True})

        @app.route('/api/grid', methods=['POST'])
        def api_grid():
            """Toggle floor grid overlay."""
            body = request.get_json(silent=True) or {}
            lock = self._state.get('lock')
            if lock:
                lock.acquire()
            try:
                if 'show' in body:
                    self._state['show_grid'] = bool(body['show'])
                else:
                    self._state['show_grid'] = not self._state.get('show_grid', True)
                result = self._state['show_grid']
            finally:
                if lock:
                    lock.release()
            return jsonify({'ok': True, 'show_grid': result})

        @app.route('/api/measure', methods=['POST'])
        def api_measure():
            """Measure distance between two clicked points on the camera feed.

            Body: {x1_frac, y1_frac, x2_frac, y2_frac} — fractional positions.
            Returns distance in cm using the tracker's pixel-to-world conversion.
            """
            body = request.get_json(silent=True)
            if not body:
                return jsonify({'error': 'Missing body'}), 400

            lock = self._state.get('lock')
            if lock:
                lock.acquire()
            try:
                frame_w = self._state.get('frame_w', 1280)
                frame_h = self._state.get('frame_h', 720)
            finally:
                if lock:
                    lock.release()

            # Store the measurement request for the main loop to process
            if lock:
                lock.acquire()
            try:
                self._state['pending_command'] = {
                    'type': 'measure',
                    'x1_px': float(body['x1_frac']) * frame_w,
                    'y1_px': float(body['y1_frac']) * frame_h,
                    'x2_px': float(body['x2_frac']) * frame_w,
                    'y2_px': float(body['y2_frac']) * frame_h,
                    'timestamp': time.time(),
                }
            finally:
                if lock:
                    lock.release()

            return jsonify({'ok': True, 'status': 'measuring'})

        # --- Battle API ---

        @app.route('/api/battle/start', methods=['POST'])
        def api_battle_start():
            """Start a battle match."""
            lock = self._state.get('lock')
            if lock:
                lock.acquire()
            try:
                self._state['pending_command'] = {
                    'type': 'start_battle',
                    'timestamp': time.time(),
                }
            finally:
                if lock:
                    lock.release()
            return jsonify({'ok': True, 'action': 'start_battle'})

        @app.route('/api/battle/stop', methods=['POST'])
        def api_battle_stop():
            """Stop a battle match."""
            lock = self._state.get('lock')
            if lock:
                lock.acquire()
            try:
                self._state['pending_command'] = {
                    'type': 'stop_battle',
                    'timestamp': time.time(),
                }
            finally:
                if lock:
                    lock.release()
            return jsonify({'ok': True, 'action': 'stop_battle'})

        @app.route('/api/battle/config', methods=['GET', 'POST'])
        def api_battle_config():
            """Get or update battle configuration."""
            if request.method == 'GET':
                # Read config from file
                import json
                config_path = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    'battle_config.json'
                )
                # Try loading from same dir as dashboard_server.py
                config_path2 = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    'battle_config.json'
                )
                for p in [config_path2, config_path]:
                    try:
                        with open(p) as f:
                            return jsonify(json.load(f))
                    except (FileNotFoundError, json.JSONDecodeError):
                        continue
                # Return defaults
                return jsonify({
                    'match_duration_s': 180,
                    'pin_duration_s': 5,
                    'safe_side': 'front',
                    'strategy': 'charge',
                    'pit_x_cm': 0, 'pit_y_cm': 0,
                    'pit_radius_cm': 20,
                    'pit_danger_radius_cm': 40,
                })
            else:
                body = request.get_json(silent=True)
                if not body:
                    return jsonify({'error': 'Missing body'}), 400
                lock = self._state.get('lock')
                if lock:
                    lock.acquire()
                try:
                    self._state['pending_command'] = {
                        'type': 'battle_config',
                        'config': body,
                        'timestamp': time.time(),
                    }
                finally:
                    if lock:
                        lock.release()
                return jsonify({'ok': True, 'config': body})

        # --- Error handlers ---

        @app.errorhandler(404)
        def not_found(_e):
            return jsonify({'error': 'Not found'}), 404

        @app.errorhandler(500)
        def server_error(_e):
            return jsonify({'error': 'Internal server error'}), 500

    def start(self):
        """Start the Flask server on a daemon background thread."""
        if self._thread and self._thread.is_alive():
            return

        self._thread = threading.Thread(
            target=self._run,
            name='dashboard-server',
            daemon=True,
        )
        self._thread.start()

    def _run(self):
        """Run the Flask app (called in background thread)."""
        self._app.run(
            host='0.0.0.0',
            port=self._port,
            debug=False,
            use_reloader=False,
            threaded=True,
        )

    @property
    def port(self):
        return self._port

    @property
    def url(self):
        return f'http://localhost:{self._port}'


def create_shared_state(**overrides):
    """Create a shared_state dict with sensible defaults and a Lock."""
    state = {
        'lock': threading.Lock(),
        'mode': 'idle',
        'x_cm': 0.0,
        'y_cm': 0.0,
        'heading_rad': 0.0,
        'detected': False,
        'fps': 0.0,
        'esp32_host': '',
        'mission_name': '',
        'mission_progress': 0.0,
        'waypoints': [],
        'trail': [],
        'available_missions': ['drive_square', 'forward_back', 'drive_circle'],
        'pending_command': None,
        'throttle_mix': 0.6,
        'steering_mix': 0.8,
        'calib_points': 0,
        'show_grid': True,
    }
    state.update(overrides)
    return state


# --- Standalone test ---
if __name__ == '__main__':
    import math

    shared = create_shared_state(esp32_host='192.168.1.100')
    server = DashboardServer(shared, port=5000)
    server.start()
    print(f'Dashboard running at {server.url}')

    # Simulate state updates
    t = 0
    try:
        while True:
            with shared['lock']:
                shared['x_cm'] = 30.0 * math.cos(t * 0.5)
                shared['y_cm'] = 30.0 * math.sin(t * 0.5)
                shared['heading_rad'] = t * 0.5 + math.pi / 2
                shared['detected'] = True
                shared['fps'] = 29.5 + (t % 3) * 0.5

                # Check for pending commands
                cmd = shared.get('pending_command')
                if cmd:
                    print(f'[main] Received command: {cmd}')
                    shared['pending_command'] = None
                    if cmd['type'] == 'mission':
                        shared['mode'] = 'auto'
                        shared['mission_name'] = cmd['name']
                        shared['mission_progress'] = 0.0
                    elif cmd['type'] == 'set_mode':
                        shared['mode'] = cmd['mode']
                        shared['mission_name'] = ''
                        shared['mission_progress'] = 0.0
                    elif cmd['type'] == 'emergency_stop':
                        shared['mode'] = 'idle'
                        shared['mission_name'] = ''
                        shared['mission_progress'] = 0.0

                # Simulate mission progress
                if shared['mode'] == 'auto' and shared['mission_progress'] < 1.0:
                    shared['mission_progress'] = min(1.0, shared['mission_progress'] + 0.02)
                    if shared['mission_progress'] >= 1.0:
                        shared['mode'] = 'idle'
                        shared['mission_name'] = ''

                # Update trail
                trail = list(shared.get('trail', []))
                trail.append((shared['x_cm'], shared['y_cm']))
                if len(trail) > 200:
                    trail = trail[-200:]
                shared['trail'] = trail

            t += 0.05
            time.sleep(0.05)
    except KeyboardInterrupt:
        print('\nShutdown.')
