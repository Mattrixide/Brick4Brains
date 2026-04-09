/* ============================================================
   Auto-Drive Dashboard — JavaScript
   Same IIFE pattern as esp32wifi dashboard.js
   ============================================================ */

(function () {
  'use strict';

  // --- Constants ---
  var POLL_INTERVAL = 200; // 5 Hz
  var CANVAS_SIZE = 400;
  var ARENA_CM = 244; // 8ft = ~244cm, maps to canvas
  var TRAIL_MAX = 200;

  // --- Mobile sidebar toggle ---
  function initSidebar() {
    var hamburger = document.querySelector('.hamburger');
    var sidebar = document.querySelector('.sidebar');
    var overlay = document.querySelector('.sidebar-overlay');

    if (!hamburger) return;

    hamburger.addEventListener('click', function () {
      sidebar.classList.toggle('open');
      overlay.classList.toggle('open');
    });

    if (overlay) {
      overlay.addEventListener('click', function () {
        sidebar.classList.remove('open');
        overlay.classList.remove('open');
      });
    }
  }

  // --- Section navigation ---
  function initNavigation() {
    document.querySelectorAll('[data-section]').forEach(function (link) {
      link.addEventListener('click', function (e) {
        e.preventDefault();
        var name = this.getAttribute('data-section');
        showSection(name);

        // Update active nav
        document.querySelectorAll('.sidebar__nav a').forEach(function (a) {
          a.classList.remove('active');
        });
        this.classList.add('active');

        // Close mobile sidebar
        document.querySelector('.sidebar').classList.remove('open');
        document.querySelector('.sidebar-overlay').classList.remove('open');
      });
    });
  }

  function showSection(name) {
    var sections = ['status', 'live', 'missions', 'settings', 'battle'];
    sections.forEach(function (s) {
      var el = document.getElementById('section-' + s);
      if (el) el.style.display = (s === name) ? '' : 'none';
    });

    // Start/stop MJPEG stream based on section visibility
    var feedImg = document.getElementById('live-feed');
    if (feedImg) {
      if (name === 'live') {
        feedImg.src = '/api/video_feed';
      } else {
        feedImg.src = '';  // stop streaming when not visible
      }
    }

    var titles = {
      status: 'Status',
      live: 'Live View',
      missions: 'Missions',
      settings: 'Settings',
      battle: 'Battle'
    };
    var titleEl = document.querySelector('.topbar__title');
    if (titleEl) titleEl.textContent = titles[name] || name;
  }

  // --- Helpers ---
  function setText(id, text) {
    var el = document.getElementById(id);
    if (el) el.textContent = text;
  }

  function showResult(id, text) {
    var el = document.getElementById(id);
    if (el) {
      el.textContent = text;
      el.style.display = 'block';
    }
  }

  function radToDeg(rad) {
    return rad * 180.0 / Math.PI;
  }

  // --- Mode color helpers ---
  function getModeClass(mode) {
    switch (mode) {
      case 'auto': return 'badge--auto';
      case 'manual': return 'badge--manual';
      default: return 'badge--idle';
    }
  }

  function getModeDotClass(mode) {
    switch (mode) {
      case 'auto': return 'status-indicator status-indicator--online';
      case 'manual': return 'status-indicator status-indicator--warning';
      default: return 'status-indicator status-indicator--offline';
    }
  }

  function getModeMetricClass(mode) {
    switch (mode) {
      case 'auto': return 'metric-card metric-card--success';
      case 'manual': return 'metric-card metric-card--warning';
      default: return 'metric-card';
    }
  }

  // --- Poll status ---
  var lastState = null;

  function pollStatus() {
    fetch('/api/status')
      .then(function (r) {
        if (!r.ok) throw new Error('HTTP ' + r.status);
        return r.json();
      })
      .then(function (state) {
        lastState = state;
        updateUI(state);
      })
      .catch(function (e) {
        console.error('Poll error:', e);
      });
  }

  function updateUI(s) {
    var mode = (s.mode || 'idle').toUpperCase();
    var modeLower = (s.mode || 'idle').toLowerCase();

    // Metric cards
    setText('m-mode', mode);
    var modeCard = document.getElementById('mc-mode');
    if (modeCard) modeCard.className = getModeMetricClass(modeLower);

    setText('m-fps', s.fps.toFixed(1));
    setText('m-position', s.x_cm.toFixed(1) + ' , ' + s.y_cm.toFixed(1));
    setText('m-heading', s.heading_deg.toFixed(1) + '\u00B0');

    // Topbar badge + live view badge
    var badge = document.getElementById('topbar-mode-badge');
    if (badge) {
      badge.textContent = mode;
      badge.className = 'badge ' + getModeClass(modeLower);
    }
    var liveBadge = document.getElementById('live-mode-badge');
    if (liveBadge) {
      liveBadge.textContent = mode;
      liveBadge.className = 'badge ' + getModeClass(modeLower);
    }
    var liveFps = document.getElementById('live-fps-badge');
    if (liveFps) {
      liveFps.textContent = s.fps.toFixed(0) + ' FPS';
    }

    // Sidebar footer
    var modeDot = document.getElementById('sidebar-mode-dot');
    if (modeDot) modeDot.className = getModeDotClass(modeLower);
    setText('sidebar-mode-text', mode);
    setText('sidebar-esp32', s.esp32_host || '--');
    setText('sidebar-detected', s.detected ? 'Yes' : 'No');

    // Detection badge
    var detBadge = document.getElementById('detected-badge');
    if (detBadge) {
      if (s.detected) {
        detBadge.textContent = 'Tracking';
        detBadge.className = 'badge badge--online';
      } else {
        detBadge.textContent = 'No Detection';
        detBadge.className = 'badge badge--offline';
      }
    }

    // Connection info table
    setText('info-esp32', s.esp32_host || '--');
    setText('info-detected', s.detected ? 'Yes' : 'No');
    setText('info-mission', s.mission_name || 'None');

    // Mission progress
    var pct = Math.round(s.mission_progress * 100);
    var fillEl = document.getElementById('mission-progress-fill');
    if (fillEl) fillEl.style.width = pct + '%';
    setText('mission-progress-text', pct + '%');

    // Calibration point count (during drive calibration)
    var countEl = document.getElementById('calib-point-count');
    if (countEl && s.calib_points !== undefined) {
      if (s.mode === 'calibrating') {
        countEl.textContent = s.calib_points + ' points captured';
      } else {
        countEl.textContent = '';
      }
    }

    // Battle section updates
    updateBattleUI(s);

    // Draw top-down view
    drawTopDown(s);
  }

  // --- Top-Down Canvas ---
  function drawTopDown(state) {
    var canvas = document.getElementById('topdown-canvas');
    if (!canvas) return;
    var ctx = canvas.getContext('2d');
    var w = canvas.width, h = canvas.height;
    var cx = w / 2, cy = h / 2;

    // Scale: map arena cm to canvas pixels
    var scale = (w - 40) / ARENA_CM; // leave 20px padding each side

    ctx.clearRect(0, 0, w, h);

    // Background
    ctx.fillStyle = '#1a1a2e';
    ctx.fillRect(0, 0, w, h);

    // Grid lines (every 30cm)
    ctx.strokeStyle = '#2a2a4e';
    ctx.lineWidth = 1;
    var gridCm = 30;
    var gridPx = gridCm * scale;
    for (var gx = cx % gridPx; gx < w; gx += gridPx) {
      ctx.beginPath();
      ctx.moveTo(gx, 0);
      ctx.lineTo(gx, h);
      ctx.stroke();
    }
    for (var gy = cy % gridPx; gy < h; gy += gridPx) {
      ctx.beginPath();
      ctx.moveTo(0, gy);
      ctx.lineTo(w, gy);
      ctx.stroke();
    }

    // Origin crosshair
    ctx.strokeStyle = '#444466';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(cx, 0);
    ctx.lineTo(cx, h);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(0, cy);
    ctx.lineTo(w, cy);
    ctx.stroke();
    ctx.setLineDash([]);

    // Origin label
    ctx.fillStyle = '#555577';
    ctx.font = '10px monospace';
    ctx.textAlign = 'left';
    ctx.fillText('0,0', cx + 4, cy - 4);

    // Trail (fading cyan line)
    var trail = state.trail || [];
    if (trail.length > 1) {
      ctx.lineWidth = 1.5;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      for (var i = 1; i < trail.length; i++) {
        var alpha = (i / trail.length) * 0.6 + 0.05;
        ctx.strokeStyle = 'rgba(0, 204, 255, ' + alpha.toFixed(2) + ')';
        ctx.beginPath();
        ctx.moveTo(cx + trail[i - 1][0] * scale, cy - trail[i - 1][1] * scale);
        ctx.lineTo(cx + trail[i][0] * scale, cy - trail[i][1] * scale);
        ctx.stroke();
      }
    }

    // Waypoints
    var waypoints = state.waypoints || [];
    waypoints.forEach(function (wp) {
      var wpx = cx + wp.x * scale;
      var wpy = cy - wp.y * scale;
      ctx.beginPath();
      ctx.arc(wpx, wpy, 5, 0, Math.PI * 2);

      if (wp.status === 'reached') {
        ctx.fillStyle = '#34a853';
      } else if (wp.status === 'current') {
        ctx.fillStyle = '#fbbc04';
      } else {
        ctx.fillStyle = '#5f6368';
      }
      ctx.fill();
      ctx.strokeStyle = 'rgba(255,255,255,0.3)';
      ctx.lineWidth = 1;
      ctx.stroke();
    });

    // Robot
    if (state.detected) {
      var rx = cx + state.x_cm * scale;
      var ry = cy - state.y_cm * scale;
      var heading = -state.heading_rad; // negate for canvas coords (y-down)

      var rw = 16, rh = 22; // robot rectangle size in pixels

      ctx.save();
      ctx.translate(rx, ry);
      ctx.rotate(heading);

      // Robot body
      ctx.fillStyle = 'rgba(0, 204, 255, 0.25)';
      ctx.strokeStyle = '#00ccff';
      ctx.lineWidth = 2;
      ctx.fillRect(-rw / 2, -rh / 2, rw, rh);
      ctx.strokeRect(-rw / 2, -rh / 2, rw, rh);

      // Front arrow (heading direction)
      ctx.fillStyle = '#00ccff';
      ctx.beginPath();
      ctx.moveTo(0, -rh / 2 - 8);
      ctx.lineTo(-6, -rh / 2);
      ctx.lineTo(6, -rh / 2);
      ctx.closePath();
      ctx.fill();

      ctx.restore();
    } else {
      // No detection — show question mark at center
      ctx.fillStyle = '#5f6368';
      ctx.font = 'bold 24px sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('?', cx, cy);
    }

    // Arena border hint
    ctx.strokeStyle = '#333355';
    ctx.lineWidth = 1;
    var arenaHalf = (ARENA_CM / 2) * scale;
    ctx.strokeRect(cx - arenaHalf, cy - arenaHalf, arenaHalf * 2, arenaHalf * 2);

    // "Click to go" hint
    ctx.fillStyle = '#555577';
    ctx.font = '10px monospace';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'bottom';
    ctx.fillText('Click to navigate', w - 6, h - 6);
  }

  // --- Click-to-navigate on top-down canvas ---
  function initClickToGo() {
    var canvas = document.getElementById('topdown-canvas');
    if (!canvas) return;

    canvas.style.cursor = 'crosshair';
    canvas.addEventListener('click', function (e) {
      var rect = canvas.getBoundingClientRect();
      var scaleX = canvas.width / rect.width;
      var scaleY = canvas.height / rect.height;
      var px = (e.clientX - rect.left) * scaleX;
      var py = (e.clientY - rect.top) * scaleY;

      var cx = canvas.width / 2;
      var cy = canvas.height / 2;
      var scale = (canvas.width - 40) / ARENA_CM;

      // Convert canvas pixels to cm (y is inverted on canvas)
      var targetX = (px - cx) / scale;
      var targetY = (cy - py) / scale;

      // Send as a "goto" mission
      fetch('/api/mission', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: 'goto',
          params: { x_cm: Math.round(targetX * 10) / 10, y_cm: Math.round(targetY * 10) / 10 }
        })
      })
        .then(function (r) { return r.json(); })
        .then(function (d) {
          if (d.error) console.error('Goto failed:', d.error);
        })
        .catch(function (e) { console.error('Goto failed:', e); });
    });
  }

  // --- Grid toggle ---
  function initGridToggle() {
    var btn = document.getElementById('grid-toggle');
    if (!btn) return;
    btn.addEventListener('click', function () {
      fetch('/api/grid', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: '{}'
      })
        .then(function (r) { return r.json(); })
        .then(function (d) {
          btn.textContent = 'Grid: ' + (d.show_grid ? 'ON' : 'OFF');
        });
    });
  }

  // --- Click on live camera feed: navigate or measure ---
  var measurePoint1 = null; // {x_frac, y_frac} or null

  function initLiveClick() {
    var feedImg = document.getElementById('live-feed');
    if (!feedImg) return;

    feedImg.addEventListener('click', function (e) {
      var rect = feedImg.getBoundingClientRect();
      var x_frac = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
      var y_frac = Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height));

      var tool = document.querySelector('input[name="live-tool"]:checked');
      var mode = tool ? tool.value : 'navigate';

      if (mode === 'navigate') {
        fetch('/api/click_goto', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ x_frac: x_frac, y_frac: y_frac })
        })
          .then(function (r) { return r.json(); })
          .then(function (d) { if (d.ok) console.log('Navigate sent'); })
          .catch(function (e) { console.error('Navigate failed:', e); });

      } else if (mode === 'measure') {
        var resultEl = document.getElementById('measure-result');
        if (!measurePoint1) {
          // First click — store it
          measurePoint1 = { x_frac: x_frac, y_frac: y_frac };
          if (resultEl) resultEl.textContent = 'Click second point...';
        } else {
          // Second click — send measurement
          fetch('/api/measure', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              x1_frac: measurePoint1.x_frac,
              y1_frac: measurePoint1.y_frac,
              x2_frac: x_frac,
              y2_frac: y_frac
            })
          })
            .then(function (r) { return r.json(); })
            .then(function () {
              // Poll status to get measurement result
              setTimeout(function () {
                fetch('/api/status').then(function (r) { return r.json(); })
                  .then(function (s) {
                    var m = s.measure_result;
                    if (m && resultEl) {
                      resultEl.textContent = m.dist_cm + ' cm  (' +
                        m.p1_cm[0] + ',' + m.p1_cm[1] + ') to (' +
                        m.p2_cm[0] + ',' + m.p2_cm[1] + ')';
                    }
                  });
              }, 200);
            })
            .catch(function (e) { console.error('Measure failed:', e); });
          measurePoint1 = null;
        }
      }
    });
  }

  // --- Mission execution ---
  function executeMission(name, params) {
    var resultId = 'res-' + name.replace(/_/g, '-');

    fetch('/api/mission', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: name, params: params })
    })
      .then(function (r) { return r.json(); })
      .then(function (d) {
        if (d.error) {
          showResult(resultId, 'Error: ' + d.error);
        } else {
          showResult(resultId, 'Started: ' + d.mission + ' ' + JSON.stringify(d.params));
        }
      })
      .catch(function (e) {
        showResult(resultId, 'Request failed: ' + e.message);
      });
  }

  function initMissions() {
    document.querySelectorAll('[data-mission]').forEach(function (btn) {
      btn.addEventListener('click', function () {
        var missionName = this.getAttribute('data-mission');
        var paramInput = this.getAttribute('data-param-input');
        var paramKey = this.getAttribute('data-param-key');
        var params = {};

        if (paramInput && paramKey) {
          var inputEl = document.getElementById(paramInput);
          if (inputEl) {
            params[paramKey] = parseFloat(inputEl.value) || 0;
          }
        }

        executeMission(missionName, params);
      });
    });
  }

  // --- Emergency stop ---
  function emergencyStop() {
    fetch('/api/stop', { method: 'POST' })
      .then(function (r) { return r.json(); })
      .then(function (d) {
        console.log('Emergency stop:', d);
      })
      .catch(function (e) {
        console.error('Emergency stop failed:', e);
      });
  }

  // --- Reset to idle ---
  function resetToIdle() {
    fetch('/api/mode', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mode: 'idle' })
    })
      .then(function (r) { return r.json(); })
      .then(function (d) {
        console.log('Reset to idle:', d);
      })
      .catch(function (e) {
        console.error('Reset failed:', e);
      });
  }

  // --- Quick actions ---
  function initQuickActions() {
    var estopBtn = document.getElementById('nav-estop');
    if (estopBtn) {
      estopBtn.addEventListener('click', function (e) {
        e.preventDefault();
        emergencyStop();
      });
    }

    var resetBtn = document.getElementById('nav-reset');
    if (resetBtn) {
      resetBtn.addEventListener('click', function (e) {
        e.preventDefault();
        resetToIdle();
      });
    }
  }

  // --- Autonomy mix sliders ---
  function initMixSliders() {
    var tSlider = document.getElementById('auto-throttle-mix');
    var tLabel = document.getElementById('auto-throttle-mix-val');
    var sSlider = document.getElementById('auto-steering-mix');
    var sLabel = document.getElementById('auto-steering-mix-val');
    if (!tSlider || !sSlider) return;

    function updateLabels() {
      tLabel.textContent = tSlider.value + '%';
      sLabel.textContent = sSlider.value + '%';
    }

    function sendMix() {
      var t = parseInt(tSlider.value) / 100;
      var s = parseInt(sSlider.value) / 100;
      fetch('/api/mix', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ throttle_mix: t, steering_mix: s })
      }).catch(function () {});
    }

    tSlider.addEventListener('input', function () { updateLabels(); sendMix(); });
    sSlider.addEventListener('input', function () { updateLabels(); sendMix(); });

    // Load current values from backend (not localStorage)
    fetch('/api/mix').then(function (r) { return r.json(); }).then(function (d) {
      tSlider.value = Math.round(d.throttle_mix * 100);
      sSlider.value = Math.round(d.steering_mix * 100);
      updateLabels();
    }).catch(function () {
      updateLabels();
      sendMix(); // push HTML defaults to backend
    });
  }

  // Get current mix values (called by mission execution)
  function getMix() {
    var tSlider = document.getElementById('auto-throttle-mix');
    var sSlider = document.getElementById('auto-steering-mix');
    return {
      throttle: tSlider ? parseInt(tSlider.value) / 100 : 0.4,
      steering: sSlider ? parseInt(sSlider.value) / 100 : 0.6
    };
  }

  // --- Calibration ---
  var calibPointCount = 0;

  function initCalibration() {
    // ChArUco calibration
    var charucoBtn = document.getElementById('calib-charuco');
    if (charucoBtn) {
      charucoBtn.addEventListener('click', function () {
        charucoBtn.disabled = true;
        charucoBtn.textContent = 'Detecting board...';
        calibAction('charuco', {}, function () {
          showResult('res-calib', 'ChArUco calibration complete! Floor grid updated.');
          var badge = document.getElementById('calib-status');
          if (badge) { badge.textContent = 'Calibrated (ChArUco)'; badge.className = 'badge badge--online'; }
          charucoBtn.disabled = false;
          charucoBtn.textContent = 'Calibrate from Board';
        });
        // Re-enable on failure too
        setTimeout(function () {
          charucoBtn.disabled = false;
          charucoBtn.textContent = 'Calibrate from Board';
        }, 5000);
      });
    }

    var startBtn = document.getElementById('calib-drive-start');
    var finishBtn = document.getElementById('calib-drive-finish');
    var autoBtn = document.getElementById('calib-auto');
    var saveBtn = document.getElementById('calib-save');
    var loadBtn = document.getElementById('calib-load');
    var clearBtn = document.getElementById('calib-clear');
    var countEl = document.getElementById('calib-point-count');
    if (!startBtn) return;

    var calibrating = false;

    startBtn.addEventListener('click', function () {
      calibAction('drive_start', {}, function () {
        calibrating = true;
        startBtn.style.display = 'none';
        finishBtn.style.display = '';
        showResult('res-calib', 'Calibrating... Drive the robot around the floor with the Xbox controller.');
        var badge = document.getElementById('calib-status');
        if (badge) { badge.textContent = 'Calibrating...'; badge.className = 'badge badge--warning'; }
      });
    });

    finishBtn.addEventListener('click', function () {
      calibAction('drive_finish', {}, function () {
        calibrating = false;
        startBtn.style.display = '';
        finishBtn.style.display = 'none';
        showResult('res-calib', 'Calibration complete! Floor grid updated.');
        var badge = document.getElementById('calib-status');
        if (badge) { badge.textContent = 'Calibrated'; badge.className = 'badge badge--online'; }
      });
    });

    if (autoBtn) {
      autoBtn.addEventListener('click', function () {
        calibAction('auto', {}, function () {
          showResult('res-calib', 'Quick auto-calibration complete.');
          var badge = document.getElementById('calib-status');
          if (badge) { badge.textContent = 'Calibrated'; badge.className = 'badge badge--online'; }
        });
      });
    }

    if (saveBtn) {
      saveBtn.addEventListener('click', function () {
        calibAction('save', {}, function () {
          showResult('res-calib', 'Calibration saved to homography.json');
        });
      });
    }

    if (loadBtn) {
      loadBtn.addEventListener('click', function () {
        calibAction('load', {}, function () {
          showResult('res-calib', 'Calibration loaded.');
          var badge = document.getElementById('calib-status');
          if (badge) { badge.textContent = 'Calibrated'; badge.className = 'badge badge--online'; }
        });
      });
    }

    if (clearBtn) {
      clearBtn.addEventListener('click', function () {
        calibAction('clear', {}, function () {
          showResult('res-calib', 'Calibration cleared');
          var badge = document.getElementById('calib-status');
          if (badge) { badge.textContent = 'Not Calibrated'; badge.className = 'badge badge--info'; }
        });
      });
    }

    // Update point count during calibration via status polling
    // (handled in updateStatus — look for calib_points in status response)
  }

  function calibAction(action, extra, onSuccess) {
    var body = { action: action };
    if (extra) Object.keys(extra).forEach(function (k) { body[k] = extra[k]; });
    fetch('/api/calibrate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    })
      .then(function (r) { return r.json(); })
      .then(function (d) { if (d.ok && onSuccess) onSuccess(); })
      .catch(function (e) { showResult('res-calib', 'Error: ' + e.message); });
  }

  // --- Settings ---
  function initSettings() {
    var input = document.getElementById('settings-esp32');
    var btn = document.getElementById('settings-esp32-save');
    if (!input || !btn) return;

    // Load saved value
    var saved = localStorage.getItem('autodrive_esp32_host');
    if (saved) input.value = saved;

    btn.addEventListener('click', function () {
      var val = input.value.trim();
      if (val) {
        localStorage.setItem('autodrive_esp32_host', val);
        showResult('res-settings', 'ESP32 address saved: ' + val);
      }
    });
  }

  // --- Battle Section ---
  var battleConfig = null;

  function updateBattleUI(s) {
    // Match timer
    var remaining = s.match_remaining_s;
    if (remaining != null) {
      var mins = Math.floor(remaining / 60);
      var secs = Math.floor(remaining % 60);
      setText('battle-timer', mins + ':' + (secs < 10 ? '0' : '') + secs);
    } else {
      setText('battle-timer', '--:--');
    }

    // Battle state badge
    var bstate = s.battle_state || '--';
    setText('battle-state-display', bstate.toUpperCase().replace('_', ' '));
    var stateBadge = document.getElementById('battle-state-badge');
    if (stateBadge) {
      stateBadge.textContent = bstate.toUpperCase().replace('_', ' ');
      var stateClass = 'badge badge--info';
      if (bstate.indexOf('charge') >= 0) stateClass = 'badge badge--danger';
      else if (bstate === 'pin') stateClass = 'badge badge--accent';
      else if (bstate.indexOf('evade') >= 0 || bstate.indexOf('retreat') >= 0) stateClass = 'badge badge--warning';
      else if (bstate === 'wait' || bstate === '--') stateClass = 'badge badge--idle';
      else if (bstate.indexOf('acquire') >= 0) stateClass = 'badge badge--info';
      else if (bstate.indexOf('pit') >= 0) stateClass = 'badge badge--online';
      else if (bstate.indexOf('unstick') >= 0 || bstate === 'wall_reverse') stateClass = 'badge badge--warning';
      else if (bstate === 'victory_dance') stateClass = 'badge badge--accent';
      else if (bstate === 'lost_aruco' || bstate === 'lost_target') stateClass = 'badge badge--idle';
      stateBadge.className = stateClass;
    }

    // Match phase display
    var phase = s.match_phase || '--';
    setText('battle-phase-display', phase.toUpperCase());

    // Pin timer
    var pinRemaining = s.pin_remaining_s;
    setText('battle-pin-timer', pinRemaining != null ? pinRemaining.toFixed(1) + 's' : '--');

    // Urgency
    var urgency = s.urgency || 0;
    setText('battle-urgency', Math.round(urgency * 100) + '%');
    var urgencyBar = document.getElementById('battle-urgency-bar');
    if (urgencyBar) {
      urgencyBar.style.width = Math.round(urgency * 100) + '%';
      if (urgency > 0.7) urgencyBar.style.background = 'var(--color-danger)';
      else if (urgency > 0.3) urgencyBar.style.background = 'var(--color-warning)';
      else urgencyBar.style.background = 'var(--color-accent)';
    }
  }

  function loadBattleConfig() {
    fetch('/api/battle/config')
      .then(function (r) { return r.json(); })
      .then(function (cfg) {
        battleConfig = cfg;
        var durInput = document.getElementById('battle-match-duration');
        if (durInput) durInput.value = cfg.match_duration_s || 180;
        var pinInput = document.getElementById('battle-pin-duration');
        if (pinInput) pinInput.value = cfg.pin_duration_s || 5;
        var stratSelect = document.getElementById('battle-strategy');
        if (stratSelect) stratSelect.value = cfg.strategy || 'charge';
        var openingSelect = document.getElementById('battle-opening');
        if (openingSelect) openingSelect.value = cfg.opening_strategy || 'charge';
        var pushInput = document.getElementById('battle-push-commit');
        if (pushInput) pushInput.value = cfg.push_commit_s || 1.0;
        // Safe side radios
        var side = cfg.safe_side || 'front';
        var radio = document.querySelector('input[name="battle-safe-side"][value="' + side + '"]');
        if (radio) radio.checked = true;
        // Pit location
        if (cfg.pit_x_cm !== undefined && cfg.pit_y_cm !== undefined) {
          setText('battle-pit-location', '(' + cfg.pit_x_cm.toFixed(0) + ', ' + cfg.pit_y_cm.toFixed(0) + ') cm');
        }
      })
      .catch(function (e) { console.error('Failed to load battle config:', e); });
  }

  function saveBattleConfig() {
    var cfg = {};
    var durInput = document.getElementById('battle-match-duration');
    if (durInput) cfg.match_duration_s = parseFloat(durInput.value);
    var pinInput = document.getElementById('battle-pin-duration');
    if (pinInput) cfg.pin_duration_s = parseFloat(pinInput.value);
    var stratSelect = document.getElementById('battle-strategy');
    if (stratSelect) cfg.strategy = stratSelect.value;
    var openingSelect = document.getElementById('battle-opening');
    if (openingSelect) cfg.opening_strategy = openingSelect.value;
    var pushInput = document.getElementById('battle-push-commit');
    if (pushInput) cfg.push_commit_s = parseFloat(pushInput.value);
    var sideRadio = document.querySelector('input[name="battle-safe-side"]:checked');
    if (sideRadio) cfg.safe_side = sideRadio.value;

    fetch('/api/battle/config', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(cfg)
    })
    .then(function (r) { return r.json(); })
    .then(function (d) {
      var result = document.getElementById('battle-config-result');
      if (result) {
        result.textContent = d.error ? ('Error: ' + d.error) : 'Config saved!';
        setTimeout(function () { result.textContent = ''; }, 3000);
      }
    })
    .catch(function (e) {
      var result = document.getElementById('battle-config-result');
      if (result) result.textContent = 'Save failed: ' + e;
    });
  }

  function initBattle() {
    var startBtn = document.getElementById('battle-start');
    var stopBtn = document.getElementById('battle-stop');
    var saveBtn = document.getElementById('battle-save-config');
    var setPitBtn = document.getElementById('battle-set-pit');

    if (startBtn) {
      startBtn.addEventListener('click', function () {
        fetch('/api/battle/start', { method: 'POST' })
          .then(function (r) { return r.json(); })
          .then(function (d) { console.log('Battle started:', d); });
      });
    }

    if (stopBtn) {
      stopBtn.addEventListener('click', function () {
        fetch('/api/battle/stop', { method: 'POST' })
          .then(function (r) { return r.json(); })
          .then(function (d) { console.log('Battle stopped:', d); });
      });
    }

    if (saveBtn) {
      saveBtn.addEventListener('click', saveBattleConfig);
    }

    if (setPitBtn) {
      setPitBtn.addEventListener('click', function () {
        alert('Click on the Status top-down canvas to set pit location (coming soon)');
      });
    }

    // Load initial config
    loadBattleConfig();
  }

  // --- Init ---
  document.addEventListener('DOMContentLoaded', function () {
    initSidebar();
    initNavigation();
    initMissions();
    initMixSliders();
    initClickToGo();
    initLiveClick();
    initGridToggle();
    initQuickActions();
    initCalibration();
    initSettings();
    initBattle();

    // Draw initial empty canvas
    drawTopDown({
      x_cm: 0, y_cm: 0, heading_rad: 0, heading_deg: 0,
      detected: false, trail: [], waypoints: [],
      mission_progress: 0, mode: 'idle', fps: 0,
      esp32_host: '', mission_name: ''
    });

    // Start polling
    pollStatus();
    setInterval(pollStatus, POLL_INTERVAL);
  });

})();
