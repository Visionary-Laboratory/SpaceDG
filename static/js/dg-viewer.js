(function () {
  const DEGRADATIONS = [
    { id: 'motion_blur', label: 'Motion Blur', levels: false },
    { id: 'low_light', label: 'Low Light' },
    { id: 'haze', label: 'Haze' },
    { id: 'defocus', label: 'Defocus' },
    { id: 'low_res', label: 'Low Resolution' },
    { id: 'water_droplets', label: 'Water Droplets' },
    { id: 'jpeg_compression', label: 'JPEG Compression' },
    { id: 'over_exposure', label: 'Over Exposure' },
    { id: 'distortion', label: 'Distortion' },
  ];

  const cleanVideo = document.getElementById('dg-clean-video');
  const degradedA = document.getElementById('dg-degraded-a');
  const degradedB = document.getElementById('dg-degraded-b');
  const cleanPane = document.getElementById('dg-pane-clean');
  const degradedPane = document.getElementById('dg-pane-degraded');
  const typeTitle = document.getElementById('dg-type-title');
  const typeNote = document.getElementById('dg-type-note');
  const typeButtons = document.getElementById('dg-type-buttons');
  const levelControl = document.getElementById('dg-level-control');
  const levelSlider = document.getElementById('dg-level-slider');
  const levelValue = document.getElementById('dg-level-value');
  const prevBtn = document.getElementById('dg-prev');
  const nextBtn = document.getElementById('dg-next');
  const viewerRoot = document.getElementById('dg-viewer');

  if (!cleanVideo || !degradedA || !degradedB || !typeButtons || !viewerRoot) {
    return;
  }

  const MAX_WARM_CONCURRENT = 2;
  const warmed = new Set();
  const warming = new Set();
  const warmQueue = [];
  let warmActive = 0;
  let currentIndex = 0;
  let currentLevel = 1;
  let levelInputTimer = null;
  let viewerVisible = false;
  let programmaticDepth = 0;
  let activeSlot = 'a';
  let loadGeneration = 0;
  let loadPairTimer = null;
  let initialLoad = true;
  let pauseOnHideTimer = null;

  function getActiveDegraded() {
    return activeSlot === 'a' ? degradedA : degradedB;
  }

  function getInactiveDegraded() {
    return activeSlot === 'a' ? degradedB : degradedA;
  }

  function activateSlot(slot) {
    activeSlot = slot;
    degradedA.classList.toggle('is-active', slot === 'a');
    degradedB.classList.toggle('is-active', slot === 'b');
    degradedA.preload = slot === 'a' ? 'auto' : 'none';
    degradedB.preload = slot === 'b' ? 'auto' : 'none';
  }

  function cleanSrc(typeId) {
    return './static/videos/dg/clean/' + typeId + '/video.mp4';
  }

  function hasLevels(typeId) {
    const item = DEGRADATIONS.find(function (d) {
      return d.id === typeId;
    });
    return !item || item.levels !== false;
  }

  function degradedSrc(typeId, level) {
    if (!hasLevels(typeId)) {
      return './static/videos/dg/degraded/' + typeId + '/video.mp4';
    }
    return './static/videos/dg/degraded/' + typeId + '/lvl' + level + '.mp4';
  }

  function mod(n, m) {
    return ((n % m) + m) % m;
  }

  function clampTime(time, video) {
    if (!isFinite(time) || time < 0) {
      return 0;
    }
    const duration = video.duration;
    if (!isFinite(duration) || duration <= 0) {
      return time;
    }
    return Math.min(time, Math.max(0, duration - 0.04));
  }

  function getMasterTime() {
    return cleanVideo.currentTime || 0;
  }

  function isProgrammatic() {
    return programmaticDepth > 0;
  }

  function beginProgrammatic() {
    programmaticDepth += 1;
  }

  function endProgrammatic() {
    programmaticDepth = Math.max(0, programmaticDepth - 1);
  }

  function seekDegradedTo(time) {
    const degraded = getActiveDegraded();
    const target = clampTime(time, degraded);
    if (Math.abs(degraded.currentTime - target) < 0.02) {
      return;
    }
    beginProgrammatic();
    degraded.currentTime = target;
    setTimeout(function () {
      endProgrammatic();
      resumeDegradedIfNeeded();
    }, 250);
  }

  function seekAndWait(video, time) {
    const target = clampTime(time, video);
    if (Math.abs(video.currentTime - target) < 0.03 && video.readyState >= 2) {
      return Promise.resolve();
    }
    return new Promise(function (resolve) {
      let settled = false;
      const finish = function () {
        if (settled) {
          return;
        }
        settled = true;
        video.removeEventListener('seeked', finish);
        endProgrammatic();
        resolve();
      };
      video.addEventListener('seeked', finish, { once: true });
      beginProgrammatic();
      video.currentTime = target;
      setTimeout(finish, 350);
    });
  }

  function deferCleanupInactive() {
    const inactive = getInactiveDegraded();
    requestAnimationFrame(function () {
      // Keep the inactive slot loaded to speed up subsequent switches.
    });
  }

  function syncDegradedToClean() {
    seekDegradedTo(getMasterTime());
  }

  function resumeDegradedIfNeeded() {
    const degraded = getActiveDegraded();
    if (!cleanVideo.paused && degraded.paused && !isProgrammatic()) {
      playVideo(degraded);
    }
  }

  function isPlaybackActive() {
    return viewerVisible && !cleanVideo.paused;
  }

  function isPaneLoading() {
    return (
      (cleanPane && cleanPane.classList.contains('is-loading')) ||
      (degradedPane && degradedPane.classList.contains('is-loading'))
    );
  }

  function syncBothTo(time) {
    beginProgrammatic();
    cleanVideo.currentTime = clampTime(time, cleanVideo);
    getActiveDegraded().currentTime = clampTime(time, getActiveDegraded());
    setTimeout(function () {
      programmaticDepth = 0;
      resumeDegradedIfNeeded();
    }, 250);
  }

  function playVideo(video) {
    const playPromise = video.play();
    if (playPromise && typeof playPromise.catch === 'function') {
      playPromise.catch(function () {});
    }
  }

  function playPair() {
    syncBothTo(getMasterTime());
    playVideo(cleanVideo);
    playVideo(getActiveDegraded());
  }

  function pausePair() {
    cleanVideo.pause();
    degradedA.pause();
    degradedB.pause();
  }

  function setPaneLoading(pane, loading) {
    if (!pane) {
      return;
    }
    pane.classList.toggle('is-loading', loading);
  }

  function bumpLoadGeneration() {
    loadGeneration += 1;
    return loadGeneration;
  }

  function isStale(gen) {
    return gen !== loadGeneration;
  }

  function waitForReady(video) {
    if (video.readyState >= 3 && !video.error) {
      return Promise.resolve();
    }
    return new Promise(function (resolve, reject) {
      const done = function () {
        video.removeEventListener('canplay', done);
        video.removeEventListener('error', done);
        if (video.error) {
          reject(video.error);
          return;
        }
        resolve();
      };
      video.addEventListener('canplay', done, { once: true });
      video.addEventListener('error', done, { once: true });
    });
  }

  function waitForFirstFrame(video) {
    if (video.error) {
      return Promise.resolve();
    }
    if (typeof video.requestVideoFrameCallback === 'function') {
      return new Promise(function (resolve) {
        video.requestVideoFrameCallback(function () {
          resolve();
        });
      });
    }
    if (video.readyState >= 2) {
      return Promise.resolve();
    }
    return new Promise(function (resolve) {
      video.addEventListener(
        'loadeddata',
        function () {
          resolve();
        },
        { once: true }
      );
      // Safety: don't hang forever if event doesn't fire.
      setTimeout(resolve, 250);
    });
  }

  function warmUrl(url) {
    if (!url || warmed.has(url) || warming.has(url)) {
      return Promise.resolve();
    }
    warming.add(url);

    return new Promise(function (resolve) {
      const hidden = document.createElement('video');
      hidden.muted = true;
      hidden.playsInline = true;
      hidden.preload = 'auto';
      hidden.setAttribute('aria-hidden', 'true');
      hidden.tabIndex = -1;
      hidden.style.cssText = 'position:absolute;width:0;height:0;opacity:0;pointer-events:none';
      document.body.appendChild(hidden);

      const finish = function () {
        warming.delete(url);
        warmed.add(url);
        hidden.remove();
        resolve();
      };

      hidden.addEventListener('canplaythrough', finish, { once: true });
      hidden.addEventListener('error', finish, { once: true });
      hidden.src = url;
      hidden.load();
    });
  }

  function drainWarmQueue() {
    if (isPaneLoading()) {
      return;
    }
    const maxConcurrent = isPlaybackActive() ? 1 : MAX_WARM_CONCURRENT;
    while (warmActive < maxConcurrent && warmQueue.length > 0) {
      const url = warmQueue.shift();
      if (!url || warmed.has(url) || warming.has(url)) {
        continue;
      }
      warmActive += 1;
      warmUrl(url).finally(function () {
        warmActive -= 1;
        drainWarmQueue();
      });
    }
  }

  function enqueueWarm(url) {
    if (!url || warmed.has(url) || warming.has(url) || warmQueue.indexOf(url) !== -1) {
      return;
    }
    warmQueue.push(url);
    drainWarmQueue();
  }

  function prioritizeWarm(urls) {
    urls.forEach(function (url) {
      if (!url) {
        return;
      }
      const idx = warmQueue.indexOf(url);
      if (idx >= 0) {
        warmQueue.splice(idx, 1);
      }
      warmQueue.unshift(url);
    });
    drainWarmQueue();
  }

  function prefetchNeighbors() {
    if (!viewerVisible || isPlaybackActive() || isPaneLoading()) {
      return;
    }
    const n = DEGRADATIONS.length;
    const cur = DEGRADATIONS[currentIndex];
    const prev = DEGRADATIONS[mod(currentIndex - 1, n)];
    const next = DEGRADATIONS[mod(currentIndex + 1, n)];

    enqueueWarm(cleanSrc(cur.id));
    enqueueWarm(degradedSrc(cur.id, currentLevel));
    enqueueWarm(cleanSrc(prev.id));
    enqueueWarm(cleanSrc(next.id));
    enqueueWarm(degradedSrc(prev.id, currentLevel));
    enqueueWarm(degradedSrc(next.id, currentLevel));

    if (hasLevels(cur.id)) {
      if (currentLevel > 1) {
        enqueueWarm(degradedSrc(cur.id, currentLevel - 1));
      }
      if (currentLevel < 5) {
        enqueueWarm(degradedSrc(cur.id, currentLevel + 1));
      }
    }
  }

  function loadVideoElement(video, src, attempt) {
    const tries = attempt || 0;
    if (video.dataset.src === src && video.readyState >= 3 && !video.error) {
      return Promise.resolve();
    }

    video.dataset.src = src;
    video.src = src;
    video.load();

    return waitForReady(video)
      .then(function () {
        warmed.add(src);
      })
      .catch(function () {
        if (tries < 1) {
          delete video.dataset.src;
          video.removeAttribute('src');
          return loadVideoElement(video, src, tries + 1);
        }
        throw new Error('Failed to load video: ' + src);
      });
  }

  function setVideoSource(video, pane, src, gen) {
    if (video.dataset.src === src && video.readyState >= 3 && !video.error) {
      if (!isStale(gen)) {
        setPaneLoading(pane, false);
      }
      return Promise.resolve();
    }

    return loadVideoElement(video, src).then(function () {
      if (isStale(gen)) {
        return;
      }
      setPaneLoading(pane, false);
    });
  }

  function swapDegradedSource(src, anchorTime, gen) {
    const active = getActiveDegraded();
    if (active.dataset.src === src && active.readyState >= 3 && !active.error) {
      return seekAndWait(active, anchorTime);
    }

    const inactive = getInactiveDegraded();
    const wasPlaying = !cleanVideo.paused;

    return loadVideoElement(inactive, src)
      .then(function () {
        if (isStale(gen)) {
          return;
        }
        return seekAndWait(inactive, anchorTime);
      })
      .then(function () {
        if (isStale(gen)) {
          return;
        }
        if (wasPlaying) {
          playVideo(inactive);
        }
        // Avoid a brief black frame: only switch once the new slot can render a frame.
        return waitForFirstFrame(inactive);
      })
      .then(function () {
        if (isStale(gen)) {
          return;
        }
        activateSlot(activeSlot === 'a' ? 'b' : 'a');
      });
  }

  function scheduleLoadPair(options) {
    clearTimeout(loadPairTimer);
    const opts = options || {};
    if (initialLoad) {
      initialLoad = false;
      return loadPair(opts);
    }
    return new Promise(function (resolve) {
      loadPairTimer = setTimeout(function () {
        resolve(loadPair(opts));
      }, 80);
    });
  }

  function loadPair(options) {
    const opts = options || {};
    const gen = bumpLoadGeneration();
    const current = DEGRADATIONS[currentIndex];
    const clean = cleanSrc(current.id);
    const degraded = degradedSrc(current.id, currentLevel);
    const anchor = opts.preserveTime ? getMasterTime() : 0;

    if (opts.reloadClean === false) {
      if (!opts.preserveTime) {
        pausePair();
      }
      return swapDegradedSource(degraded, anchor, gen).then(function () {
        if (isStale(gen)) {
          return;
        }
        if (!opts.preserveTime) {
          syncBothTo(0);
          playPair();
        }
      });
    }

    pausePair();
    setPaneLoading(cleanPane, true);
    setPaneLoading(degradedPane, true);

    return setVideoSource(cleanVideo, cleanPane, clean, gen)
      .then(function () {
        if (isStale(gen)) {
          return;
        }
        return swapDegradedSource(degraded, anchor, gen);
      })
      .then(function () {
        if (isStale(gen)) {
          return;
        }
        playPair();
        prefetchNeighbors();
      })
      .catch(function () {
        if (!isStale(gen)) {
          playPair();
        }
      })
      .finally(function () {
        if (!isStale(gen)) {
          setPaneLoading(cleanPane, false);
          setPaneLoading(degradedPane, false);
        }
      });
  }

  function updateUI() {
    const current = DEGRADATIONS[currentIndex];
    typeTitle.textContent = current.label;
    const showLevels = hasLevels(current.id);
    if (levelControl) {
      levelControl.classList.toggle('is-hidden', !showLevels);
    }
    if (typeNote) {
      if (current.id === 'motion_blur') {
        typeNote.textContent =
          'Note: Motion blur is synthesized from the input frame rate; a single severity is shown (no level slider).';
        typeNote.style.display = '';
      } else {
        typeNote.textContent = '';
        typeNote.style.display = 'none';
      }
    }
    if (showLevels) {
      levelSlider.value = String(currentLevel);
      levelValue.textContent = String(currentLevel);
    }

    typeButtons.querySelectorAll('.dg-type-btn').forEach(function (btn, idx) {
      const isActive = idx === currentIndex;
      btn.classList.toggle('is-active', isActive);
      btn.setAttribute('aria-selected', isActive ? 'true' : 'false');
    });

    return scheduleLoadPair({ preserveTime: false, reloadClean: true });
  }

  function setType(index) {
    currentIndex = mod(index, DEGRADATIONS.length);
    updateUI();
  }

  function setLevel(level) {
    const current = DEGRADATIONS[currentIndex];
    if (!hasLevels(current.id)) {
      return;
    }
    currentLevel = Math.min(5, Math.max(1, level));
    levelValue.textContent = String(currentLevel);
    levelSlider.value = String(currentLevel);
    loadPair({ preserveTime: true, reloadClean: false });
  }

  function bindDegradedEvents(video) {
    video.addEventListener('seeked', function () {
      if (!video.classList.contains('is-active')) {
        return;
      }
      endProgrammatic();
      resumeDegradedIfNeeded();
    });
    video.addEventListener('waiting', function () {
      if (!video.classList.contains('is-active')) {
        return;
      }
      resumeDegradedIfNeeded();
    });
    video.addEventListener('stalled', function () {
      if (!video.classList.contains('is-active')) {
        return;
      }
      resumeDegradedIfNeeded();
    });
  }

  cleanVideo.addEventListener('seeking', function () {
    seekDegradedTo(cleanVideo.currentTime);
  });
  cleanVideo.addEventListener('seeked', function () {
    programmaticDepth = 0;
    resumeDegradedIfNeeded();
  });
  cleanVideo.addEventListener('play', function () {
    syncDegradedToClean();
    playVideo(getActiveDegraded());
    drainWarmQueue();
  });
  cleanVideo.addEventListener('pause', function () {
    if (isProgrammatic()) {
      return;
    }
    // Keep degraded video playing even if clean pauses.
    drainWarmQueue();
  });

  bindDegradedEvents(degradedA);
  bindDegradedEvents(degradedB);

  DEGRADATIONS.forEach(function (item, index) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'dg-type-btn';
    btn.textContent = item.label;
    btn.setAttribute('role', 'tab');
    btn.setAttribute('aria-selected', index === 0 ? 'true' : 'false');
    btn.addEventListener('click', function () {
      setType(index);
    });
    typeButtons.appendChild(btn);
  });

  prevBtn.addEventListener('click', function () {
    setType(currentIndex - 1);
  });

  nextBtn.addEventListener('click', function () {
    setType(currentIndex + 1);
  });

  levelSlider.addEventListener('input', function () {
    const level = parseInt(levelSlider.value, 10);
    levelValue.textContent = String(level);
    clearTimeout(levelInputTimer);
    levelInputTimer = setTimeout(function () {
      setLevel(level);
    }, 120);
  });

  levelSlider.addEventListener('change', function () {
    clearTimeout(levelInputTimer);
    setLevel(parseInt(levelSlider.value, 10));
  });

  document.addEventListener('keydown', function (event) {
    if (!viewerVisible) {
      return;
    }
    if (event.key === 'ArrowLeft') {
      setType(currentIndex - 1);
    } else if (event.key === 'ArrowRight') {
      setType(currentIndex + 1);
    }
  });

  if ('IntersectionObserver' in window) {
    const observer = new IntersectionObserver(
      function (entries) {
        entries.forEach(function (entry) {
          viewerVisible = entry.isIntersecting;
          if (viewerVisible) {
            clearTimeout(pauseOnHideTimer);
            prefetchNeighbors();
            playPair();
          }
          else {
            clearTimeout(pauseOnHideTimer);
            pauseOnHideTimer = setTimeout(function () {
              if (!viewerVisible || document.hidden) {
                pausePair();
              }
            }, 180);
          }
        });
      },
      { rootMargin: '200px 0px' }
    );
    observer.observe(viewerRoot);
  } else {
    viewerVisible = true;
  }

  document.addEventListener('visibilitychange', function () {
    if (document.hidden) {
      pausePair();
      return;
    }
    if (viewerVisible) {
      playPair();
    }
  });

  activateSlot('a');
  updateUI();
})();
