/* ============================================================
   Brick for Brains — Dashboard JavaScript
   ============================================================ */

(function () {
  'use strict';

  /* --- Mobile sidebar toggle --- */
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

  /* --- Active nav link highlighting --- */
  function initActiveNav() {
    var path = window.location.pathname.split('/').pop() || 'index.html';
    var links = document.querySelectorAll('.sidebar__nav a');
    links.forEach(function (link) {
      var href = link.getAttribute('href');
      if (href === path || (href === 'index.html' && (path === '' || path === '/'))) {
        link.classList.add('active');
      }
    });
  }

  /* --- FRD Table Filtering --- */
  function initFRDFilters() {
    var categoryFilter = document.getElementById('filter-category');
    var priorityFilter = document.getElementById('filter-priority');
    var searchInput = document.getElementById('filter-search');
    var table = document.getElementById('frd-table');

    if (!table) return;

    var rows = table.querySelectorAll('tbody tr.fr-row');
    var detailRows = table.querySelectorAll('tbody tr.fr-detail');
    var countEl = document.getElementById('filter-count');

    function applyFilters() {
      var cat = categoryFilter ? categoryFilter.value : 'all';
      var pri = priorityFilter ? priorityFilter.value : 'all';
      var search = searchInput ? searchInput.value.toLowerCase().trim() : '';
      var visible = 0;

      rows.forEach(function (row, i) {
        var rowCat = row.getAttribute('data-category') || '';
        var rowPri = row.getAttribute('data-priority') || '';
        var rowText = row.textContent.toLowerCase();

        var matchCat = (cat === 'all' || rowCat === cat);
        var matchPri = (pri === 'all' || rowPri === pri);
        var matchSearch = (!search || rowText.indexOf(search) !== -1);

        var show = matchCat && matchPri && matchSearch;
        row.style.display = show ? '' : 'none';

        // Also hide corresponding detail row
        if (detailRows[i]) {
          if (!show) {
            detailRows[i].style.display = 'none';
            detailRows[i].classList.remove('open');
          } else {
            // Keep detail visibility as-is (user toggled)
            if (!detailRows[i].classList.contains('open')) {
              detailRows[i].style.display = 'none';
            }
          }
        }

        if (show) visible++;
      });

      if (countEl) {
        countEl.textContent = visible;
      }
    }

    if (categoryFilter) categoryFilter.addEventListener('change', applyFilters);
    if (priorityFilter) priorityFilter.addEventListener('change', applyFilters);
    if (searchInput) searchInput.addEventListener('input', applyFilters);

    // Category tab buttons
    var tabBtns = document.querySelectorAll('.tab-btn[data-category]');
    tabBtns.forEach(function (btn) {
      btn.addEventListener('click', function () {
        tabBtns.forEach(function (b) { b.classList.remove('active'); });
        btn.classList.add('active');
        if (categoryFilter) {
          categoryFilter.value = btn.getAttribute('data-category');
          applyFilters();
        }
      });
    });

    // Initial count
    applyFilters();
  }

  /* --- FRD row expand/collapse --- */
  function initFRDExpand() {
    var table = document.getElementById('frd-table');
    if (!table) return;

    table.addEventListener('click', function (e) {
      var row = e.target.closest('tr.fr-row');
      if (!row) return;

      var detailId = row.getAttribute('data-detail');
      if (!detailId) return;

      var detailRow = document.getElementById(detailId);
      if (!detailRow) return;

      var chevron = row.querySelector('.chevron');

      if (detailRow.classList.contains('open')) {
        detailRow.classList.remove('open');
        detailRow.style.display = 'none';
        if (chevron) chevron.classList.remove('open');
      } else {
        detailRow.classList.add('open');
        detailRow.style.display = '';
        if (chevron) chevron.classList.add('open');
      }
    });
  }

  /* --- PRD Table of Contents scroll spy --- */
  function initTocScrollSpy() {
    var tocLinks = document.querySelectorAll('.toc a');
    if (tocLinks.length === 0) return;

    var sections = [];
    tocLinks.forEach(function (link) {
      var id = link.getAttribute('href');
      if (id && id.startsWith('#')) {
        var el = document.getElementById(id.substring(1));
        if (el) sections.push({ link: link, el: el });
      }
    });

    if (sections.length === 0) return;

    function onScroll() {
      var scrollPos = window.scrollY + 120;
      var current = null;

      for (var i = sections.length - 1; i >= 0; i--) {
        if (sections[i].el.offsetTop <= scrollPos) {
          current = sections[i];
          break;
        }
      }

      tocLinks.forEach(function (l) { l.classList.remove('active'); });
      if (current) current.link.classList.add('active');
    }

    window.addEventListener('scroll', onScroll, { passive: true });
    onScroll();
  }

  /* --- Timestamp --- */
  function initTimestamp() {
    var el = document.getElementById('last-updated');
    if (el) {
      var now = new Date();
      el.textContent = now.toLocaleDateString('en-US', {
        year: 'numeric', month: 'long', day: 'numeric'
      }) + ' at ' + now.toLocaleTimeString('en-US', {
        hour: '2-digit', minute: '2-digit'
      });
    }
  }

  /* --- Init on DOMContentLoaded --- */
  document.addEventListener('DOMContentLoaded', function () {
    initSidebar();
    initActiveNav();
    initFRDFilters();
    initFRDExpand();
    initTocScrollSpy();
    initTimestamp();
  });

})();
