const MOBILE_QUERY = '(max-width: 960px)';

export function mountMobileUX({ sheetSel = '#bottomSheet', fabSel = '#fab', session } = {}) {
  const sheet = document.querySelector(sheetSel);
  const fab = document.querySelector(fabSel);
  const dock = document.getElementById('actionsDock');

  const media = window.matchMedia(MOBILE_QUERY);
  const onChange = (event) => {
    if (event.matches) enableMobile();
    else disableMobile();
  };

  const enableMobile = () => {
    if (!sheet) return;

    if (sheet.dataset.mobileBound === 'true') {
      if (sheet.dataset.userInteracted !== 'true') {
        setInitialTop(session, sheet);
      }
      updateSessionDecor();
      return;
    }

    sheet.dataset.mobileBound = 'true';
    sheet.classList.add('sheet');
    setInitialTop(session, sheet);

    let startY = 0;
    let startTop = 0;

    const onTouchStart = (event) => {
      if (!event.touches?.length) return;
      startY = event.touches[0].clientY;
      startTop = sheet.offsetTop;
      sheet.classList.add('sheet--dragging');
    };

    const onTouchMove = (event) => {
      if (!event.touches?.length) return;
      const dy = event.touches[0].clientY - startY;
      const next = Math.min(
        Math.max(startTop + dy, window.innerHeight * 0.2),
        window.innerHeight - 72
      );
      sheet.style.top = `${next}px`;
    };

    const onTouchEnd = () => {
      sheet.classList.remove('sheet--dragging');
      const threshold = window.innerHeight * 0.55;
      const open = sheet.offsetTop < threshold;
      sheet.dataset.state = open ? 'open' : 'collapsed';
      sheet.style.top = open ? '22vh' : 'calc(100vh - 80px)';
      sheet.dataset.userInteracted = 'true';
    };

    sheet.addEventListener('touchstart', onTouchStart, { passive: true });
    sheet.addEventListener('touchmove', onTouchMove, { passive: true });
    sheet.addEventListener('touchend', onTouchEnd);

    if (fab) {
      fab.classList.add('fab');
      fab.addEventListener('click', toggleDock);
    }

    updateSessionDecor();
  };

  const disableMobile = () => {
    if (sheet) {
      sheet.classList.remove('sheet', 'sheet--dragging');
      delete sheet.dataset.mobileBound;
       delete sheet.dataset.userInteracted;
      sheet.style.top = '';
      sheet.dataset.state = '';
    }
    if (fab) {
      fab.classList.remove('fab', 'fab-muted');
      fab.removeEventListener('click', toggleDock);
    }
    if (dock) {
      dock.classList.remove('visible', 'after-hours');
      dock.style.display = '';
    }
  };

  const toggleDock = () => {
    if (!dock) return;
    dock.classList.toggle('visible');
  };

  const updateSessionDecor = () => {
    if (!dock) return;
    if (session === 'after') {
      dock.classList.add('after-hours');
      fab?.classList.add('fab-muted');
    } else {
      dock.classList.remove('after-hours');
      fab?.classList.remove('fab-muted');
    }
  };

  if (media.matches) enableMobile();
  media.addEventListener('change', onChange);
}

function setInitialTop(session, sheet) {
  if (!sheet) return;
  let top;
  if (session === 'premkt') {
    top = '35vh';
    sheet.dataset.state = 'open';
  } else if (session === 'after') {
    top = 'calc(100vh - 88px)';
    sheet.dataset.state = 'collapsed';
  } else {
    top = 'calc(100vh - 80px)';
    sheet.dataset.state = 'collapsed';
  }
  sheet.style.top = top;
}
