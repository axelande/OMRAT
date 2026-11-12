/* Collapsible sidebar navigation.
 *
 * conf.py writes the whole toctree into every page (sidebar_collapse = False).
 * This script puts an arrow in front of every entry that has headings
 * underneath, folds them all except the page being read, and remembers in
 * sessionStorage which ones the reader opened.  Without JavaScript the tree
 * stays fully open.
 */
(function () {
    'use strict';

    var STORE_KEY = 'omrat-docs-nav-open';

    function pageKey(link) {
        // Absolute URL without the #fragment; the current page links to "#".
        return link ? link.href.split('#')[0] : '';
    }

    function loadOpen() {
        try {
            return JSON.parse(window.sessionStorage.getItem(STORE_KEY) || '{}');
        } catch (e) {
            return {};
        }
    }

    function saveOpen(state) {
        try {
            window.sessionStorage.setItem(STORE_KEY, JSON.stringify(state));
        } catch (e) {
            /* storage blocked: the tree still works, it just forgets */
        }
    }

    function setOpen(li, button, open) {
        li.classList.toggle('nav-open', open);
        button.setAttribute('aria-expanded', open ? 'true' : 'false');
        button.title = open ? 'Hide the headings of this page'
                            : 'Show the headings of this page';
    }

    function init() {
        var wrapper = document.querySelector('div.sphinxsidebarwrapper');
        if (!wrapper) {
            return;
        }
        var state = loadOpen();
        var items = wrapper.querySelectorAll('li[class*="toctree-l"]');
        Array.prototype.forEach.call(items, function (li) {
            var sub = null;
            var link = null;
            Array.prototype.forEach.call(li.children, function (child) {
                if (child.tagName === 'UL' && !sub) { sub = child; }
                if (child.tagName === 'A' && !link) { link = child; }
            });
            li.classList.add('nav-item');
            if (!sub) {
                return;
            }
            li.classList.add('nav-has-children');
            var key = pageKey(link);
            var button = document.createElement('button');
            button.type = 'button';
            button.className = 'nav-toggle';
            if (link) {
                button.setAttribute('aria-label',
                                    'Headings of ' + link.textContent);
            }
            var open = li.classList.contains('current') || state[key] === true;
            setOpen(li, button, open);
            button.addEventListener('click', function () {
                var nowOpen = !li.classList.contains('nav-open');
                setOpen(li, button, nowOpen);
                var current = loadOpen();
                if (nowOpen) {
                    current[key] = true;
                } else {
                    delete current[key];
                }
                saveOpen(current);
            });
            li.insertBefore(button, li.firstChild);
        });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
}());
